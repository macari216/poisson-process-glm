import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import nemos as nmo
import pynapple as nap
import numpy as np

from time import perf_counter

import paglm.core as paglm

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import ContinuousPA, ContinuousMC
from poisson_point_process.poisson_process_obs_model import PolynomialApproximation, MonteCarloApproximation
from poisson_point_process.utils import quadratic
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral, RaisedCosineLogEval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

# generate data
n_neurons = 5
history_window = 0.004
n_times = 6
tot_time_sec = np.logspace(1, 4, n_times)
test_time_sec = np.logspace(0, 3, n_times)
binsize = 0.0001
window_size = int(history_window / binsize)
n_basis_funcs = 100
n_bins_tot = (tot_time_sec / binsize).astype(int)
n_batches_scan = 1
# inverse_link = jax.nn.softplus
# link_f = lambda x: jnp.log(jnp.exp(x) -1.)
inverse_link = jnp.exp
link_f = jnp.log

step_sizes = [7e-4, 2e-3, 1e-3, 2e-4, 2e-4, 4e-5]

num_fun_l = 4
basis_fn_l = GenLaguerreEval(history_window, num_fun_l)
kernels_l = basis_fn_l(-np.linspace(0, history_window, window_size))
# kernels_sim = basis_fn_l(-np.linspace(0, history_window, window_size * 2))

kernels = kernels_l

basis_rc = RaisedCosineLogEval(history_window, n_basis_funcs, width=45., time_scaling=20.)
kernels_sim = basis_rc(-np.linspace(0, history_window, window_size * 2))
kernels_rc = basis_rc(-np.linspace(0, history_window, window_size))

kfold = 10
times = np.zeros((kfold, n_times, 5))
scores = np.zeros((kfold, n_times, 5))
scores_mse = np.zeros((kfold, n_times, 5))
scores_ll = np.zeros((kfold, n_times, 5))

# order: EDb, PAD, MC, PAC, H

for k in range(kfold):
    np.random.seed(123 + k)

    pres_rate_per_sec = 15
    posts_rate_per_sec = 3
    # rescaled proportionally to the binsize
    bias_true = np.log(posts_rate_per_sec) + np.log(binsize / 2)
    # posts_rate_sim = inverse_link(posts_rate_per_sec + np.log(binsize)) / binsize
    weights_true = np.random.normal(0, 0.2, n_neurons * n_basis_funcs)

    T = int(n_bins_tot[-1] * 2)
    # y_counts = np.zeros(T)
    lam_posts = np.zeros(T)
    # X_counts = np.zeros((T, n_neurons))
    batch_size = int(T / 30)

    tt0 = perf_counter()
    with jax.default_device(cpu):
        spike_times = []
        spike_times_y = []
        spike_ids = []
        i = 0
        for s in range(0, T, batch_size):
            t0 = perf_counter()
            bsize = min(batch_size, T - s)
            _, y_c, X_c, fr = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize / 2,
                                                      bsize, n_neurons, weights_true, window_size * 2, kernels_sim,
                                                      inverse_link)
            lam_posts[s:s + bsize] = fr

            st_X, sid_X = simulate.poisson_times(X_c, bsize * (binsize / 2), binsize / 2)
            st_y, _ = simulate.poisson_times(y_c[:, None], bsize * (binsize / 2), binsize / 2)

            spike_times.append(st_X + (s * (binsize / 2)))
            spike_times_y.append(st_y + (s * (binsize / 2)))
            spike_ids.append(sid_X)
            i += 1
            print(f"{i}, {perf_counter() - t0}")

        spike_times = jnp.concatenate(spike_times)
        spike_times_y = jnp.concatenate(spike_times_y)
        spike_ids = jnp.concatenate(spike_ids)

        X_spikes = jnp.vstack((spike_times, spike_ids))
        target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)
        y_spikes = jnp.vstack((spike_times_y, target_idx))

        sorted_spikes = [nap.Ts(np.array(X_spikes[0][X_spikes[1] == n] + binsize / 2)) for n in range(n_neurons)]
        sorted_spikes.append(nap.Ts(np.array(y_spikes[0] + binsize / 2)))
        spikes_tsgroup = nap.TsGroup({n: sorted_spikes[n] for n in range(n_neurons + 1)})
        all_counts = jnp.array(spikes_tsgroup.count(binsize, nap.IntervalSet(0, tot_time_sec[-1])))

        X_discrete = all_counts[:, :-1]
        y_discrete = all_counts[:, -1]

    # test set for evaluation
    with jax.default_device(cpu):
        T = int((test_time_sec[-1] + history_window) / binsize)
        batch_size = int(T / 10)

        test_times = []
        # test_times_y = []
        test_ids = []
        y_test = np.zeros(T)
        lam_posts_test = np.zeros(T)
        X_counts_test = np.zeros((T, n_neurons))
        for s in range(0, T, batch_size):
            bsize = min(batch_size, T - s)
            _, y_ct, X_ct, fr_t = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                                          bsize, n_neurons, weights_true, window_size, kernels_rc,
                                                          inverse_link)
            y_test[s:s + bsize] = y_ct
            lam_posts_test[s:s + bsize] = fr_t
            X_counts_test[s:s + bsize] = X_ct

            st_Xt, sid_Xt = simulate.poisson_times(X_ct, bsize * (binsize), binsize)
            # st_yt, _ = simulate.poisson_times(y_ct[:, None], bsize * (binsize), binsize)

            test_times.append(st_Xt + (s * (binsize)))
            # test_times_y.append(st_yt + (s * (binsize)))
            test_ids.append(sid_Xt)

        test_times = jnp.concatenate(test_times)
        # test_times_y = jnp.concatenate(test_times_y)
        test_ids = jnp.concatenate(test_ids)

        X_spikes_test = jnp.vstack((test_times[test_times > history_window], test_ids[test_times > history_window]))
        # target_idx_test = jnp.searchsorted(test_times, test_times_y)
        # y_spikes_test = jnp.vstack((test_times_y, target_idx_test))

        X_test = nmo.convolve.create_convolutional_predictor(kernels, X_counts_test).reshape(-1, n_neurons * num_fun_l)[
                 window_size:]
        y_test, lam_posts_test = jnp.array(y_test)[window_size:], lam_posts_test[window_size:]

        X_spikes_test = X_spikes_test.at[0].set(X_spikes_test[0] - history_window)

    print(f"generated data {perf_counter() - tt0}")

    for tm in range(n_times):
        X_spikes_tm = X_spikes[:, X_spikes[0] < tot_time_sec[tm]]
        X_spikes_test_tm = X_spikes_test[:, X_spikes_test[0] < test_time_sec[tm]]
        y_spikes_tm = y_spikes[:, y_spikes[0] < tot_time_sec[tm]]
        X_discrete_tm = X_discrete[:n_bins_tot[tm]]
        y_discrete_tm = y_discrete[:n_bins_tot[tm]]
        X_counts_test_tm = X_counts_test[:int(test_time_sec[tm] / binsize) + window_size]
        y_test_tm = y_test[:int(test_time_sec[tm] / binsize)]
        X_test_tm = X_test[:int(test_time_sec[tm] / binsize)]
        lam_posts_tm = lam_posts[:n_bins_tot[tm]]

        # select interval
        interval = [np.percentile(link_f(lam_posts / (binsize / 2)), 2.5),
                    np.percentile(link_f(lam_posts / (binsize / 2)), 99.5)]
        interval_discrete = [np.percentile(link_f(lam_posts * 2), 2.5), np.percentile(link_f(lam_posts * 2), 99.5)]

        # POLYNOMIAL APPROXIMATION CONTINUOUS
        obs_model_pa = PolynomialApproximation(
            inverse_link_function=inverse_link,
            n_basis_funcs=num_fun_l,
            n_batches_scan=n_batches_scan,
            history_window=history_window,
            approx_interval=interval,
            eval_function=GenLaguerreEval(history_window, num_fun_l),
            int_function=GenLaguerreInt(history_window, num_fun_l),
            prod_int_function=GenLaguerreProdIntegral(history_window, num_fun_l),
        )

        tt0 = perf_counter()
        model_pa = ContinuousPA(
            solver_name="LBFGS",
            observation_model=obs_model_pa,
            solver_kwargs={"tol": 1e-12}
        ).fit_closed_form(X_spikes_tm, y_spikes_tm)
        times[k, tm, 3] = perf_counter() - tt0

        # MONTE CARLO
        obs_model = MonteCarloApproximation(
            inverse_link_function=inverse_link,
            n_basis_funcs=num_fun_l,
            n_batches_scan=n_batches_scan,
            history_window=history_window,
            eval_function=GenLaguerreEval(history_window, num_fun_l),
            mc_n_samples=jnp.minimum(n_bins_tot[tm], 500000),
        )
        model_mc = ContinuousMC(
            solver_name="GradientDescent",
            observation_model=obs_model,
            random_key=jax.random.PRNGKey(0),
            solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": step_sizes[tm]})
        num_iter = 3500
        # error = np.zeros(num_iter)
        tt0 = perf_counter()
        params = model_mc.initialize_params(X_spikes_tm, y_spikes_tm)
        state = model_mc.initialize_state(X_spikes_tm, y_spikes_tm, params)
        for step in range(num_iter):
            params, state = model_mc.update(params, state, X_spikes_tm, y_spikes_tm)
            # error[step] = state.error
            if step % 300 == 0:
                # print(f"step {step}, error {error[step]}")
                print(f"step {step}")
        times[k, tm, 2] = perf_counter() - tt0

        # HYBRID
        obs_model_h = MonteCarloApproximation(
            inverse_link_function=inverse_link,
            n_basis_funcs=num_fun_l,
            n_batches_scan=n_batches_scan,
            history_window=history_window,
            eval_function=basis_fn_l,
            mc_n_samples=jnp.minimum(int(tot_time_sec[tm] * 2000), 500000),
        )

        model_mc_h = ContinuousMC(
            solver_name="GradientDescent",
            observation_model=obs_model_h,
            random_key=jax.random.PRNGKey(0),
            solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
        num_iter = 1000
        # error_h = np.zeros(num_iter)
        tt0 = perf_counter()
        init_params = (model_pa.coef_, jnp.atleast_1d(model_pa.intercept_))
        params = model_mc_h.initialize_params(X_spikes_tm, y_spikes_tm, init_params=init_params)
        state = model_mc_h.initialize_state(X_spikes_tm, y_spikes_tm, params)
        for step in range(num_iter):
            # t0 = perf_counter()
            params, state = model_mc_h.update(params, state, X_spikes_tm, y_spikes_tm)
            # error_h[step] = state.error
            if step % 300 == 0:
                # print(f"step {step}, error {error_h[step]}, time {perf_counter() - t0}")
                print(f"step {step}")
        times[k, tm, 4] = (perf_counter() - tt0) + times[k, tm, 3]


        # BATCHED POLYNOMIAL APPROXIMATION DISCRETE
        def sufficient_stats(X, y, kernels, n_batches, ws):
            n_fun = kernels.shape[1]
            n_neurons = X.shape[1]
            N = n_fun * n_neurons
            T = y.shape[0]
            batch_size = int(np.ceil(T / n_batches))

            sum_x = np.zeros(N + 1)
            sum_yx = np.zeros(N + 1)
            sum_xxT = np.zeros((N + 1, N + 1))
            sum_yxxT = np.zeros((N + 1, N + 1))

            for i in range(0, T, batch_size):
                yb = y[i: i + batch_size + ws]
                Xb = X[i: i + batch_size + ws]

                Xb = nmo.convolve.create_convolutional_predictor(kernels, Xb).reshape(-1, N)
                Xb = np.concatenate((Xb, np.ones([Xb.shape[0], 1])), axis=1)

                yb, Xb = yb[~np.isnan(Xb).any(axis=1)], Xb[~np.isnan(Xb).any(axis=1)]

                sum_x += np.sum(Xb, axis=0)
                sum_yx += np.sum(yb[:, np.newaxis] * Xb, axis=0)
                sum_xxT += Xb.T @ Xb
                sum_yxxT += Xb.T @ (yb[:, np.newaxis] * Xb)

            return [sum_x, sum_yx, sum_xxT, sum_yxxT]


        tt0 = perf_counter()
        with jax.default_device(cpu):
            suff_discrete = sufficient_stats(X_discrete_tm, y_discrete_tm, kernels, 20, window_size)
        weights_d, interval_d = paglm.fit_paglm(inverse_link, suff_discrete, [interval_discrete])
        times[k, tm, 1] = perf_counter() - tt0

        # SCORES
        # pred_exact = model_exact.predict(X)
        # pred_exact_fit = model_fit.predict(X_test)
        pred_paglm = quadratic(np.dot(X_test_tm, weights_d[:-1]) + weights_d[-1], inverse_link, interval_discrete)
        pred_mc = model_mc.predict(X_spikes_test_tm, binsize, test_time_sec[tm])
        pred_pa = model_pa.predict(X_spikes_test_tm, binsize, test_time_sec[tm])
        pred_h = model_mc_h.predict(X_spikes_test_tm, binsize, test_time_sec[tm])

        obs_model_exact = nmo.observation_models.PoissonObservations(inverse_link)

        # scores[k,0] = obs_model_exact.log_likelihood(y_counts, pred_exact)
        scores[k, tm, 1] = obs_model_exact.pseudo_r2(y_test_tm, pred_paglm)
        scores[k, tm, 2] = obs_model_exact.pseudo_r2(y_test_tm, pred_mc * binsize)
        scores[k, tm, 3] = obs_model_exact.pseudo_r2(y_test_tm, pred_pa * binsize)
        scores[k, tm, 4] = obs_model_exact.pseudo_r2(y_test_tm, pred_h * binsize)

        scores_ll[k, tm, 1] = obs_model_exact.log_likelihood(y_test_tm, pred_paglm)
        scores_ll[k, tm, 2] = obs_model_exact.log_likelihood(y_test_tm, pred_mc * binsize)
        scores_ll[k, tm, 3] = obs_model_exact.log_likelihood(y_test_tm, pred_pa * binsize)
        scores_ll[k, tm, 4] = obs_model_exact.log_likelihood(y_test_tm, pred_h * binsize)

        weights_paglm, bias_paglm = weights_d[:-1].reshape(n_neurons, num_fun_l), weights_d[-1]
        weights_pa, bias_pa = model_pa.coef_.reshape(-1, num_fun_l), model_pa.intercept_
        weights_mc, bias_mc = model_mc.coef_.reshape(-1, num_fun_l), model_mc.intercept_
        weights_h, bias_h = model_mc_h.coef_.reshape(-1, num_fun_l), model_mc_h.intercept_

        filters_mc = np.dot(weights_mc, kernels.T) + bias_mc
        filters_pa = np.dot(weights_pa, kernels.T) + bias_pa
        filters_h = np.dot(weights_h, kernels.T) + bias_h
        filters_paglm = np.dot(weights_paglm, kernels.T) + bias_paglm
        filters_true = np.dot(weights_true.reshape(-1, n_basis_funcs), kernels_rc.T) + np.log(
            posts_rate_per_sec) + np.log(binsize)

        filters_mc = inverse_link(filters_mc)
        filters_pa = quadratic(filters_pa, inverse_link, interval)
        filters_h = inverse_link(filters_h)
        filters_paglm = quadratic(filters_paglm, inverse_link, interval_discrete) / binsize
        filters_true = inverse_link(filters_true) / binsize

        scores_mse[k, tm, 1] = jnp.mean(jnp.square(filters_true - filters_paglm))
        scores_mse[k, tm, 2] = jnp.mean(jnp.square(filters_true - filters_mc))
        scores_mse[k, tm, 3] = jnp.mean(jnp.square(filters_true - filters_pa))
        scores_mse[k, tm, 4] = jnp.mean(jnp.square(filters_true - filters_h))

results = {"times": times, "scores": scores, "scores_mse": scores_mse, "scores_ll": scores_ll}

np.savez(f"/mnt/home/amedvedeva/ceph/time_perf_alltoone.npz", **results)
print("Script terminated")