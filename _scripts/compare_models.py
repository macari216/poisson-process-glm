import os
import jax
import jax.numpy as jnp
import nemos as nmo
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from nemos.tree_utils import tree_l2_norm, tree_sub

from time import perf_counter

import paglm.core as paglm

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import ContinuousPA, ContinuousMC
from poisson_point_process.poisson_process_obs_model import PolynomialApproximation, MonteCarloApproximation
from poisson_point_process.utils import quadratic
from poisson_point_process.basis import RaisedCosineLogEval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# generate data
n_neurons = 5
history_window = 0.004
tot_time_sec = 1000
test_time_sec = 100
binsize = 0.0001
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
# inverse_link = jax.nn.softplus
# link_f = lambda x: jnp.log(jnp.exp(x) -1.)
inverse_link = jnp.exp
link_f = jnp.log

rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window

basis_fn = RaisedCosineLogEval(history_window, n_basis_funcs)

kfold = 100
times = np.zeros((kfold, 4))
scores = np.zeros((kfold, 4))
scores_mse = np.zeros((kfold, 4))

# order: EDb, PAD, MC, PAC, (EDf)

for k in range(kfold):
    print(k)
    np.random.seed(123 + k)

    rc_basis_sim = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size * 2)
    _, kernels_sim = rc_basis.evaluate_on_grid(window_size * 2)

    pres_rate_per_sec = 6
    posts_rate_per_sec = 4
    # rescaled proportionally to the binsize
    bias_true = posts_rate_per_sec + np.log(binsize)
    weights_true = np.random.normal(0, 0.3, n_neurons * n_basis_funcs)
    X, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize / 2,
                                                               n_bins_tot * 2, n_neurons, weights_true, window_size * 2,
                                                               kernels_sim,
                                                               inverse_link)

    print(f"mean spikes per neuron: {jnp.mean(jnp.hstack((y_counts[:, None], X_counts)).sum(0))}")
    print(f"max spikes per bin: {jnp.hstack((y_counts[:, None], X_counts)).max()}")
    print(f"ISI (ms): {tot_time_sec * 1000 / jnp.mean(jnp.hstack((y_counts[:, None], X_counts)).sum(0))}")

    spike_times, spike_ids = simulate.poisson_times(X_counts, tot_time_sec, binsize / 2)
    spike_times_y, _ = simulate.poisson_times(y_counts[:, None], tot_time_sec, binsize / 2)
    X_spikes = jnp.vstack((spike_times, spike_ids))
    target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)
    y_spikes = jnp.vstack((spike_times_y, target_idx))

    sorted_spikes = [nap.Ts(np.array(X_spikes[0][X_spikes[1] == n] + binsize / 2)) for n in range(n_neurons)]
    sorted_spikes.append(nap.Ts(np.array(y_spikes[0] + binsize / 2)))
    spikes_tsgroup = nap.TsGroup({n: sorted_spikes[n] for n in range(n_neurons + 1)})
    all_counts = jnp.array(spikes_tsgroup.count(binsize, nap.IntervalSet(0, tot_time_sec)))

    X_discrete = all_counts[:, :-1]
    X_discrete = nmo.convolve.create_convolutional_predictor(kernels, X_discrete).reshape(-1, n_neurons * n_basis_funcs)
    y_discrete = all_counts[:, -1]
    X_discrete, y_discrete = X_discrete[window_size:], y_discrete[window_size:]

    X_discrete = jnp.concatenate((X_discrete, jnp.ones([X_discrete.shape[0], 1])), axis=1)

    # test set for evaluation
    X_test, y_test, X_counts_test, lam_posts_test = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec,
                                                                            binsize,
                                                                            int(test_time_sec / (binsize)), n_neurons,
                                                                            weights_true, window_size, kernels,
                                                                            inverse_link)
    test_times, test_ids = simulate.poisson_times(X_counts_test, test_time_sec, binsize)
    test_times_y, _ = simulate.poisson_times(y_test[:, None], test_time_sec, binsize)
    X_spikes_test = jnp.vstack((test_times, test_ids))
    test_idx = jnp.searchsorted(test_times, test_times_y)
    y_spikes_test = jnp.vstack((test_times_y, test_idx))

    # select interval
    interval = [np.percentile(link_f(lam_posts / (binsize / 2)), 2),
                np.percentile(link_f(lam_posts / (binsize / 2)), 99.5)]
    interval_discrete = [np.percentile(link_f(lam_posts * 2), 2), np.percentile(link_f(lam_posts * 2), 99.5)]

    # POLYNOMIAL APPROXIMATION CONTINUOUS
    obs_model_pa = PolynomialApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        window_size=80,
        approx_interval=interval,
        eval_function=basis_fn,
    )

    tt0 = perf_counter()
    model_pa = ContinuousPA(
        solver_name="LBFGS",
        observation_model=obs_model_pa,
        solver_kwargs={"tol": 1e-12}
    ).fit_closed_form(X_spikes, y_spikes)
    times[k, 3] = perf_counter() - tt0

    obs_model_exact = nmo.observation_models.PoissonObservations(inverse_link)

    # MONTE CARLO
    obs_model = MonteCarloApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        eval_function=basis_fn,
        mc_n_samples=y_spikes.shape[1] * 5,
    )
    model_mc = ContinuousMC(
        solver_name="GradientDescent",
        observation_model=obs_model,
        random_key=jax.random.PRNGKey(0),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 1e-5})
    num_iter = 3000
    tt0 = perf_counter()
    params = model_mc.initialize_params(X_spikes, y_spikes)
    state = model_mc.initialize_state(X_spikes, y_spikes, params)
    for step in range(num_iter):
        params, state = model_mc.update(params, state, X_spikes, y_spikes)
        if step % 300 == 0:
            print(f"step {step}")
    times[k, 2] = perf_counter() - tt0


    # BATCHED POLYNOMIAL APPROXIMATION DISCRETE
    def sufficient_stats(X, y, n_batches):
        T = y.shape[0]
        batch_size = int(np.ceil(T / n_batches))
        append_X = np.zeros((-T % batch_size, X.shape[1]))
        append_y = np.zeros(-T % batch_size)
        X, y = np.vstack((X, append_X)), np.hstack((y, append_y))

        sum_x = np.sum(X, axis=0)

        sum_yx = sum(
            np.sum((y[i: i + batch_size])[:, np.newaxis] * X[i: i + batch_size], axis=0)
            for i in range(0, T, batch_size)
        )
        sum_xxT = sum(
            X[i: i + batch_size].T @ X[i: i + batch_size]
            for i in range(0, T, batch_size)
        )
        sym_yxxT = sum(
            X[i: i + batch_size].T @ ((y[i: i + batch_size])[:, np.newaxis] * X[i: i + batch_size])
            for i in range(0, T, batch_size)
        )

        return [sum_x, sum_yx, sum_xxT, sym_yxxT]


    tt0 = perf_counter()
    suff_discrete = sufficient_stats(X_discrete, y_discrete, 10)
    weights_d, interval_d = paglm.fit_paglm(inverse_link, suff_discrete, [interval_discrete])
    times[k, 1] = perf_counter() - tt0

    # SCORES
    # pred_exact = model_exact.predict(X)
    # pred_exact_fit = model_fit.predict(X_test)
    pred_paglm = quadratic(np.dot(X_test, weights_d[:-1]) + weights_d[-1], inverse_link, interval_discrete)
    pred_mc = model_mc.predict(X_spikes_test, binsize, test_time_sec)
    pred_pa = model_pa.predict(X_spikes_test, binsize, test_time_sec)

    # scores[k,0] = obs_model_exact.log_likelihood(y_counts, pred_exact)
    scores[k, 1] = obs_model_exact.pseudo_r2(y_test, pred_paglm)
    scores[k, 2] = obs_model_exact.pseudo_r2(y_test, pred_mc * binsize)
    scores[k, 3] = obs_model_exact.pseudo_r2(y_test, pred_pa * binsize)

    weights_paglm, bias_paglm = weights_d[:-1].reshape(n_neurons, n_basis_funcs), weights_d[-1]
    weights_pa, bias_pa = model_pa.coef_.reshape(-1, n_basis_funcs), model_pa.intercept_
    weights_mc, bias_mc = model_mc.coef_.reshape(-1, n_basis_funcs), model_mc.intercept_

    filters_mc = np.dot(weights_mc, kernels.T) + bias_mc
    filters_pa = np.dot(weights_pa, kernels.T) + bias_pa
    filters_paglm = np.dot(weights_paglm, kernels.T) + bias_paglm
    filters_true = np.dot(weights_true.reshape(-1, n_basis_funcs), kernels.T) + bias_true

    filters_mc = inverse_link(filters_mc)
    filters_pa = quadratic(filters_pa, inverse_link, interval)
    filters_paglm = quadratic(filters_paglm, inverse_link, interval_discrete) / binsize
    filters_true = inverse_link(filters_true) / binsize

    scores_mse[k, 1] = jnp.mean(jnp.square(filters_true - filters_paglm))
    scores_mse[k, 2] = jnp.mean(jnp.square(filters_true - filters_mc))
    scores_mse[k, 3] = jnp.mean(jnp.square(filters_true - filters_pa))

results = {"times": times, "scores": scores, "scores_mse": scores_mse}

np.savez(f"/mnt/home/amedvedeva/ceph/time_perf_mm.npz", **results)
print("Script terminated")
