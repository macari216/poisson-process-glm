import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from time import perf_counter
from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np

import nemos as nmo

import pynapple as nap

import pickle

import paglm.core as paglm

from poisson_point_process.poisson_process_glm import PopulationContinuousPA, PopulationContinuousMC
from poisson_point_process.poisson_process_obs_model import PopulationPolynomialApproximation, MonteCarloApproximation
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral
from poisson_point_process.utils import quadratic

from nemos.tree_utils import tree_l2_norm, tree_sub

# jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]


# load data
with open("/mnt/home/amedvedeva/ceph/saved_data/spike_times_by_unit.pkl", "rb") as f:
    spike_times_by_unit = pickle.load(f)
invalid_times = np.load("/mnt/home/amedvedeva/ceph/saved_data/invalid_times.npy")
invalid_times = nap.IntervalSet(invalid_times[:,0], invalid_times[:,1])

spikes_tsgroup = nap.TsGroup({
    i: nap.Ts(data)
    for i, (_, data) in enumerate(islice(spike_times_by_unit.items(), 127))
})
valid_times = spikes_tsgroup.time_support.set_diff(invalid_times[1:])

spikes_tsgroup = spikes_tsgroup.restrict(valid_times)
spikes_tsgroup = nap.TsGroup({key: spikes_tsgroup.data[i] for key, i in enumerate(spikes_tsgroup.index)})


kfolds = 5

train_duration = 100
test_duration = 100

rec_interval = nap.IntervalSet(valid_times.start[0],  valid_times.end[-1])

train_intervals = [nap.IntervalSet(
    rec_interval.start[0] + i * train_duration,
    rec_interval.start[0] + i * train_duration + train_duration
) for i in range(kfolds)]
test_intervals = [nap.IntervalSet(train_intervals[i].end[-1], train_intervals[i].end[-1]+test_duration) for i in range(kfolds)]

for k in range(kfolds):
    train_int = train_intervals[k].intersect(valid_times)
    spikes_tsgroup_100s = spikes_tsgroup.restrict(train_int).getby_threshold('rate', 1, op = '>').getby_threshold('rate', 25, op = '<')
    spikes_tsgroup_100s = nap.TsGroup({key: spikes_tsgroup_100s.data[i] for key, i in enumerate(spikes_tsgroup_100s.index)})
    spikes_tsgroup_full = spikes_tsgroup[spikes_tsgroup_100s.keys()]
    spikes_tsgroup_full = nap.TsGroup({key: spikes_tsgroup_full.data[i] for key, i in enumerate(spikes_tsgroup_full.index)})

    n_neurons = len(spikes_tsgroup_100s)
    inverse_link = jnp.exp
    link_f = jnp.log
    history_window = 0.006
    binsize = 0.0001
    window_size = int(history_window / binsize)
    n_basis_funcs = 4
    c = 2.0
    # c = 1.5
    basis = GenLaguerreEval(history_window, n_basis_funcs, c=c)
    kernels = basis(-jnp.linspace(0, history_window, window_size))

    spikes_train = spikes_tsgroup_100s.restrict(train_int)
    spikes_n = jnp.array([spikes_train[n].shape[0] for n in range(n_neurons)])
    spike_ids = jnp.repeat(jnp.arange(n_neurons), spikes_n)
    spike_times = jnp.concatenate([spikes_train.data[n].t for n in range(n_neurons)])
    sorted_indices = jnp.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    spike_ids = spike_ids[sorted_indices]
    X_spikes = jnp.vstack((spike_times, spike_ids))
    y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))

    test_int = test_intervals[k].intersect(valid_times)
    spikes_test = spikes_tsgroup_full.restrict(test_int.intersect(valid_times))
    spikes_n = jnp.array([spikes_test[n].shape[0] for n in range(n_neurons)])
    spike_ids = jnp.repeat(jnp.arange(n_neurons), spikes_n)
    spike_times = jnp.concatenate([spikes_test.data[n].t for n in range(n_neurons)])
    sorted_indices = jnp.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    spike_ids = spike_ids[sorted_indices]
    X_test = jnp.vstack((spike_times, spike_ids))

    y_test = jax.device_put(jnp.array(spikes_test.count(bin_size=binsize).to_numpy()), device=cpu)

    bounds = [-0.2, 0.5]
    mean_rates = spikes_train.rates.to_numpy()
    # mask = ~jnp.isnan(mean_rates)
    # mean_rates = mean_rates[mask]
    interval = [
        link_f(mean_rates) + bounds[0],
        link_f(mean_rates) + bounds[1],
    ]
    interval_discrete = [
        link_f(mean_rates * binsize) + bounds[0],
        link_f(mean_rates * binsize) + bounds[1],
    ]

    # initialize results
    times = np.zeros((kfolds, 5))
    scores_ll = np.zeros((kfolds,  5))
    scores_r2 = np.zeros((kfolds,  5))
    errors = np.zeros((kfolds, 600, 3))
    params_pa = []
    params_mc = []
    params_h = []
    params_pad = []
    params_db = []

    # POLYNOMIAL APPROXIMATION CONTINUOUS
    n_batches_scan = 10
    obs_model_pa = PopulationPolynomialApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        approx_interval=interval,
        eval_function=basis,
        int_function=GenLaguerreInt(history_window, n_basis_funcs, c=c),
        prod_int_function=GenLaguerreProdIntegral(history_window, n_basis_funcs, c=c),
    )
    tt0 = perf_counter()
    model_pa = PopulationContinuousPA(
        solver_name="LBFGS",
        observation_model=obs_model_pa,
        regularizer_strength=200,
        solver_kwargs={"tol": 1e-12}
    ).fit_closed_form(X_spikes, y_spikes)
    times[k, 2] = perf_counter() - tt0
    print(f"fit PA-c model, time: {times[k, 2]}")
    params_pa.append(np.vstack((model_pa.coef_, model_pa.intercept_)))

    ### MONTE CARLO
    init_lr = 1e-3
    def lr_schedule_mc(step):
        initial_lr = init_lr
        decay_rate = 0.999
        min_rate = 1e-6
        return jnp.clip(initial_lr * decay_rate ** step, min_rate)


    n_batches_scan = 10
    obs_model = MonteCarloApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        eval_function=basis,
        mc_n_samples=2000000,
    )
    model_mc = PopulationContinuousMC(
        solver_name="GradientDescent",
        regularizer="Ridge",
        regularizer_strength=1,
        observation_model=obs_model,
        random_key=jax.random.PRNGKey(0),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_mc})

    tt0 = perf_counter()
    # initialize the model
    init_params = (jnp.zeros((n_basis_funcs * n_neurons, n_neurons)), jnp.atleast_1d(link_f(mean_rates)))
    params = model_mc.initialize_params(X_spikes, y_spikes, init_params=init_params)
    state = model_mc.initialize_state(X_spikes, y_spikes, params)
    # run updates
    num_iter = 600
    error = np.zeros(600)
    for step in range(num_iter):
        t0 = perf_counter()
        params, state = model_mc.update(params, state, X_spikes, y_spikes)
        error[step] = state.error
        if step % 50 == 0:
            print(f"step {step}, error: {error[step]}, step: {state.stepsize}")
    times[k,  3] = (perf_counter() - tt0)
    print(f"fit MC model, time: {times[k,  3]}")
    errors[k,:,0] = error
    params_mc.append(np.vstack((model_mc.coef_, model_mc.intercept_)))

    ### HYBRID
    init_lr_h = 9e-4
    def lr_schedule_h(step):
        initial_lr = init_lr_h
        decay_rate = 0.999
        min_rate = 1e-6
        return jnp.clip(initial_lr * decay_rate ** step, min_rate)

    obs_model_h = MonteCarloApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=10,
        history_window=history_window,
        eval_function=basis,
        mc_n_samples=2000000,
    )
    model_mc_h = PopulationContinuousMC(
        solver_name="GradientDescent",
        regularizer="Ridge",
        regularizer_strength=1,
        observation_model=obs_model_h,
        random_key=jax.random.PRNGKey(0),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_h})
    num_iter = 500
    error_h = np.zeros(600)
    tt0 = perf_counter()
    pa_params = (model_pa.coef_, jnp.atleast_1d(model_pa.intercept_))
    params = model_mc_h.initialize_params(X_spikes, y_spikes, init_params=pa_params)
    state = model_mc_h.initialize_state(X_spikes, y_spikes, params)
    for step in range(num_iter):
        t0 = perf_counter()
        params, state = model_mc_h.update(params, state, X_spikes, y_spikes)
        error_h[step] = state.error
        if step % 50 == 0:
            print(f"step {step}, error {error_h[step]}, time {perf_counter() - t0}")
    times[k, 4] = (perf_counter() - tt0) + times[k, 2]
    print(f"fit H model, time: {times[k,  4]}")
    errors[k, :,  1] = error_h
    params_h.append(np.vstack((model_mc_h.coef_, model_mc_h.intercept_)))

    ### PA-d
    def sufficient_stats(spike_times, kernels, batch_size, ws, binsize):
        n_fun = kernels.shape[1]
        n_neurons = len(spike_times)
        N = n_fun * n_neurons
        full_ts = nap.IntervalSet(spike_times.time_support.start[0], spike_times.time_support.end[-1])
        gaps = full_ts.set_diff(spike_times.time_support)

        n_batches = int(
            jnp.ceil((spike_times.time_support.end[-1] - spike_times.time_support.start[0]) / batch_size))
        batch_intervals = [
            nap.IntervalSet(spike_times.time_support.start[0] + i * batch_size,
                            spike_times.time_support.start[0] + i * batch_size + batch_size + history_window).set_diff(
                gaps)
            for i in range(n_batches)
        ]

        batch_intervals = [
            bi for bi in batch_intervals
            if not bi.shape[0] == 0
        ]

        sum_x = jnp.zeros(N + 1)
        sum_yx = jnp.zeros((N + 1, n_neurons))
        sum_xxT = jnp.zeros((N + 1, N + 1))
        sum_yxxT = jnp.zeros((N + 1, N + 1, n_neurons))

        for i, batch in enumerate(batch_intervals):
            yb = jnp.array(
                spike_times.count(ep=spike_times.time_support.intersect(batch), bin_size=binsize).to_numpy())
            Xb = jnp.array(nmo.convolve.create_convolutional_predictor(kernels, yb).reshape(-1, N))
            yb, Xb = yb[int(ws / binsize):], Xb[int(ws / binsize):]
            Xb = np.concatenate((Xb, np.ones([Xb.shape[0], 1])), axis=1)

            sum_x = sum_x + jnp.sum(Xb, axis=0)
            sum_yx = sum_yx + Xb.T @ yb
            sum_xxT = sum_xxT + Xb.T @ Xb
            print(f"{i + 1}/{len(batch_intervals)}")
            # sum_yxxT = sum_yxxT + jnp.einsum('ti,tj,tn->ijn', Xb, Xb, yb)

        return [sum_x, sum_yx, sum_xxT, sum_yxxT]


    tt0 = perf_counter()
    with jax.default_device(cpu):
        suff_discrete = sufficient_stats(spikes_train, kernels, 20., history_window, binsize)
    Cinv = 200 * np.eye(suff_discrete[2].shape[0])
    Cinv[-1, -1] = 0.0
    weights_d, interval_d = paglm.fit_paglm_population(inverse_link, suff_discrete, interval_discrete, Cinv=Cinv)
    times[k, 1] = perf_counter() - tt0
    print(f"fit PA-d model, time: {times[k, 1]}")
    params_pad.append(weights_d)

    ## DB
    def svrg_full_grad(X, bint, kernels, ws, params, loss_grad, loss_fun):
        N = int(train_duration / binsize)
        D = kernels.shape[1] * n_neurons
        tfg0 = perf_counter()
        grad = jax.tree_util.tree_map(lambda p: jax.numpy.zeros(p.shape), params)
        loss = 0

        for b in bint:
            yb = jnp.array(
                X.count(ep=X.time_support.intersect(b), bin_size=binsize).to_numpy())
            Xb = jnp.array(nmo.convolve.create_convolutional_predictor(kernels, yb).reshape(-1, D))

            yb, Xb = yb[ws:], Xb[ws:]

            grad = jax.tree_util.tree_map(
                lambda g, b: (g * Xb.shape[0] + b * N) / N,
                grad,
                loss_grad(params, Xb, yb),
            )
        loss_b = loss_fun(params, Xb, yb)
        if loss_b.size > 1:
            loss_b = loss_b.sum()

        loss += loss_b

        return grad, loss



    N = n_basis_funcs * n_neurons
    n_ep = 50
    batch_size = 15.
    full_ts = nap.IntervalSet(spikes_train.time_support.start[0], spikes_train.time_support.end[-1])
    gaps = full_ts.set_diff(spikes_train.time_support)

    n_batches = int(
        jnp.ceil((spikes_train.time_support.end[-1] - spikes_train.time_support.start[0]) / batch_size))

    bint = [
        nap.IntervalSet(spikes_train.time_support.start[0] + i * batch_size,
                        spikes_train.time_support.start[0] + i * batch_size + batch_size + history_window).set_diff(
            gaps)
        for i in range(n_batches)
    ]

    bint = [
        bi for bi in bint
        if not bi.shape[0] == 0
    ]

    obs_model_exact = nmo.observation_models.PoissonObservations(inverse_link)
    model_exact = nmo.glm.PopulationGLM(
        solver_name="SVRG",
        # regularizer="Ridge",
        # regularizer_strength=0.01,
        observation_model=obs_model_exact,
        solver_kwargs={"tol": 1e-12, "stepsize": 40000})

    loss = np.zeros(600)
    tt0 = perf_counter()

    y0 = jnp.array(
        spikes_train.count(ep=spikes_train.time_support.intersect(bint[0]), bin_size=binsize).to_numpy())
    X0 = jnp.array(nmo.convolve.create_convolutional_predictor(kernels, y0).reshape(-1, N))
    y0, X0 = y0[window_size:], X0[window_size:]

    init_params = (jnp.zeros((n_basis_funcs * n_neurons, n_neurons)), jnp.atleast_1d(link_f(mean_rates * binsize)))
    params = model_exact.initialize_params(X0, y0, init_params)
    state = model_exact.initialize_state(X0, y0, params)
    loss_grad = jax.jit(jax.grad(model_exact._solver_loss_fun_))
    step = 0
    for ep in range(n_ep):
        shuffled = bint.copy()
        np.random.shuffle(shuffled)
        full_grad, full_loss = svrg_full_grad(spikes_train, bint, kernels, window_size,
                                              params, loss_grad, model_exact._predict_and_compute_loss)
        state = state._replace(full_grad_at_reference_point=full_grad)
        loss[ep] = full_loss
        state = state._replace(full_grad_at_reference_point=full_grad)

        for b in shuffled:
            yb = jnp.array(
                spikes_train.count(ep=spikes_train.time_support.intersect(b), bin_size=binsize).to_numpy())
            Xb = jnp.array(nmo.convolve.create_convolutional_predictor(kernels, yb).reshape(-1, N))

            yb, Xb = yb[window_size:], Xb[window_size:]
            params, state = model_exact.update(params, state, Xb, yb)
            step += 1

        # update state for this epoch
        state = state._replace(error=tree_l2_norm(tree_sub(params, state.reference_point)) /
                                     tree_l2_norm(state.reference_point))
        state = state._replace(reference_point=params)
        if ep % 10 == 0:
            print(f"ep {ep}, loss {loss[ep]}")
    times[k, 0] = perf_counter() - tt0
    print(f"fit DG, time: {times[k, 0]}")
    errors[k, :, 2] = loss
    params_db.append(np.vstack((model_exact.coef_, model_exact.intercept_)))

    ## COMPUTE SCORES
    X_test_d = nmo.convolve.create_convolutional_predictor(kernels, y_test).reshape(-1, N)
    X_test_d = X_test_d[window_size:]
    pred_pa = model_pa.predict(X_spikes, 30, binsize, (test_intervals[k].start[0], test_intervals[k].end[-1]))
    pred_mc = model_mc.predict(X_spikes, 30, binsize, (test_intervals[k].start[0], test_intervals[k].end[-1]))
    pred_h = model_mc_h.predict(X_spikes, 30, binsize, (test_intervals[k].start[0], test_intervals[k].end[-1]))
    pred_paglm = quadratic(np.dot(X_test_d, weights_d[:-1]) + weights_d[-1], inverse_link, interval_discrete)
    pred_db = model_exact.predict(X_test_d)

    scores_ll[k, 0] = obs_model_exact.log_likelihood(y_test[window_size:], pred_db)
    scores_ll[k, 1] = obs_model_exact.log_likelihood(y_test[window_size:], pred_paglm)
    scores_ll[k, 2] = obs_model_exact.log_likelihood(y_test, pred_pa*binsize)
    scores_ll[k, 3] = obs_model_exact.log_likelihood(y_test, pred_mc*binsize)
    scores_ll[k, 4] = obs_model_exact.log_likelihood(y_test, pred_h*binsize)

    scores_r2[k, 0] = obs_model_exact.pseudo_r2(y_test[window_size:], pred_db)
    scores_r2[k, 1] = obs_model_exact.pseudo_r2(y_test[window_size:], pred_paglm)
    scores_r2[k, 2] = obs_model_exact.pseudo_r2(y_test, pred_pa*binsize)
    scores_r2[k, 3] = obs_model_exact.pseudo_r2(y_test, pred_mc*binsize)
    scores_r2[k, 4] = obs_model_exact.pseudo_r2(y_test, pred_h*binsize)

    results = {
        "scores_ll": scores_ll,
        "scores_r2": scores_r2,
        "times": times,
        "errors": errors,
        "params_db": params_db,
        "params_pad": params_pad,
        "params_pa": params_pa,
        "params_mc": params_mc,
        "params_h": params_h,
        "kernels": kernels,
        "binsize": binsize,
    }

    np.savez(f"/mnt/home/amedvedeva/ceph/real_data_100s.npz", **results)
    print(f"FIT K={k}")

print("Script terminated")
