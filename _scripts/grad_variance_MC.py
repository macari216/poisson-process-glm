import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo
import pynapple as nap
import scipy.sparse as sp

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import PopulationContinuousMC
from poisson_point_process.poisson_process_obs_model import MonteCarloApproximation
from poisson_point_process.basis import RaisedCosineLogEval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]


def rsn_w(p=0.15, neg=0.3, m=0., v=1., n=100, bf=10):
    conx = sp.random(n, n, density=p)
    conx.data = np.ones(shape=conx.nnz)
    conx = conx.toarray()
    W = np.zeros((n, n, bf))
    for i in range(n):
        for j in range(n):
            if conx[i, j] != 0:
                W[i, j] = np.abs(np.random.normal(m, v, size=bf))
        w = W.reshape(-1, bf)
        for i in (np.random.randint(0, n ** 2, int(n ** 2 * neg))):
            w[i] = -w[i]
        W = w.reshape(n, n, bf)

    return W


n_neurons = 15
history_window = 0.004
tot_time_sec = 100
binsize = 0.0001
window_size = int(history_window / binsize)
n_basis_funcs = 5
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
inverse_link = jnp.exp
link_f = jnp.log

basis_fn = RaisedCosineLogEval(history_window, n_basis_funcs)
kernels = basis_fn(-np.linspace(0, history_window, window_size))
kernels_sim = basis_fn(-np.linspace(0, history_window, window_size * 2))

max_iter = 2000
n_samples = 100

kfold = 5

stepsize_mc = np.zeros((kfold,max_iter))
stepsize_db = np.zeros((kfold,max_iter))

loss_mc_full = np.zeros((kfold,max_iter))
loss_db_full = np.zeros((kfold,max_iter))
loss_mc = np.zeros((kfold,max_iter))
loss_db = np.zeros((kfold,max_iter))

for k in range(kfold):

    np.random.seed(123 + k)

    baseline_fr = 2.1
    biases_hz = jnp.log(jnp.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons)))
    biases = biases_hz + jnp.log(binsize / 2)
    weights_true = jnp.array(rsn_w(p=0.4, neg=0.2, m=0, v=0.5, n=n_neurons, bf=n_basis_funcs))
    for i in range(n_neurons):
        weights_true = weights_true.at[i, i].set(-jnp.abs(weights_true[i, i]))
    params = (weights_true, biases)
    filters_true = np.einsum("ijk,tk->ijt", weights_true, kernels) + biases[None, :, None] - jnp.log(binsize)

    with jax.default_device(cpu):
        spike_counts, _ = simulate.poisson_counts_recurrent(
            n_bins_tot * 2, n_neurons, window_size * 2, kernels_sim, params, inverse_link
        )

        spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize / 2)

        X_spikes = jnp.vstack((spike_times, spike_ids))
        y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))

        sorted_spikes = [nap.Ts(np.array(X_spikes[0][X_spikes[1] == n] + binsize)) for n in range(n_neurons)]
        spikes_tsgroup = nap.TsGroup({n: sorted_spikes[n] for n in range(n_neurons)})
        spike_counts = jnp.array(spikes_tsgroup.count(binsize, nap.IntervalSet(0, tot_time_sec)))

        X_full = nmo.convolve.create_convolutional_predictor(kernels, spike_counts).reshape(-1,
                                                                                            n_neurons * n_basis_funcs)
        X_full, spike_counts, = X_full[window_size:], spike_counts[window_size:]


    def lr_schedule_mc(step):
        initial_lr = 35e-4
        decay_rate = 0.999
        return initial_lr * decay_rate ** step


    def lr_schedule_db(step):
        initial_lr = 60000
        decay_rate = 0.999
        return initial_lr * decay_rate ** step


    #### optimization
    obs_model_mc_full = MonteCarloApproximation(
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        mc_n_samples=2000000,
        eval_function=basis_fn,
    )

    model_mc_full = PopulationContinuousMC(
        solver_name="GradientDescent",
        observation_model=obs_model_mc_full,
        random_key=jax.random.PRNGKey(0),
        recording_time=nap.IntervalSet(0, tot_time_sec),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_mc}
    )
    params_mc_full = model_mc_full.initialize_params(X_spikes, y_spikes)
    state_mc_full = model_mc_full.initialize_state(X_spikes, y_spikes, params_mc_full)

    # initialize DB model
    obs_model_db_full = nmo.observation_models.PoissonObservations(inverse_link)
    model_db_full = nmo.glm.PopulationGLM(
        solver_name="GradientDescent",
        observation_model=obs_model_db_full,
        solver_kwargs={"tol": 1e-12, "acceleration": False, "stepsize": lr_schedule_db}
    )
    params_db_full = model_db_full.initialize_params(X_full, spike_counts)
    state_db_full = model_db_full.initialize_state(X_full, spike_counts, params_db_full)

    obs_model_mc = MonteCarloApproximation(
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        mc_n_samples=n_samples,
        eval_function=basis_fn,
    )

    model_mc = PopulationContinuousMC(
        solver_name="GradientDescent",
        observation_model=obs_model_mc,
        random_key=jax.random.PRNGKey(0),
        recording_time=nap.IntervalSet(0, tot_time_sec),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_mc}
    )
    params_mc = model_mc.initialize_params(X_spikes, y_spikes)
    state_mc = model_mc.initialize_state(X_spikes, y_spikes, params_mc)

    # initialize DB model
    obs_model_db = nmo.observation_models.PoissonObservations(inverse_link)
    model_db = nmo.glm.PopulationGLM(
        solver_name="GradientDescent",
        observation_model=obs_model_db,
        solver_kwargs={"tol": 1e-12, "acceleration": False, "stepsize": lr_schedule_db}
    )
    params_db = model_db.initialize_params(X_full, spike_counts)
    state_db = model_db.initialize_state(X_full, spike_counts, params_db)
    bst = np.arange(window_size, n_bins_tot - window_size, n_samples)

    for step in range(max_iter):
        params_mc_full, state_mc_full = model_mc_full.update(params_mc_full, state_mc_full, X_spikes, y_spikes)
        stepsize_mc[k, step] = state_mc_full.stepsize
        loss_mc_full[k, step], _ = model_mc_full._predict_and_compute_loss(params_mc_full, X_spikes, y_spikes,
                                                                           state_mc_full.aux)

        params_mc, state_mc = model_mc.update(params_mc, state_mc, X_spikes, y_spikes)
        loss_mc[k, step], _ = model_mc._predict_and_compute_loss(params_mc, X_spikes, y_spikes, state_mc.aux)

        params_db_full, state_db_full = model_db_full.update(params_db_full, state_db_full, X_full, spike_counts)
        stepsize_db[k, step] = state_db_full.stepsize
        loss_db_full[k, step] = model_db_full._predict_and_compute_loss(params_db_full, X_full, spike_counts)

        st = jnp.array(np.random.choice(bst))
        params_db, state_db = model_db.update(params_db, state_db, X_full[st:st + n_samples],
                                              spike_counts[st:st + n_samples])
        loss_db[k, step] = model_db._predict_and_compute_loss(params_db, X_full[st:st + n_samples],
                                                              spike_counts[st:st + n_samples])
        if step % 200 == 0:
            print(
                f"step {k, step}, mcf {loss_mc_full[k, step]}, mc {loss_mc[k, step]}, dbf {loss_db_full[k, step]}, {loss_db[k, step]}")

    results = {
        "stepsize_mc": stepsize_mc,
        "stepsize_db": stepsize_db,
        "loss_mc_full": loss_mc_full,
        "loss_db_full": loss_db_full,
        "loss_mc": loss_mc,
        "loss_db": loss_db,
    }

    np.savez("/mnt/home/amedvedeva/ceph/loss_comparison.npz", **results)

print("Script terminated")


















# import os
#
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#
# from time import perf_counter
#
# import jax
# import jax.numpy as jnp
# import numpy as np
# import nemos as nmo
# import pynapple as nap
# import matplotlib.pyplot as plt
# import scipy.sparse as sp
#
# from poisson_point_process import simulate
# from poisson_point_process.poisson_process_glm import ContinuousMC, PopulationContinuousMC
# from poisson_point_process.poisson_process_obs_model import MonteCarloApproximation
# from poisson_point_process.basis import LaguerreEval, RaisedCosineLogEval
#
# jax.config.update("jax_enable_x64", True)
# os.environ["JAX_PLATFORM_NAME"] = "gpu"
#
# cpu = jax.devices("cpu")[0]
# gpu = jax.devices("gpu")[0]
#
# n_neurons = 15
# history_window = 0.004
# tot_time_sec = 100
# binsize = 0.0001
# window_size = int(history_window / binsize)
# n_basis_funcs = 5
# n_bins_tot = int(tot_time_sec / binsize)
# n_batches_scan = 1
# inverse_link = jnp.exp
# link_f = jnp.log
#
# basis_fn = RaisedCosineLogEval(history_window, n_basis_funcs)
# kernels = basis_fn(-np.linspace(0, history_window, window_size))
# kernels_sim = basis_fn(-np.linspace(0, history_window, window_size * 2))
#
#
# def rsn_w(p=0.15, neg=0.3, m=0., v=1., n=100, bf=10):
#     conx = sp.random(n, n, density=p)
#     conx.data = np.ones(shape=conx.nnz)
#     conx = conx.toarray()
#     W = np.zeros((n, n, bf))
#     for i in range(n):
#         for j in range(n):
#             if conx[i, j] != 0:
#                 W[i, j] = np.abs(np.random.normal(m, v, size=bf))
#         w = W.reshape(-1, bf)
#         for i in (np.random.randint(0, n ** 2, int(n ** 2 * neg))):
#             w[i] = -w[i]
#         W = w.reshape(n, n, bf)
#
#     return W
#
#
# baseline_fr = 2.1
# biases = jnp.log(jnp.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons))) + jnp.log(binsize/2)
#
# # weights_true = jnp.array(np.random.normal(0, 0.1, size=(n_neurons, n_neurons, n_basis_funcs)))
# weights_true = jnp.array(rsn_w(p=0.4, neg=0.2, m=0., v=0.5, n=n_neurons, bf=n_basis_funcs))
# params = (weights_true, biases)
#
# filters_true = np.dot(weights_true.reshape(-1, n_basis_funcs), kernels_sim.T) + np.log(
#             baseline_fr) + np.log(binsize)
#
# with jax.default_device(cpu):
#     spike_counts, _ = simulate.poisson_counts_recurrent(
#         n_bins_tot * 2, n_neurons, window_size * 2, kernels_sim, params, inverse_link
#     )
#
#     spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize / 2)
#
#     X_spikes = jnp.vstack((spike_times, spike_ids))
#     y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))
#
#     sorted_spikes = [nap.Ts(np.array(X_spikes[0][X_spikes[1] == n] + binsize)) for n in range(n_neurons)]
#     spikes_tsgroup = nap.TsGroup({n: sorted_spikes[n] for n in range(n_neurons)})
#     spike_counts = jnp.array(spikes_tsgroup.count(binsize, nap.IntervalSet(0, tot_time_sec)))
#
#     X_full = nmo.convolve.create_convolutional_predictor(kernels, spike_counts).reshape(-1, n_neurons * n_basis_funcs)
#
# max_iter = 2000
# k_iter = [1, max_iter]
# # k_iter = jnp.linspace(1, max_iter, 8).astype(int)
# # step_sizes = []
# # sample_sizes = [100, 200]
# sample_sizes = jnp.logspace(2, 5, 4).astype(int)
#
# grad_samples = 100
# all_keys = jax.random.split(jax.random.PRNGKey(1), grad_samples).reshape(grad_samples, -1)
#
#
# def lr_schedule_mc(step):
#     initial_lr = 35e-4
#     decay_rate = 0.999
#     return initial_lr * decay_rate ** step
#
#
# def lr_schedule_db(step):
#     initial_lr = 60000
#     decay_rate = 0.999
#     return initial_lr * decay_rate ** step
#
#
# grad_dict = {}
#
# # # MC model for full grad
# # obs_model_mc_full = MonteCarloApproximation(
# #     n_basis_funcs=n_basis_funcs,
# #     n_batches_scan=n_batches_scan,
# #     history_window=history_window,
# #     mc_n_samples=2000000,
# #     eval_function=basis_fn,
# # )
# #
# # model_mc_full = PopulationContinuousMC(
# #     solver_name="GradientDescent",
# #     observation_model=obs_model_mc_full,
# #     random_key=jax.random.PRNGKey(0),
# #     solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_mc}
# # )
# # params_mc_full = model_mc_full.initialize_params(X_spikes, y_spikes)
# # state_mc_full = model_mc_full.initialize_state(X_spikes, y_spikes, params_mc_full)
# # error_mc_full = np.zeros(max_iter)
# #
# # # initialize DB model
# # obs_model_db_full = nmo.observation_models.PoissonObservations(inverse_link)
# # model_db_full = nmo.glm.PopulationGLM(
# #     solver_name="GradientDescent",
# #     observation_model=obs_model_db_full,
# #     solver_kwargs={"tol": 1e-12, "acceleration": False, "stepsize": lr_schedule_db}
# # )
# # params_db_full = model_db_full.initialize_params(X_full, spike_counts)
# # state_db_full = model_db_full.initialize_state(X_full, spike_counts, params_db_full)
# # error_db_full = np.zeros(max_iter)
# #
# # stepsize_mc_full = np.zeros(max_iter)
# # stepsize_db = np.zeros(max_iter)
# #
# # params_mc_k = []
# # params_db_k = []
# # for k in range(1, max_iter + 1):
# #     params_mc_full, state_mc_full = model_mc_full.update(params_mc_full, state_mc_full, X_spikes, y_spikes)
# #     error_mc_full[k - 1] = state_mc_full.error
# #     stepsize_mc_full[k - 1] = state_mc_full.stepsize
# #
# #     params_db_full, state_db_full = model_db_full.update(params_db_full, state_db_full, X_full, spike_counts)
# #     error_db_full[k - 1] = state_db_full.error
# #     stepsize_db[k - 1] = state_db_full.stepsize
# #
# #     if k in k_iter:
# #         params_mc_k.append(params_mc_full)
# #         params_db_k.append(params_db_full)
# #     if k % 100 == 0:
# #         print(f"step {k}")
# # print("trained full models")
# #
# # grad_dict["error_mc_full"] = error_mc_full
# # grad_dict["error_db"] = error_db_full
# # grad_dict["step_mc_full"] = stepsize_mc_full
# # grad_dict["step_db"] = stepsize_db
# #
# # # set up gradient functions
# # loss_grad_mc_full = jax.grad(
# #     lambda p: model_mc_full._solver_loss_fun_(p, X_spikes, y_spikes, all_keys[0]), has_aux=True
# # )
# # loss_grad_db_full = jax.grad(
# #     lambda p: model_db_full._solver_loss_fun_(p, X_full[window_size:], spike_counts[window_size:])
# # )
# #
# # # compute full gradients
# # grad_k_mc_full_w = []
# # grad_k_db_full_w = []
# # grad_k_mc_full_b = []
# # grad_k_db_full_b = []
# # for k in range(len(k_iter)):
# #     if k == 0:
# #         grad_norm_mc = loss_grad_mc_full(params_mc_k[k])[0]
# #         grad_norm_db = loss_grad_db_full(params_db_k[k])
# #         grad_dict["grad_norm_mc_w"] = grad_norm_mc[0]
# #         grad_dict["grad_norm_mc_b"] = grad_norm_mc[1]
# #         grad_dict["grad_norm_db_w"] = grad_norm_db[0]
# #         grad_dict["grad_norm_db_b"] = grad_norm_db[1]
# #
# #     grad_k_mc_full_w.append(loss_grad_mc_full(params_mc_k[k])[0][0])
# #     grad_k_db_full_w.append(loss_grad_db_full(params_db_k[k])[0])
# #     grad_k_mc_full_b.append(loss_grad_mc_full(params_mc_k[k])[0][1])
# #     grad_k_db_full_b.append(loss_grad_db_full(params_db_k[k])[1])
# #
# # grad_dict["grad_k_mc_full_w"] = grad_k_mc_full_w
# # grad_dict["grad_k_db_full_w"] = grad_k_db_full_w
# # grad_dict["grad_k_mc_full_b"] = grad_k_mc_full_b
# # grad_dict["grad_k_db_full_b"] = grad_k_db_full_b
# #
# # print("computed full grads")
# #
# # for i, n_samples in enumerate(sample_sizes):
# #     grad_k_mc_st_w = []
# #     grad_k_db_st_w = []
# #     grad_k_mc_st_b = []
# #     grad_k_db_st_b = []
# #
# #     # initialize MC model
# #     obs_model_mc = MonteCarloApproximation(
# #         n_basis_funcs=n_basis_funcs,
# #         n_batches_scan=n_batches_scan,
# #         history_window=history_window,
# #         mc_n_samples=n_samples,
# #         eval_function=basis_fn,
# #     )
# #
# #     model_mc = PopulationContinuousMC(
# #         solver_name="GradientDescent",
# #         observation_model=obs_model_mc,
# #         random_key=jax.random.PRNGKey(0),
# #         solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1}
# #     )
# #     params_mc = model_mc.initialize_params(X_spikes, y_spikes)
# #     state_mc = model_mc.initialize_state(X_spikes, y_spikes, params_mc)
# #
# #     #initialize DB model
# #     obs_model_db = nmo.observation_models.PoissonObservations(inverse_link)
# #     model_db = nmo.glm.PopulationGLM(
# #         solver_name="GradientDescent",
# #         observation_model=obs_model_db,
# #         solver_kwargs={"tol": 1e-12, "acceleration": False, "stepsize": -1}
# #     )
# #     params_db = model_db.initialize_params(X_full, spike_counts)
# #     state_db = model_db.initialize_state(X_full, spike_counts, params_db)
# #
# #     # gradient function
# #     loss_grad_mc = jax.vmap(
# #         jax.grad(
# #             lambda p, k: model_mc._solver_loss_fun_(p, X_spikes, y_spikes, k), has_aux=True
# #         ),
# #         in_axes=(None, 0)
# #     )
# #
# #     loss_grad_db = jax.vmap(
# #         jax.grad(
# #             model_db._solver_loss_fun_
# #         ),
# #         in_axes=(None, 0, 0)
# #     )
# #
# #     # draw random batches
# #     bst = np.arange(window_size, n_bins_tot - window_size, n_samples)
# #     samples = jnp.array(np.random.choice(bst, size=grad_samples))
# #
# #     Xb = jax.vmap(lambda st: jax.lax.dynamic_slice_in_dim(X_full[window_size:], st, n_samples))(samples)
# #     yb = jax.vmap(lambda st: jax.lax.dynamic_slice_in_dim(spike_counts[window_size:], st, n_samples))(samples)
# #
# #     for k in range(len(k_iter)):
# #         grad_k_mc_st_w.append(loss_grad_mc(params_mc_k[k], all_keys)[0][0])
# #         grad_k_db_st_w.append(loss_grad_db(params_db_k[k], Xb, yb)[0])
# #         grad_k_mc_st_b.append(loss_grad_mc(params_mc_k[k], all_keys)[0][1])
# #         grad_k_db_st_b.append(loss_grad_db(params_db_k[k], Xb, yb)[1])
# #
# #     grad_dict[f"{n_samples} grad_k_mc_st_w"] = grad_k_mc_st_w
# #     grad_dict[f"{n_samples} grad_k_db_st_w"] = grad_k_db_st_w
# #     grad_dict[f"{n_samples} grad_k_mc_st_b"] = grad_k_mc_st_b
# #     grad_dict[f"{n_samples} grad_k_db_st_b"] = grad_k_db_st_b
# #
# #     print(f"fit {n_samples}, {i + 1}/{len(sample_sizes)}")
#
#
# for i, n_samples in enumerate(sample_sizes):
#     # initialize MC model
#     obs_model_mc = MonteCarloApproximation(
#         n_basis_funcs=n_basis_funcs,
#         n_batches_scan=n_batches_scan,
#         history_window=history_window,
#         mc_n_samples=n_samples,
#         eval_function=basis_fn,
#     )
#
#     model_mc = PopulationContinuousMC(
#         solver_name="GradientDescent",
#         observation_model=obs_model_mc,
#         random_key=jax.random.PRNGKey(0),
#         solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1}
#     )
#     params_mc = model_mc.initialize_params(X_spikes, y_spikes)
#     state_mc = model_mc.initialize_state(X_spikes, y_spikes, params_mc)
#
#     #initialize DB model
#     obs_model_db = nmo.observation_models.PoissonObservations(inverse_link)
#     model_db = nmo.glm.PopulationGLM(
#         solver_name="GradientDescent",
#         observation_model=obs_model_db,
#         solver_kwargs={"tol": 1e-12, "acceleration": False, "stepsize": -1}
#     )
#     params_db = model_db.initialize_params(X_full, spike_counts)
#     state_db = model_db.initialize_state(X_full, spike_counts, params_db)
#
#     bst = np.arange(window_size, n_bins_tot - window_size, n_samples)
#
#     mse_mc_k = []
#     mse_db_k = []
#     error_mc = np.zeros(max_iter)
#     error_db = np.zeros(max_iter)
#     for k in range(1, max_iter + 1):
#         params_mc, state_mc = model_mc.update(params_mc, state_mc, X_spikes, y_spikes)
#         error_mc[k-1] = state_mc.error
#
#         st = jnp.array(np.random.choice(bst))
#         params_db, state_db = model_db.update(params_db, state_db, X_full[st:st+n_samples], spike_counts[st:st+n_samples])
#         error_db[k - 1] = state_db.error
#
#         if k in k_iter:
#             filters_mc = np.einsum("jki,tk->ijt", model_mc.coef_.reshape(-1, n_basis_funcs, n_neurons), kernels) + model_mc.intercept_
#             filters_db = np.einsum("jki,tk->ijt", model_db.coef_.reshape(-1, n_basis_funcs, n_neurons), kernels) + model_db.intercept_ + np.log(binsize)
#             mse_db_k.append(jnp.mean(jnp.square(inverse_link(filters_true) - inverse_link(filters_db))))
#             mse_mc_k.append(jnp.mean(jnp.square(inverse_link(filters_true) - inverse_link(filters_mc))))
#
#         if k % 100 == 0:
#             print(f"step {k}")
#     print("trained stochastic models")
#
#     grad_dict[f"{n_samples} mse_db_k"] = mse_db_k
#     grad_dict[f"{n_samples} mse_mc_k"] = mse_mc_k
#
#     print(f"fit {n_samples}, {i + 1}/{len(sample_sizes)}")
#
#
#
#
#
# np.savez(f"/mnt/home/amedvedeva/ceph/var_grads_mse.npz", **grad_dict)
# print("Script terminated")
#
#
# # grad_dict = np.load("/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbird_output/var_grads.npz", allow_pickle=True)
# # w_grad = []
# # b_grad = []
# # for i, key in enumerate(grad_dict.keys()):
# #     if i%2==0:
# #         w_grad.append(grad_dict[key])
# #     else:
# #         b_grad.append(grad_dict[key])
# #
# # w_var = jnp.var(jnp.array(w_grad),axis=1)
# # b_var = jnp.var(jnp.array(b_grad),axis=1)
# #
# # mean_w = w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2).mean(1)
# # se_w = w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2).std(1) / np.sqrt(n_neurons)
# # plt.figure()
# # plt.plot(M_size, b_var, label="bias")
# # plt.plot(M_size, mean_w,label=f"weights", c='r')
# # plt.fill_between(M_size, mean_w - se_w, mean_w+se_w, alpha=0.3, color='r')
# # plt.vlines(y_counts.sum(),0, mean_w.max(), color='k', label='K')
# # # plt.xlim(0,8000)
# # plt.xlabel("MC sample size")
# # plt.ylabel("grad variance")
# # plt.legend()
#
# # plt.show()