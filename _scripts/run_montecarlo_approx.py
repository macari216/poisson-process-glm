import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.utils import compute_max_window_size
from poisson_point_process.poisson_process_glm import ContinuousGLM

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

all_to_all = False

# generate data
n_neurons = 2
target_neu_id = 0
history_window = 0.3
tot_time_sec = 1000
binsize = 0.01
window_size = int(history_window / binsize)
n_basis_funcs = 2
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
inverse_link = jax.nn.softplus
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
_, kernels = rc_basis.evaluate_on_grid(window_size)

np.random.seed(216)

t0 = perf_counter()
if all_to_all:
    # all-to-all connectivity (bias and weights)

    # true parameters
    baseline_fr = 2.3
    biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr/10, n_neurons))) + np.log(binsize)
    bias_true = biases[target_neu_id]

    weights_true = jnp.array(np.random.normal(-0.1, 0.01, size=(n_neurons, n_neurons, n_basis_funcs)))
    params = (weights_true, biases)

    spike_counts, _ = simulate.poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, kernels, params, inverse_link)

    print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
    print(f"max spikes per bin: {spike_counts.max()}")
    print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(spike_counts.sum(0))}")
    # plt.hist(spike_counts.sum(0), bins=50)
    # plt.show()

    spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize)

    # spikes_tsgroup = nap.TsGroup({n: nap.Ts(np.array(spike_times[neuron_indices == n]))
    #                               for n in range(n_neurons)},
    #                                 nap.IntervalSet(0, tot_time_sec))

    X_spikes = jnp.vstack((spike_times,spike_ids))
    spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
    y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

    print(f"y spikes total: {y_spikes.shape[1]}")
    print(f"X spikes total: {X_spikes.shape[1]}")

    rc_basis.set_input_shape(spike_counts)

    X_discrete = rc_basis.compute_features(spike_counts)
    y_discrete = spike_counts[:, target_neu_id]

else:
    # all-to-one connectivity
    pres_rate_per_sec = 10
    posts_rate_per_sec = 2
    # rescaled proportionally to the binsize
    bias_true =  posts_rate_per_sec + np.log(binsize)
    weights_true = np.random.normal(0, 1, n_neurons * n_basis_funcs)
    X, y_counts, X_counts, lam_posts =  simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                               n_bins_tot, n_neurons, weights_true, window_size, rc_basis, inverse_link)
    inv_rates = np.log(np.exp(lam_posts) - 1)

    print(jnp.hstack((y_counts[:,None], X_counts)).shape)

    print(f"mean spikes per neuron: {jnp.mean(jnp.hstack((y_counts[:,None], X_counts)).sum(0))}")
    print(f"max spikes per bin: {jnp.hstack((y_counts[:,None], X_counts)).max()}")
    print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(jnp.hstack((y_counts[:,None], X_counts)).sum(0))}")
    plt.hist(inv_rates, bins=50)
    plt.show()

    spike_times, spike_ids = simulate.poisson_times(jnp.hstack((y_counts[:,None], X_counts)), tot_time_sec, binsize)
    X_spikes = jnp.vstack((spike_times,spike_ids))
    spike_idx_target = jnp.arange(len(spike_times))[spike_ids == 0]
    y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

    rc_basis.set_input_shape(y_counts)

    X_discrete = jnp.hstack((rc_basis.compute_features(y_counts), X))
    y_discrete = y_counts

print(f"generated data: {np.round(perf_counter() - t0, 5)}")

# maximum number of spikes in window across the entire dataset
max_window = int(compute_max_window_size(
        jnp.array([-history_window, 0]), X_spikes[0], X_spikes[0]
    ))

obs_model_kwargs = {
    "n_basis_funcs": n_basis_funcs,
    "history_window": history_window,
    "inverse_link_function": inverse_link,
    "max_window": max_window,
    "n_batches_scan": n_batches_scan,
    "mc_random_key": jax.random.PRNGKey(0),
}

# model = ContinuousGLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1., obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True})
model = ContinuousGLM(solver_name="GradientDescent", obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True, "stepsize": 1, "acceleration": False})

true_params = (jnp.concatenate((jnp.zeros(2),weights_true)), jnp.array([bias_true]))
params = model.initialize_params(X_spikes, y_spikes, init_params=true_params)
print(params)
state = model.initialize_state(X_spikes, y_spikes, params)

num_iter = 200
tt0 = perf_counter()
error = np.zeros(num_iter)
for step in range(num_iter):
    t0 = perf_counter()
    params, state = model.update(params, state, X_spikes, y_spikes)
    # print(params)
    error[step] = model._negative_log_likelihood(X_spikes, y_spikes, params, state.aux)
    if step % 10 == 0:
        print(f"step {step}, {perf_counter() - t0}")
print(f"fitted model, {perf_counter() - tt0}")

obs_model = nmo.observation_models.PoissonObservations(inverse_link)
model_exact = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model).fit(X_discrete, y_discrete)

weights_nemos, bias_nemos = model_exact.coef_.reshape(-1,n_basis_funcs)[1:], model_exact.intercept_
weights_mc, bias_mc = model.coef_.reshape(-1,n_basis_funcs)[1:], model.intercept_
filters_mc = np.dot(weights_mc, kernels.T)
filters_nemos = np.dot(weights_nemos, kernels.T)
if len(weights_true.shape) == 3:
    weights_true = weights_true[target_neu_id]
filters_true = np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T)

sc_f = np.max(np.abs(filters_mc)) / np.max(np.abs(filters_true))
print(weights_true)
print(weights_mc)

fig, axs = plt.subplots(1,2,figsize=(7,5))
for n, ax in enumerate(axs.flat):
    # ax.plot(jax.nn.softplus(filters_mc[n]), c='r', label='approx')
    ax.plot(filters_true[n], c='g', label='true')
    ax.plot(filters_nemos[n], c='k', label='exact')
    # ax.plot(filters_mc[n], c='r', label='approx')
    # ax.plot(filters_true[n], c='g', label='true')
    # ax.plot(filters_nemos[n], c='k', label='exact')
plt.legend()

fig, axs = plt.subplots(1,2,figsize=(7,5))
for n, ax in enumerate(axs.flat):
    ax.plot(filters_mc[n], c='r', label='approx')
    # ax.plot(jax.nn.softplus(filters_true[n]), c='g', label='true')

plt.figure()
plt.plot(error)
plt.show()