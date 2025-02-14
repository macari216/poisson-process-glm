import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.utils import compute_max_window_size
from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

all_to_all = False

# generate data
n_neurons = 5
target_neu_id = 0
history_window = 0.3
tot_time_sec = 1000
binsize = 0.01
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
inverse_link = jax.nn.softplus
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window

np.random.seed(123)

t0 = perf_counter()
if all_to_all:
    # all-to-all connectivity (bias and weights)

    # true parameters
    baseline_fr = 2.3
    biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr/10, n_neurons))) + np.log(binsize)
    bias_true = biases[target_neu_id]

    weights_true = jnp.array(np.random.normal(0, 0.1, size=(n_neurons, n_neurons, n_basis_funcs)))
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
    weights_true = np.random.normal(0, 0.3, n_neurons * n_basis_funcs)
    # weights_true = jnp.array([-0.5, -0.7, 0.1, 0.9])
    X, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                               n_bins_tot, n_neurons, weights_true, window_size, rc_basis, inverse_link)
    inv_rates = np.log(np.exp(lam_posts) - 1)

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
    "mc_n_samples": y_spikes.shape[1]*5
}
print(y_spikes.shape[1]*5)

# model = ContinuousGLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1., obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True})
# model = ContinuousGLM(solver_name="LBFGS", obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True})
model = ContinuousMC(solver_name="GradientDescent", obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 0.0005})

true_params = (jnp.concatenate((jnp.zeros(n_basis_funcs),weights_true)), jnp.array([bias_true]))
# true_params = (weights_true, jnp.array([posts_rate_per_sec]))
# true_params = (jnp.zeros_like(weights_true), jnp.array([bias_true]))
params = model.initialize_params(X_spikes, y_spikes, init_params=true_params)
print(weights_true)
state = model.initialize_state(X_spikes, y_spikes, params)

num_iter = 1000
tt0 = perf_counter()
error = np.zeros(num_iter)
error_sr = 10
for step in range(num_iter):
    t0 = perf_counter()
    params, state = model.update(params, state, X_spikes, y_spikes)
    t1 = perf_counter()
    # error[step] = model._negative_log_likelihood(X_spikes, y_spikes, params, state.aux)
    if step % error_sr == 0:
        error[step] = model._negative_log_likelihood(X_spikes, y_spikes, params, state.aux)
        print(f"step {step}, {t1 - t0}")
        print(params)
print(f"fitted model, {perf_counter() - tt0}")
error = error[error!=0]

obs_model = nmo.observation_models.PoissonObservations(inverse_link)
model_exact = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model).fit(X_discrete, y_discrete)

weights_nemos, bias_nemos = model_exact.coef_.reshape(-1,n_basis_funcs)[1:], model_exact.intercept_
weights_mc, bias_mc = model.coef_.reshape(-1,n_basis_funcs), model.intercept_
filters_mc = np.dot(weights_mc[1:], kernels.T) + bias_mc
filters_nemos = np.dot(weights_nemos, kernels.T) + bias_nemos
if len(weights_true.shape) == 3:
    weights_true = weights_true[target_neu_id]
filters_true = np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T) + bias_true

sc_f = np.max(np.abs(filters_mc)) / np.max(np.abs(filters_true))
print(f"true weights: {weights_true}")
print(f"inferred weights: {model.coef_}")

fig, axs = plt.subplots(3,2,figsize=(7,9))
axs = axs.flat
for n, ax in enumerate(axs):
    if n==0:
        ax.plot(np.arange(0, num_iter,error_sr),error, c='k')
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    else:
        line1 = ax.plot(time, jax.nn.softplus(filters_true[n-1]), c='g', label='true')
        line2 = ax.plot(time, jax.nn.softplus(filters_nemos[n-1]), c='k', label='exact')
        line3 = ax.plot(time, jax.nn.softplus(filters_mc[n-1]) * binsize, c='r', label='approx')
        ax.set_xlabel("time from spike")
        ax.set_ylabel("gain")
        # ax1 = ax.twinx()
        # line3 = ax1.plot(time, jax.nn.softplus(filters_mc[n-1])*binsize, c='r', label='approx')
        # ax1.tick_params(axis='y', labelcolor='r')

        if n==len(axs)-1:
            # ax1.set_ylabel("gain (approx)", color='r')
            lines = [line1[0], line2[0], line3[0]]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")

plt.tight_layout()
plt.show()


# fig, axs = plt.subplots(1,2,figsize=(9,5))
# for n, ax in enumerate(axs.flat):
#     line1 = ax.plot(time, filters_true[n], c='g', label='true')
#     line2 = ax.plot(time, filters_nemos[n], c='k', label='exact')
#     ax.set_xlabel("time from spike")
#     ax.set_ylabel("gain")
#     ax2 = ax.twinx()
#     line3 = ax2.plot(time, filters_mc[n], c='r', label='approx')
#     if n==1:
#         ax2.set_ylabel("gain (MC)", color='r')
#         ax2.tick_params(axis='y', labelcolor='r')
#         lines = [line1[0], line2[0], line3[0]]
#         labels = [l.get_label() for l in lines]
#         ax.legend(lines, labels, loc="upper right")
#     else:
#         ax2.tick_params(labelright=False)
# plt.tight_layout()

# plt.figure()
# plt.plot(error, c='k')
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()