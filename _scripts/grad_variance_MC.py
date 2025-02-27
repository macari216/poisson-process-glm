import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.utils import compute_max_window_size, adjust_indices_and_spike_times
from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# n_neurons = 100
# target_neu_id = 0
# history_window = 0.3
# tot_time_sec = 1000
# binsize = 0.01
# window_size = int(history_window / binsize)
# n_basis_funcs = 8
# n_bins_tot = int(tot_time_sec / binsize)
# n_batches_scan = 1
# inverse_link = jax.nn.softplus
# rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
# time, kernels = rc_basis.evaluate_on_grid(window_size)
# time *= history_window

n_neurons = 1
target_neu_id = 1
history_window = 0.1
tot_time_sec = 10
binsize = 0.001
window_size = int(history_window / binsize)
n_basis_funcs = 2
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window
inverse_link = jnp.exp

np.random.seed(216)

# all-to-one connectivity
pres_rate_per_sec = 2
posts_rate_per_sec = 0.1
# rescaled proportionally to the binsize
bias_true = posts_rate_per_sec + np.log(binsize)
# bias_true = jnp.log(jnp.exp(binsize*jax.nn.softplus(posts_rate_per_sec))-1)
# weights_true = np.random.normal(0, 0.3, n_neurons * n_basis_funcs)
weights_true = jnp.array([5., 3.])
X, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                                           n_bins_tot, n_neurons, weights_true, window_size, rc_basis,
                                                           inverse_link)
inv_rates = np.log(np.exp(lam_posts) - 1)

print(f"mean spikes per neuron: {jnp.mean(jnp.hstack((y_counts[:, None], X_counts)).sum(0))}")
print(f"max spikes per bin: {jnp.hstack((y_counts[:, None], X_counts)).max()}")
print(f"ISI (ms): {tot_time_sec * 1000 / jnp.mean(jnp.hstack((y_counts[:, None], X_counts)).sum(0))}")

spike_times, spike_ids = simulate.poisson_times(jnp.hstack((X_counts, y_counts[:, None])), tot_time_sec, binsize)
X_spikes = jnp.vstack((spike_times, spike_ids))
spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

rc_basis.set_input_shape(y_counts)

X_discrete = jnp.hstack((X, rc_basis.compute_features(y_counts)))
y_discrete = y_counts

# baseline_fr = 2.1
# biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons))) + np.log(binsize)
# bias_true = biases[target_neu_id]
#
# weights_true = jnp.array(np.random.normal(0, 0.01, size=(n_neurons, n_neurons, n_basis_funcs)))
# params = (weights_true, biases)
#
# spike_counts, _ = simulate.poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, kernels, params, inverse_link)
#
# print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
# print(f"max spikes per bin: {spike_counts.max()}")
# print(f"ISI (ms): {tot_time_sec * 1000 / jnp.mean(spike_counts.sum(0))}")
#
# spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize)
# X_spikes = jnp.vstack((spike_times, spike_ids))
# spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
# y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

max_window = int(compute_max_window_size(
        jnp.array([-history_window, 0]), X_spikes[0], X_spikes[0]
    ))

X_shifted, y_shifted = adjust_indices_and_spike_times(X_spikes, y_spikes, history_window, max_window)
X_spikes, y_spikes = X_shifted, y_shifted

num_iter = 1
grad_sample = 500
warmup = 0
key = jax.random.PRNGKey(1)
all_keys = jax.random.split(key, int((num_iter-warmup)*grad_sample)).reshape(num_iter-warmup, grad_sample, -1)
grad_dict = {}
#M_size = [100, 500, 1000, 5000, 10000, 20000, 50000]
M_size = [100, 250, 500, 750, 1000, 2000, 3000, 5000, 7500, 10000]
for c, M in enumerate(M_size):
    obs_model_kwargs = {
        "n_basis_funcs": n_basis_funcs,
        "history_window": history_window,
        "inverse_link_function": inverse_link,
        "max_window": max_window,
        "n_batches_scan": n_batches_scan,
        "mc_random_key": jax.random.PRNGKey(0),
        "mc_n_samples": M,
        "total time": tot_time_sec
    }

    model = ContinuousMC(solver_name="GradientDescent", obs_model_kwargs=obs_model_kwargs,
                         solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 0.00005})

    params = model.initialize_params(X_spikes, y_spikes, init_params=(jnp.array([5.,3.]), jnp.array([0.1])))
    state = model.initialize_state(X_spikes, y_spikes, params)

    loss_grad = jax.vmap(jax.grad(lambda p, k: model._predict_and_compute_loss(p, X_spikes, y_spikes, k), has_aux=True), in_axes=(None, 0))
    for step in range(num_iter):
        params, state = model.update(params, state, X_spikes, y_spikes)
        if step >= warmup:
            t0 = perf_counter()
            grads_0, grads_1 = loss_grad(params, all_keys[step-warmup])[0]
            print(f"step {step} completed, {perf_counter() - t0}")

    grad_dict[f"{M} w"] = jnp.array(grads_0)
    grad_dict[f"{M} b"] = jnp.array(grads_1)

    print(f"fit M={M}, {c+1}/{len(M_size)}")
    print(repr(jnp.var(jnp.array(grads_0), axis=0)))
    print(repr(jnp.var(jnp.array(grads_1), axis=0)))

# np.savez(f"/mnt/home/amedvedeva/ceph/var_grads.npz", **grad_dict)

# print("Script terminated")


# var_grads = np.load("/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbird_output/var_grads.npz", allow_pickle=True)
w_grad = []
b_grad = []
for i, key in enumerate(grad_dict.keys()):
    if i%2==0:
        w_grad.append(grad_dict[key])
    else:
        b_grad.append(grad_dict[key])

w_var = jnp.var(jnp.array(w_grad),axis=1)
b_var = jnp.var(jnp.array(b_grad),axis=1)
#
# plt.figure()
# plt.plot(M_size, b_var, label="bias")
# for n in range(6):
#     plt.plot(M_size, w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2)[:,n], label=f"n{n} weights")
# plt.vlines(y_counts.sum(),0, w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2).max(), color='k', label='K')
# # plt.xlim(0,8000)
# plt.xlabel("MC sample size")
# plt.ylabel("grad variance")
# plt.legend()

mean_w = w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2).mean(1)
se_w = w_var.reshape(len(M_size),-1,n_basis_funcs).mean(2).std(1) / np.sqrt(n_neurons)
plt.figure()
plt.plot(M_size, b_var, label="bias")
plt.plot(M_size, mean_w,label=f"weights", c='r')
plt.fill_between(M_size, mean_w - se_w, mean_w+se_w, alpha=0.3, color='r')
plt.vlines(y_counts.sum(),0, mean_w.max(), color='k', label='K')
# plt.xlim(0,8000)
plt.xlabel("MC sample size")
plt.ylabel("grad variance")
plt.legend()

plt.show()