import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

n_neurons = 9
target_neu_id = 0
history_window = 0.1
tot_time_sec = 1000
binsize = 0.001
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window
inverse_link = jnp.exp

# n_neurons = 1
# target_neu_id = 1
# history_window = 0.1
# tot_time_sec = 10
# binsize = 0.001
# window_size = int(history_window / binsize)
# n_basis_funcs = 2
# n_bins_tot = int(tot_time_sec / binsize)
# n_batches_scan = 1
# rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
# time, kernels = rc_basis.evaluate_on_grid(window_size)
# time *= history_window
# inverse_link = jnp.exp

np.random.seed(216)

baseline_fr = 2.1
biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons))) + np.log(binsize)
bias_true = biases[target_neu_id]
posts_rate_per_sec = biases[target_neu_id] - np.log(binsize)

weights_true = jnp.array(np.random.normal(0, 0.1, size=(n_neurons, n_neurons, n_basis_funcs)))
params = (weights_true, biases)

spike_counts, rates = simulate.poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, kernels, params,
                                                        inverse_link)

print(f"total spikes: {spike_counts.sum()}, y spikes: {spike_counts[:, target_neu_id].sum()}")
print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
print(f"max spikes per bin: {spike_counts.max()}")
print(f"ISI (ms): {tot_time_sec * 1000 / jnp.mean(spike_counts.sum(0))}")

spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize)

X_spikes = jnp.vstack((spike_times, spike_ids))
spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

y_counts = spike_counts[:, target_neu_id]

num_iter = 1
grad_sample = 500
warmup = 0
key = jax.random.PRNGKey(1)
all_keys = jax.random.split(key, int((num_iter-warmup)*grad_sample)).reshape(num_iter-warmup, grad_sample, -1)
grad_dict = {}
# M_size = [100, 250, 500, 750, 1000, 2000, 3000, 5000, 7500, 10000, 20000, 50000]
M_size = [2000, 3000, 5000, 7500, 10000, 20000, 35000, 50000]
for c, M in enumerate(M_size):
    obs_model_kwargs = {
        "n_basis_funcs": n_basis_funcs,
        "history_window": history_window,
        "inverse_link_function": inverse_link,
        "n_batches_scan": n_batches_scan,
        "mc_random_key": jax.random.PRNGKey(0),
        "mc_n_samples": M,
    }

    model = ContinuousMC(solver_name="GradientDescent", obs_model_kwargs=obs_model_kwargs,
                         solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 1e-7})

    true_params = (weights_true[target_neu_id].ravel(), jnp.array([posts_rate_per_sec]))
    params = model.initialize_params(X_spikes, y_spikes)
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