import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from time import perf_counter


def raised_cosine_log_eval(x, ws, n_basis_funcs, width=2., time_scaling=50.):
    """jax only raised cosine log."""
    last_peak = 1 - width / (n_basis_funcs + width - 1)
    peaks = jnp.linspace(0, last_peak, n_basis_funcs)
    delta = peaks[1] - peaks[0]

    x = - x / ws

    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    x = jnp.log(time_scaling * x + 1) / jnp.log(
        time_scaling + 1
    )


    basis_funcs = 0.5 * (
            jnp.cos(
                jnp.clip(
                    np.pi * (x[:, None] - peaks[None]) / (delta * width),
                    -np.pi,
                    np.pi,
                )
            )
            + 1
    )
    return basis_funcs

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def sum_basis_and_dot(weights, dts, ws, n_basis_funcs, width=2., time_scaling=50.):
    """compilable linear-non-linear transform"""
    fx = raised_cosine_log_eval(dts, ws, n_basis_funcs, width, time_scaling)
    return jnp.sum(fx*weights)

# sum_basis_and_dot_vmap = jax.vmap(sum_basis_and_dot, in_axes=(0, 0, None, None), out_axes=0)


@jax.jit
def tot_spk_in_window(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single neuron"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)

@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    return jax.lax.dynamic_slice(array, (i - window_size,), (window_size,))

def scan_fn(lam_s, i):
    dts =slice_array(tot_spikes_new, i, max_window) - jax.lax.dynamic_slice(tot_spikes_new, (i,), (1,))
    idxs = slice_array(neuron_ids_new, i, max_window)
    ll = jax.nn.softplus(
        sum_basis_and_dot(weights[idxs], dts, history_window, n_basis_funcs)+bias)

    lam_s += jnp.sum(ll)
    return jnp.sum(lam_s), None

def scan_fn_log(lam_s, i):
    dts =slice_array(tot_spikes_new, i, max_window) - jax.lax.dynamic_slice(tot_spikes_new, (i,), (1,))
    idxs = slice_array(neuron_ids_new, i, max_window)
    ll = jnp.log(jax.nn.softplus(
        sum_basis_and_dot(weights[idxs], dts, history_window, n_basis_funcs)+bias)
    )
    lam_s += jnp.sum(ll)
    return jnp.sum(lam_s), None

scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
scan_vmap_log = jax.vmap(lambda idxs: jax.lax.scan(scan_fn_log, jnp.array(0), idxs))

# generate data
n_neurons = 10
spk_hz = 200
tot_time_sec = 60
tot_spikes_n = int(tot_time_sec * spk_hz * n_neurons)
history_window = 0.1
n_basis_funcs = 5
print(f"total spikes: {tot_spikes_n}")

np.random.seed(123)

# model parameters
weights = jnp.array(0.01 * np.random.randn(n_neurons*n_basis_funcs)).reshape(n_neurons,n_basis_funcs)
bias = jnp.array(np.random.randn())

# full dataset
tot_spikes = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
neuron_ids = np.random.choice(n_neurons, size=len(tot_spikes))

# postsynaptic neuron
target_neu_id = 0
neu_spk_idx = jnp.arange(len(tot_spikes))[neuron_ids == target_neu_id]
neu_spikes = tot_spikes[neu_spk_idx]

# define max window and adjust indices
max_window = tot_spk_in_window(jnp.array([-history_window,0]), neu_spikes, tot_spikes)
max_window = int(max_window)
print(f"max window: {max_window}")

delta_idx = jax.nn.relu(max_window - neu_spk_idx[0])
update_idx = neu_spk_idx + delta_idx
tot_spikes_new = np.hstack((jnp.full(delta_idx, -history_window - 1), tot_spikes))
neuron_ids_new = np.hstack((jnp.full(delta_idx, 0), neuron_ids))

assert jnp.all(tot_spikes_new[update_idx]==neu_spikes)

n_batches_scan = 10
update_idx = np.hstack((update_idx, jnp.full(-update_idx.size % n_batches_scan, delta_idx)))
update_idx_array = update_idx.reshape(update_idx.size//n_batches_scan,-1)

# first term
# this works
lam_y = 0
for idx in update_idx:
    dts = tot_spikes_new[idx-max_window: idx] - tot_spikes_new[idx]
    w_idxs =neuron_ids_new[idx-max_window: idx]
    lam_y += jnp.log(jax.nn.softplus(sum_basis_and_dot(weights[w_idxs], dts, history_window, n_basis_funcs)+bias))

########
t0 = perf_counter()
out, _ = scan_vmap_log(update_idx_array)

print(jnp.sum(out))
print(np.round(perf_counter()-t0,5))
print(lam_y)
print(np.allclose(jnp.sum(out), lam_y))

# Monte Carlo estimate
# draw M random samples
M = neu_spikes.size
s_m = np.random.choice(tot_spikes[tot_spikes>history_window], size=M)
epsilon_m = np.random.uniform(0, history_window, size=M)
tau_m = s_m + epsilon_m

#compute bounds for history window
mc_window = tot_spk_in_window(jnp.array([-history_window,0]), tau_m, tot_spikes)
mc_idx = jnp.searchsorted(tot_spikes, tau_m,'right')

print(f"mc window: {mc_window}")

delta_mc = jnp.full(-mc_idx.size % n_batches_scan, delta_idx)
mc_idx = np.hstack((mc_idx, delta_mc))
mc_idx_array = mc_idx.reshape(mc_idx.size//n_batches_scan,-1)

# compute 1/M sum_M lambda(tau_m)
mc_sum, _ = scan_vmap(mc_idx_array)
sub, _ = scan_vmap(delta_mc[:,None])

mc_estimate = (jnp.sum(mc_sum) - jnp.sum(sub)) / M

loss = mc_estimate - lam_y

print(loss)