import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from functools import partial


def raised_cosine_log(x, ws, n_basis_funcs, width=2., time_scaling=50.):
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
    fx = raised_cosine_log(dts, ws, n_basis_funcs, width, time_scaling)
    return jnp.sum(fx*weights)


@jax.jit
def tot_spk_in_window(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single window"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)

# generate data
n_neurons = 10
spk_hz = 100
tot_time_sec = 60
tot_spikes_n = int(tot_time_sec * spk_hz * n_neurons)
history_window = 0.1
n_basis_funcs = 5

np.random.seed(123)

weights = jnp.array(0.01 * np.random.randn(n_neurons*n_basis_funcs)).reshape(n_neurons,n_basis_funcs)

# full dataset
tot_spikes = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
neuron_ids = np.random.choice(n_neurons, size=len(tot_spikes))

# postsynaptic neuron
target_neu_id = 0
neu_spk_idx = jnp.arange(len(tot_spikes))[neuron_ids == target_neu_id]
neu_spikes = tot_spikes[neu_spk_idx]

# define max window and adjust indices
max_window = tot_spk_in_window(jnp.array([-history_window,0]), neu_spikes, tot_spikes)

delta_idx = jax.nn.relu(max_window - neu_spk_idx[0])
update_idx = neu_spk_idx + delta_idx
tot_spikes_new = np.hstack((jnp.full(delta_idx, -history_window - 1), tot_spikes))
neuron_ids_new = np.hstack((jnp.full(delta_idx, 0), neuron_ids))

assert jnp.all(tot_spikes_new[update_idx]==neu_spikes)


# first term
# this works
# +b nado escho vsunut'
lam_y = 0
for idx in update_idx:
    dts = tot_spikes_new[idx-max_window: idx] - tot_spikes_new[idx]
    lam_y += jnp.log(jax.nn.softplus(sum_basis_and_dot(weights[neuron_ids_new[idx-max_window: idx]], dts, history_window, n_basis_funcs)))

# this does not work
# I also don't think it's a better option than lax.scan because it will return
# an array with cumulative sum instead of scalar sum
# assoc_scan_fun = lambda x, y: jnp.add(sum_basis_and_dot(weights, lax.dynamic_slice_in_dim(tot_spikes_new, x-max_window, max_window)),
#                             sum_basis_and_dot(weights, lax.dynamic_slice_in_dim(tot_spikes_new, y-max_window, max_window)))
#
# lam_sum = lax.associative_scan(assoc_scan_fun, update_idx)
#
#
# # this also does not work
# def scan_func(lam_sum, idx):
#     spk_in_w = lax.dynamic_slice_in_dim(tot_spikes_new, idx-max_window,max_window)
#     neu_in_w = lax.dynamic_slice_in_dim(neuron_ids_new, idx-max_window,max_window)
#     spike = lax.dynamic_slice_in_dim(tot_spikes_new, idx,1)
#     dts = spk_in_w - spike
#
#     update = sum_basis_and_dot(weights[neu_in_w], dts, history_window, n_basis_funcs)
#     return lam_sum + jnp.log(jax.nn.softplus(update)), None
#
# lam_y, _ = lax.scan(scan_func, 0, update_idx)

# Monte Carlo estimate
# draw M random samples
M = neu_spikes.size
s_m = np.random.choice(tot_spikes[tot_spikes>history_window], size=M)
epsilon_m = np.random.uniform(0, history_window, size=M)
tau_m = s_m + epsilon_m

#compute bounds for history window
mc_window = tot_spk_in_window(jnp.array([-history_window,0]), tau_m, tot_spikes)
mc_idx = jnp.searchsorted(tot_spikes, tau_m,'right')

# compute 1/M sum_M lambda(tau_m)
mc_sum = 0

# need to replace this with scan too
for i in range(M):
    dts = tot_spikes_new[mc_idx[i]-mc_window: mc_idx[i]] - tau_m[i]
    mc_sum += jax.nn.softplus(sum_basis_and_dot(weights[neuron_ids_new[mc_idx[i]-mc_window: mc_idx[i]]], dts, history_window, n_basis_funcs))
mc_estimate = mc_sum / M

loss = mc_estimate - lam_y

print(loss)