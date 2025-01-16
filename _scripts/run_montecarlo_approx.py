import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from poisson_point_process import (
    compute_unnormalized_log_likelihood,
    raised_cosine_log_eval,
)
from poisson_point_process.monte_carlo_approx import (
    sum_basis_and_dot,
)
from poisson_point_process.utils import tot_spk_in_window

jax.config.update("jax_enable_x64", True)

os.environ["JAX_PLATFORM_NAME"] = "cpu"
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
weights = jnp.array(0.01 * np.random.randn(n_neurons * n_basis_funcs)).reshape(
    n_neurons, n_basis_funcs
)
bias = jnp.array(np.random.randn())

# full dataset
spike_times_concat = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
spike_id_concat = np.random.choice(n_neurons, size=len(spike_times_concat))

# postsynaptic neuron
target_neu_id = 0
neu_spk_idx = jnp.arange(len(spike_times_concat))[spike_id_concat == target_neu_id]
spike_times_neuron = spike_times_concat[neu_spk_idx]

# define max window and adjust indices
max_window = tot_spk_in_window(
    jnp.array([-history_window, 0]), spike_times_neuron, spike_times_concat
)
max_window = int(max_window)
print(f"max window: {max_window}")

delta_idx = jax.nn.relu(max_window - neu_spk_idx[0])
shifted_idx = neu_spk_idx + delta_idx
tot_spikes_new = np.hstack(
    (jnp.full(delta_idx, -history_window - 1), spike_times_concat)
)
neuron_ids_new = np.hstack((jnp.full(delta_idx, 0), spike_id_concat))

assert jnp.all(tot_spikes_new[shifted_idx] == spike_times_neuron)

n_batches_scan = 10
shifted_idx = np.hstack(
    (shifted_idx, jnp.full(-shifted_idx.size % n_batches_scan, delta_idx))
)
update_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

basis_fn = lambda delta_ts: raised_cosine_log_eval(
    delta_ts, history_window, n_basis_funcs, width=2, time_scaling=50
)

# first term
# this works
lam_y = 0
for idx in shifted_idx:
    dts = tot_spikes_new[idx - max_window : idx] - tot_spikes_new[idx]
    w_idxs = neuron_ids_new[idx - max_window : idx]
    lam_y += jnp.log(
        jax.nn.softplus(sum_basis_and_dot(weights[w_idxs], dts, basis_fn) + bias)
    )

########
t0 = perf_counter()
out, scan_vmapped = compute_unnormalized_log_likelihood(
    weights,
    bias,
    spike_times_concat.copy(),
    spike_id_concat,
    neu_spk_idx,
    history_window,
    max_window,
    basis_fn=basis_fn,
    n_batches_scan=10,
    inverse_link=jax.nn.softplus,
)

print(jnp.sum(out))
print(np.round(perf_counter() - t0, 5))
print(lam_y)
print(np.allclose(jnp.sum(out), lam_y))

# Monte Carlo estimate
# draw M random samples
M = spike_times_neuron.size
s_m = np.random.choice(spike_times_concat[spike_times_concat > history_window], size=M)
epsilon_m = np.random.uniform(0, history_window, size=M)
tau_m = s_m + epsilon_m

# compute bounds for history window
mc_window = tot_spk_in_window(
    jnp.array([-history_window, 0]), tau_m, spike_times_concat
)
mc_idx = jnp.searchsorted(spike_times_concat, tau_m, "right")


print(f"mc window: {mc_window}")

delta_mc = jnp.full(-mc_idx.size % n_batches_scan, delta_idx)
mc_idx = np.hstack((mc_idx, delta_mc))
mc_idx_array = mc_idx.reshape(mc_idx.size // n_batches_scan, -1)


# compute 1/M sum_M lambda(tau_m)
mc_sum, _ = scan_vmapped(mc_idx_array)
sub, _ = scan_vmapped(delta_mc[:, None])

mc_estimate = (jnp.sum(mc_sum) - jnp.sum(sub)) / M

loss = mc_estimate - lam_y

print(loss)
