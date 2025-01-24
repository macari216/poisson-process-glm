import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from poisson_point_process.monte_carlo_approx import sum_basis_and_dot, compute_summed_ll
from poisson_point_process import utils
from poisson_point_process.basis import raised_cosine_log_eval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# generate data
n_neurons = 10
target_neu_id = 0
spk_hz = 200
tot_time_sec = 60
tot_spikes_n = int(tot_time_sec * spk_hz * n_neurons)
history_window = 0.1
n_basis_funcs = 5
n_batches_scan = 5
print(f"total spikes: {tot_spikes_n}")

np.random.seed(123)

# generate uniform spikes
spike_times_concat = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
spike_id_concat = np.random.choice(n_neurons, size=len(spike_times_concat))
all_spikes = jnp.vstack((spike_times_concat,spike_id_concat)).T
spike_idx_target = jnp.arange(len(spike_times_concat))[spike_id_concat == target_neu_id]
spike_times_target = spike_times_concat[spike_idx_target]
y_spikes = jnp.vstack((spike_times_target, spike_idx_target)).T

# compute scan window and adjust indices
max_window, tot_spikes_new, neuron_ids_new, shifted_idx = \
    utils.compute_max_window_and_adjust(
        spike_times_concat,
        spike_id_concat,
        history_window,
        spike_times_target,
        spike_idx_target
    )

# model parameters
weights = jnp.array(0.01 * np.random.randn(n_neurons * n_basis_funcs)).reshape(
    n_neurons, n_basis_funcs
)
bias = jnp.array(np.random.randn())
params = (weights,bias)

basis_fn = lambda delta_ts: raised_cosine_log_eval(
    delta_ts, history_window, n_basis_funcs, width=2, time_scaling=50
)

#### first term
lam_y_loop = 0
for idx in shifted_idx:
    dts = tot_spikes_new[idx - max_window : idx+1] - tot_spikes_new[idx]
    w_idxs = neuron_ids_new[idx - max_window : idx+1]
    lam_y_loop += jnp.log(
        jax.nn.softplus(sum_basis_and_dot(weights[w_idxs], dts, basis_fn) + bias)
    )

########
t0 = perf_counter()
log_lam_y = compute_summed_ll(
    tot_spikes_new.copy(),
    neuron_ids_new,
    shifted_idx,
    n_batches_scan,
    max_window,
    params,
    basis_fn=basis_fn,
    inverse_link=jax.nn.softplus,
    log=True
)

print(log_lam_y)
print(np.round(perf_counter() - t0, 5))
print(lam_y_loop)
print(np.allclose(jnp.sum(log_lam_y), lam_y_loop))

# Monte Carlo estimate
# draw M random samples
M = spike_times_target.size
s_m = np.random.choice(spike_times_concat[spike_times_concat<=tot_time_sec-history_window], size=M)
epsilon_m = np.random.uniform(0, history_window, size=M)
tau_m = s_m + epsilon_m
tau_m_idx = jnp.searchsorted(spike_times_concat, tau_m, "right")

mc_window, tot_spikes_new, neuron_ids_new, shifted_mc_idx = \
    utils.compute_max_window_and_adjust(
        spike_times_concat,
        spike_id_concat,
        history_window,
        tau_m,
        tau_m_idx
    )

mc_sum = compute_summed_ll(
    tot_spikes_new.copy(),
    neuron_ids_new,
    shifted_mc_idx,
    n_batches_scan,
    mc_window,
    params,
    basis_fn=basis_fn,
    inverse_link=jax.nn.softplus,
    log=False
)

mc_estimate = mc_sum / M

neg_ll = mc_estimate - log_lam_y

print(neg_ll)
