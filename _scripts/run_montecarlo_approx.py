import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from poisson_point_process.monte_carlo_approx import negative_log_likelihood
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
inverse_link = jax.nn.softplus
print(f"total spikes: {tot_spikes_n}")

np.random.seed(123)

# generate uniform spikes
spike_times_concat = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
spike_id_concat = np.random.choice(n_neurons, size=len(spike_times_concat))
X_spikes = jnp.vstack((spike_times_concat,spike_id_concat))
spike_idx_target = jnp.arange(len(spike_times_concat))[spike_id_concat == target_neu_id]
spike_times_target = spike_times_concat[spike_idx_target]
y_spikes = jnp.vstack((spike_times_target, spike_idx_target))

# model parameters
weights = jnp.array(0.01 * np.random.randn(n_neurons * n_basis_funcs)).reshape(
    n_neurons, n_basis_funcs
)
bias = jnp.array(np.random.randn())
params = (weights,bias)

basis_fn = lambda delta_ts: raised_cosine_log_eval(
    delta_ts, history_window, n_basis_funcs, width=2, time_scaling=50
)

neg_ll = negative_log_likelihood(X_spikes, y_spikes, history_window, params, n_batches_scan, basis_fn, inverse_link)

print(neg_ll)
