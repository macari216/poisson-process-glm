import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.monte_carlo_approx import negative_log_likelihood
from poisson_point_process.utils import compute_max_window_size
from poisson_point_process.basis import raised_cosine_log_eval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# generate data
tot_time_sec = 100
binsize = 0.0001
n_neurons = 15
target_neu_id = 0
history_window= 0.004
window_size = int(history_window/binsize)
n_basis_funcs = 8
rc_basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs, "conv", window_size=window_size)
_, kernels = rc_basis.evaluate_on_grid(window_size)
n_batches_scan = 5
inverse_link = jax.nn.softplus

# true parameters
biases = jnp.array(-8 + np.random.randn(n_neurons))
weights =  jnp.array(0.2 * np.random.normal(size=(n_neurons, n_neurons, n_basis_funcs)))
params = (weights, biases)

# all-to-all connectivity (bias and weights)
t0 = perf_counter()
init_spikes = np.zeros((window_size, n_neurons))
spike_counts = simulate.generate_poisson_counts_recurrent(tot_time_sec, binsize, n_neurons, kernels, params, init_spikes, inverse_link)

print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
print(f"max spikes per bin: {spike_counts.max()}")
print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(spike_counts.sum(0))}")
plt.hist(spike_counts.sum(0), bins=50)
plt.show()

spike_times, spike_ids = simulate.generate_poisson_times(spike_counts, tot_time_sec, binsize)

# spikes_tsgroup = nap.TsGroup({n: nap.Ts(np.array(spike_times[neuron_indices == n]))
#                               for n in range(n_neurons)},
#                                 nap.IntervalSet(0, tot_time_sec))

X_spikes = jnp.vstack((spike_times,spike_ids))
spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))
print(f"generated data: {np.round(perf_counter() - t0, 5)}")

# model params
w_hat = jnp.array(0.01 * np.random.randn(n_neurons * n_basis_funcs)).reshape(
    n_neurons, n_basis_funcs
)
bias_hat = jnp.array(np.random.randn())
model_params = (w_hat, bias_hat)

# function for evaluating basis functions at tau
basis_fn = lambda delta_ts: raised_cosine_log_eval(
    delta_ts, history_window, n_basis_funcs, width=2, time_scaling=50
)

# maximum number of spikes in window across the entire dataset
max_window = int(compute_max_window_size(
        jnp.array([-history_window, 0]), X_spikes[0], X_spikes[0]
    ))

# compute neg ll with given X, y, params
t0 = perf_counter()
neg_ll = negative_log_likelihood(X_spikes, y_spikes, model_params, history_window, max_window, basis_fn, n_batches_scan, inverse_link)
print(f"computed neg log likelihood: {np.round(perf_counter() - t0, 5)}")

# loss_grad = jax.jit(
#     jax.grad(negative_log_likelihood, argnums=2),
#     static_argnums=(3,4,5,6,7)
# )

# compute one gradient step
loss_grad = jax.grad(negative_log_likelihood, argnums=2)

t0 = perf_counter()
grad = loss_grad(X_spikes, y_spikes, model_params, history_window, max_window, basis_fn, n_batches_scan, inverse_link)
print(f"computed one grad step: {np.round(perf_counter() - t0, 5)}")

print(grad[0].shape, grad[1])
