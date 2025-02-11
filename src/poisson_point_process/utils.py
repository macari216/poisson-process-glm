from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def compute_max_window_size(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single neuron"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)


@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    return jax.lax.dynamic_slice(array, (0,i - window_size), (2,window_size,))

@partial(jax.jit, static_argnums=(2,3))
def adjust_indices_and_spike_times(X, y, history_window, max_window):
    shifted_idx = y[1].astype(int) + max_window
    shifted_y =  jnp.vstack((y[0], shifted_idx))

    shift = jnp.vstack((jnp.full(max_window, -history_window - 1), jnp.full(max_window, 0)))
    shifted_X = jnp.hstack((shift, X))

    return shifted_X, shifted_y

@partial(jax.jit, static_argnums=1)
def reshape_for_vmap(spike_idx, n_batches_scan):
    padding = jnp.full(-spike_idx.size % n_batches_scan, spike_idx[0])
    shifted_idx = jnp.hstack(
        (spike_idx, padding)
    )
    shifted_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

    return shifted_idx_array, padding
