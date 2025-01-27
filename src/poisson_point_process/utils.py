from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def tot_spk_in_window(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single neuron"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)


@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    return jax.lax.dynamic_slice(array, (0,i - window_size), (2,window_size,))

def compute_max_window_and_adjust(X_spikes, history_window, y_spikes):
    max_window = tot_spk_in_window(
        jnp.array([-history_window, 0]), y_spikes[0], X_spikes[0]
    )
    max_window = int(max_window)

    delta_idx = jax.nn.relu(max_window - y_spikes[1,0].astype(int))
    shifted_idx = y_spikes[1].astype(int) + delta_idx

    shift = jnp.vstack((jnp.full(delta_idx, -history_window - 1), jnp.full(delta_idx, 0)))

    X_spikes_new = jnp.hstack((shift, X_spikes))

    assert jnp.all(X_spikes_new[0,shifted_idx] == X_spikes[0, y_spikes[1].astype(int)])

    return max_window, X_spikes_new, shifted_idx


def reshape_for_vmap(spike_idx, n_batches_scan):
    padding = jnp.full(-spike_idx.size % n_batches_scan, spike_idx[0])
    shifted_idx = np.hstack(
        (spike_idx, padding)
    )
    shifted_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

    return shifted_idx_array, padding
