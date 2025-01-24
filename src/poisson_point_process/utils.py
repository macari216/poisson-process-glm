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
    return jax.lax.dynamic_slice(array, (i - window_size,), (window_size,))

def compute_max_window_and_adjust(spike_times, neuron_ids, history_window, target_times, target_idx):
    max_window = tot_spk_in_window(
        jnp.array([-history_window, 0]), target_times, spike_times
    )
    max_window = int(max_window)

    delta_idx = jax.nn.relu(max_window - target_idx[0])
    shifted_idx = target_idx + delta_idx
    tot_spikes_new = jnp.hstack(
        (jnp.full(delta_idx, -history_window - 1), spike_times)
    )
    neuron_ids_new = jnp.hstack((jnp.full(delta_idx, 0), neuron_ids))

    assert jnp.all(tot_spikes_new[shifted_idx] == spike_times[target_idx])

    return max_window, tot_spikes_new, neuron_ids_new, shifted_idx


def reshape_for_vmap(spike_idx, n_batches_scan):
    padding = jnp.full(-spike_idx.size % n_batches_scan, spike_idx[0])
    shifted_idx = np.hstack(
        (spike_idx, padding)
    )
    shifted_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

    return shifted_idx_array, padding
