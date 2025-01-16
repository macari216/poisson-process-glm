from functools import partial
import jax
import jax.numpy as jnp

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

