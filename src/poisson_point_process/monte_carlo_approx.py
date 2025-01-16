
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
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

@partial(jax.jit, static_argnums=(2, ))
def sum_basis_and_dot(weights, dts, basis_fn):
    """compilable linear-non-linear transform"""
    fx = basis_fn(dts)
    return jnp.sum(fx*weights)


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

@partial(jax.jit, static_argnums=(3, 4))
def linear_non_linear(dts, weights, bias, basis_fn, inverse_link=jax.nn.softplus):
    ll = inverse_link(
        sum_basis_and_dot(weights, dts, basis_fn) + bias
    )
    return ll

def compute_unnormalized_log_likelihood(
        weights,
        bias,
        spike_times_concat,
        spike_id_concat,
        spike_times_neuron,
        spike_index_neuron,
        window_size,
        basis_fn,
        n_batches_scan=1,
        inverse_link=jax.nn.softplus,
):
    max_window = tot_spk_in_window(jnp.array([-window_size, 0]), spike_times_neuron, spike_times_concat)
    max_window = int(max_window)

    # define max window and adjust indices
    delta_idx = jax.nn.relu(max_window - spike_index_neuron[0])
    shifted_idx = spike_index_neuron + delta_idx

    spike_times_concat = np.hstack((jnp.full(delta_idx, -window_size - 1), spike_times_concat))
    spike_id_concat = np.hstack((jnp.full(delta_idx, 0), spike_id_concat))

    shifted_idx = np.hstack((shifted_idx, jnp.full(-shifted_idx.size % n_batches_scan, delta_idx)))
    update_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

    # body of the scan function
    def scan_fn(lam_s, i):
        dts = slice_array(spike_times_concat, i, max_window) - jax.lax.dynamic_slice(spike_times_concat, (i,), (1,))
        idxs = slice_array(spike_id_concat, i, max_window)
        ll = linear_non_linear(dts, weights[idxs], bias, basis_fn, inverse_link)
        lam_s += jnp.sum(ll)
        return jnp.sum(lam_s), None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
    out, _ = scan_vmap(update_idx_array)
    return out