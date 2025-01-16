from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from.utils import slice_array

@partial(jax.jit, static_argnums=(2,))
def sum_basis_and_dot(weights, dts, basis_fn):
    """compilable linear-non-linear transform"""
    fx = basis_fn(dts)
    return jnp.sum(fx * weights)


@partial(jax.jit, static_argnums=(3, 4))
def linear_non_linear(dts, weights, bias, basis_fn, inverse_link=jax.nn.softplus):
    ll = inverse_link(sum_basis_and_dot(weights, dts, basis_fn) + bias)
    return ll


def compute_unnormalized_log_likelihood(
    weights,
    bias,
    spike_times_concat,
    spike_id_concat,
    neu_spk_idx,
    history_window_size,
    max_window_size,
    basis_fn,
    n_batches_scan=1,
    inverse_link=jax.nn.softplus,
):
    max_window_size = int(max_window_size)

    # define max window and adjust indices
    delta_idx = jax.nn.relu(max_window_size - neu_spk_idx[0])
    shifted_idx = neu_spk_idx + delta_idx

    spike_times_concat = np.hstack(
        (jnp.full(delta_idx, -history_window_size - 1), spike_times_concat)
    )
    spike_id_concat = np.hstack((jnp.full(delta_idx, 0), spike_id_concat))

    shifted_idx = np.hstack(
        (shifted_idx, jnp.full(-shifted_idx.size % n_batches_scan, delta_idx))
    )
    update_idx_array = shifted_idx.reshape(shifted_idx.size // n_batches_scan, -1)

    # body of the scan function
    def scan_fn(lam_s, i):
        dts = slice_array(
            spike_times_concat, i, max_window_size
        ) - jax.lax.dynamic_slice(spike_times_concat, (i,), (1,))
        idxs = slice_array(spike_id_concat, i, max_window_size)
        ll = jnp.log(
            linear_non_linear(dts, weights[idxs], bias, basis_fn, inverse_link)
        )
        lam_s += jnp.sum(ll)
        return jnp.sum(lam_s), None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
    out, _ = scan_vmap(update_idx_array)
    return out, scan_vmap
