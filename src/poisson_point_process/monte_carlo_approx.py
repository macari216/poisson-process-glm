from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import slice_array, reshape_for_vmap, compute_max_window_and_adjust


@partial(jax.jit, static_argnums=(2,))
def sum_basis_and_dot(weights, dts, basis_fn):
    """compilable linear-non-linear transform"""
    fx = basis_fn(dts)
    return jnp.sum(fx * weights)


@partial(jax.jit, static_argnums=(3, 4))
def linear_non_linear(dts, weights, bias, basis_fn, inverse_link=jax.nn.softplus):
    ll = inverse_link(sum_basis_and_dot(weights, dts, basis_fn) + bias)
    return ll

def compute_summed_ll(
    spike_times_concat,
    spike_id_concat,
    shifted_idx,
    n_batches_scan,
    max_window_size,
    params,
    basis_fn,
    inverse_link=jax.nn.softplus,
    log=True
):
    if log:
        optional_log = jnp.log
    else:
        optional_log = lambda x: x

    weights, bias = params

    shifted_idx_array, padding = reshape_for_vmap(shifted_idx, n_batches_scan)

    # body of the scan function
    def scan_fn(lam_s, i):
        dts = slice_array(
            spike_times_concat, i+1, max_window_size+1
        ) - jax.lax.dynamic_slice(spike_times_concat, (i,), (1,))
        idxs = slice_array(spike_id_concat, i+1, max_window_size+1)
        ll = optional_log(
            linear_non_linear(dts, weights[idxs], bias, basis_fn, inverse_link)
        )
        lam_s += jnp.sum(ll)
        return jnp.sum(lam_s), None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
    out, _ = scan_vmap(shifted_idx_array)
    sub, _ = scan_vmap(padding[:,None])
    return jnp.sum(out) - jnp.sum(sub)