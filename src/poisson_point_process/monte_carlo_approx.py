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
    X_spikes,
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
        spk_in_window = slice_array(
            X_spikes, i + 1, max_window_size + 1
        )

        dts = spk_in_window[0] - jax.lax.dynamic_slice(X_spikes, (0, i), (1, 1))

        ll = optional_log(
            linear_non_linear(dts[0], weights[spk_in_window[1].astype(int)], bias, basis_fn, inverse_link)
        )
        lam_s += jnp.sum(ll)
        return jnp.sum(lam_s), None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
    out, _ = scan_vmap(shifted_idx_array)
    sub, _ = scan_vmap(padding[:,None])
    return jnp.sum(out) - jnp.sum(sub)


def data_ll(X_spikes, y_spikes, history_window, params, n_batches_scan, basis_fn, inverse_link):
    max_window, X_spikes_new, shifted_idx = \
        compute_max_window_and_adjust(
            X_spikes,
            history_window,
            y_spikes
        )

    log_lam_y = compute_summed_ll(
        X_spikes_new.copy(),
        shifted_idx,
        n_batches_scan,
        max_window,
        params,
        basis_fn=basis_fn,
        inverse_link=inverse_link,
        log=True
    )

    return log_lam_y


def norm_mc_approx(X_spikes, M, history_window, params, n_batches_scan, basis_fn, inverse_link):
    s_m = np.random.choice(X_spikes[X_spikes[:, 0] < X_spikes[-1, 0] - history_window, 0], size=M)
    epsilon_m = np.random.uniform(0, history_window, size=M)
    tau_m = s_m + epsilon_m
    tau_m_idx = jnp.searchsorted(X_spikes[:,0], tau_m, "right")
    mc_spikes = jnp.vstack((tau_m, tau_m_idx)).T

    mc_window, X_spikes_new, shifted_mc_idx = \
        compute_max_window_and_adjust(
            X_spikes,
            history_window,
            mc_spikes
        )

    mc_sum = compute_summed_ll(
        X_spikes_new,
        shifted_mc_idx,
        n_batches_scan,
        mc_window,
        params,
        basis_fn=basis_fn,
        inverse_link=inverse_link,
        log=False
    )

    return mc_sum / M

def negative_log_likelihood(X_spikes, y_spikes, history_window, params, n_batches_scan, basis_fn, inverse_link):
    log_lam_y = data_ll(X_spikes, y_spikes, history_window, params, n_batches_scan, basis_fn, inverse_link)
    mc_estimate = norm_mc_approx(X_spikes, y_spikes.shape[1], history_window, params, n_batches_scan, basis_fn, inverse_link)

    return mc_estimate - log_lam_y