from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import slice_array, reshape_for_vmap, adjust_indices_and_spike_times

@partial(jax.jit, static_argnums=(2,))
def sum_basis_and_dot(weights, dts, basis_fn):
    """compilable linear-non-linear transform"""
    fx = basis_fn(dts)
    return jnp.sum(fx * weights)


@partial(jax.jit, static_argnums=(3, 4))
def linear_non_linear(dts, weights, bias, basis_fn, inverse_link=jax.nn.softplus):
    ll = inverse_link(sum_basis_and_dot(weights, dts, basis_fn) + bias)
    return ll

def draw_mc_sample_points(X_spikes, M, history_window):
    s_m = np.random.choice(X_spikes[0, X_spikes[0] < X_spikes[0, -1] - history_window], size=M)
    epsilon_m = np.random.uniform(0, history_window, size=M)
    tau_m = s_m + epsilon_m
    tau_m_idx = jnp.searchsorted(X_spikes[0], tau_m, "right")
    mc_spikes = jnp.vstack((tau_m, tau_m_idx))

    return mc_spikes

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
    optional_log = jnp.log if log else lambda x: x

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

def compute_log_likelihood_term(X_spikes, y_spikes, history_window, max_window, params, n_batches_scan, basis_fn, inverse_link, optional_log=True):
    X_spikes_new, shifted_idx = \
        adjust_indices_and_spike_times(
            X_spikes,
            y_spikes,
            history_window,
            max_window
        )

    summed_ll = compute_summed_ll(
        X_spikes_new,
        shifted_idx,
        n_batches_scan,
        max_window,
        params,
        basis_fn=basis_fn,
        inverse_link=inverse_link,
        log=optional_log
    )

    return summed_ll

def negative_log_likelihood(X_spikes, y_spikes, params, history_window, max_window, basis_fn, n_batches_scan=5, inverse_link=jax.nn.softplus):
    log_lam_y = compute_log_likelihood_term(X_spikes, y_spikes, history_window, max_window, params,
                                            n_batches_scan, basis_fn, inverse_link, optional_log=True)
    mc_samples = draw_mc_sample_points(X_spikes, y_spikes.shape[1], history_window)
    mc_estimate = compute_log_likelihood_term(X_spikes, mc_samples, history_window, max_window, params,
                                            n_batches_scan, basis_fn, inverse_link, optional_log=False)


    return mc_estimate/y_spikes.shape[1] - log_lam_y