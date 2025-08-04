from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
from numpy.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from itertools import combinations


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

@partial(jax.jit, static_argnums=(1,2))
def adjust_indices_and_spike_times(
        X: ArrayLike,
        history_window: float,
        max_window: int,
        y: Optional=None,
):
    shift = jnp.vstack((jnp.full(max_window, -history_window - 1), jnp.full(max_window, 0)))
    shifted_X = jnp.hstack((shift, X))
    if y is not None:
        shifted_idx = y[-1].astype(int) + max_window
        shifted_y = jnp.vstack((y[:-1], shifted_idx))
        return shifted_X, shifted_y
    else:
        return shifted_X

@partial(jax.jit, static_argnums=1)
def reshape_for_vmap(spikes, n_batches_scan):
    padding_shape = (-spikes.shape[1] % n_batches_scan,)
    padding = jnp.full((spikes.shape[0],) + padding_shape, spikes[:, :1])
    shifted_spikes = jnp.hstack(
        (spikes, padding)
    )
    shifted_spikes_array = shifted_spikes.reshape(spikes.shape[0],n_batches_scan,-1).transpose(1,2,0)

    return shifted_spikes_array, padding.transpose(1,0)

def concat_params(params):
    if len(params[0].shape) == 1:
        return jnp.vstack((params[0][:,None], params[1][:,None]))
    elif len(params[0].shape) == 2:
        return jnp.vstack((params[0], params[1]))
    else:
        raise ValueError(
            f"Weights must be either 1d or 2d array, the provided weights have shape {params[0].shape}"
        )

def reshape_w(weights, n_basis_funcs):
    if len(weights.shape) == 1:
        return weights.reshape(-1, n_basis_funcs, 1)
    elif len(weights.shape) == 2:
        n_target_neurons = weights.shape[1]
        return weights.reshape(-1, n_basis_funcs, n_target_neurons)
    else:
        raise ValueError(
            f"Weights must be either 1d or 2d array, the provided weights have shape {weights.shape}"
        )

def compute_chebyshev(f, approx_intervals, power=2, nx=1000):
    """compute chebyshev polynomial coefficients for
    approximating nonlinearity on an array of given intervals"""
    def compute_chebyshev_single(start, end):
        xxw = jnp.arange(-1.0 + 1.0 / nx, 1.0, 1.0 / (0.5 * nx))
        xx = 0.5 * (end - start) * xxw + 0.5 * (start + end)
        Bx = jnp.zeros([nx, power + 1])
        for i in range(0, power + 1):
            Bx = Bx.at[:, i].set(jnp.power(xx, i))
        errwts_cheby = 1.0 / jnp.sqrt(1 - xxw ** 2)
        Dx = jnp.diag(errwts_cheby)
        fx = f(xx)
        coef_cheby = jnp.linalg.lstsq(Bx.T @ Dx @ Bx, Bx.T @ Dx @ fx, rcond=None)[0]
        return coef_cheby.T
    starts, ends = approx_intervals
    return jax.vmap(compute_chebyshev_single, in_axes=(0,0), out_axes=1)(jnp.atleast_1d(starts), jnp.atleast_1d(ends))

def quadratic(x, f, interval):
    coefs = compute_chebyshev(f, interval)
    return coefs[0]+coefs[1]*x + coefs[2]*(x**2)

def comb(N, k):
    return jax.lax.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N + 1 - k))

def std_laguerre_binom(i, j):
    binom_l = jnp.where(j <= i, comb(i, j), 0.0)
    binom_b = jnp.where(j <= i, comb(i, j), 0.0)
    return binom_l, binom_b

def gen_laguerre_binom(i, j, i_a, j_a, alpha):
    binom_l = jnp.where(j <= i, comb(i+alpha, i-j), 0.0)
    binom_b = jnp.where(j_a <= i_a, comb(i_a, j_a), 0.0)
    return binom_l, binom_b

def jax_pairs(neuron_ids):
    return jnp.array(list(combinations(neuron_ids, 2)))