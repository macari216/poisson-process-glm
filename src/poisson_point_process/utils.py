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

def precompute_spike_indices(spike_ids, neuron_ids, max_spikes):
    def one_neuron_indices(target_id):
        return jnp.nonzero(spike_ids == target_id, size=max_spikes, fill_value=spike_ids.size + 5)[0]
    #
    # post_indices = []
    # for target_id in neuron_ids:
    #     post_indices.append(jnp.nonzero(spike_ids == target_id, size=max_spikes, fill_value=spike_ids.size + 5)[0])

    batch_size = 10
    post_indices = []

    for i in range(0, len(neuron_ids), batch_size):
        batch_ids = jnp.arange(i, min(i + batch_size, len(neuron_ids)))
        batched = jax.vmap(one_neuron_indices)(batch_ids)
        post_indices.append(batched)

    return jnp.concatenate(post_indices, axis=0)

    # return jnp.stack(post_indices)

def compute_chebyshev(f, approx_interval, power=2, dx=0.01):
    """jax only implementation"""
    xx = jnp.arange(approx_interval[0] + dx / 2.0, approx_interval[1], dx)
    nx = xx.shape[0]
    xxw = jnp.arange(-1.0 + 1.0 / nx, 1.0, 1.0 / (0.5 * nx))
    Bx = jnp.zeros([nx, power + 1])
    for i in range(0, power + 1):
        Bx = Bx.at[:, i].set(jnp.power(xx, i))
    errwts_cheby = 1.0 / jnp.sqrt(1 - xxw ** 2)
    Dx = jnp.diag(errwts_cheby)
    fx = f(xx)
    coef_cheby = jnp.linalg.lstsq(Bx.T @ Dx @ Bx, Bx.T @ Dx @ fx, rcond=None)[0]
    return coef_cheby

def quadratic(x, f, interval):
    coefs = compute_chebyshev(f, interval, power=2, dx=0.01)
    return coefs[0]+coefs[1]*x + coefs[2]*(x**2)

def comb(N, k):
    return jax.lax.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N + 1 - k))

def jax_pairs(neuron_ids):
    return jnp.array(list(combinations(neuron_ids, 2)))