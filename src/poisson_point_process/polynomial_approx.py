import jax
import jax.numpy as jnp
from scipy.integrate import simpson

from itertools import combinations

from .basis import raised_cosine_log_eval
from .utils import reshape_for_vmap, slice_array

jax.config.update("jax_enable_x64", True)

def compute_m(spikes_n, x, basis_fn):
    phi = jnp.array(simpson(basis_fn(-x), x=x, axis=0))
    # phi = basis_fn(-x).sum(0)
    return (spikes_n[:, None] * phi).ravel()

def phi_product_int(delta_idx, x, ws, n_basis_funcs, history_window):
    """compute int(phi(tau)phi(tau-delta_x))dtau"""
    # set bounds to the overlap interval
    x1 = x[delta_idx:]
    x2 = x[:ws - delta_idx]

    phi_ts1 = raised_cosine_log_eval(-x1, history_window, n_basis_funcs)
    phi_ts2 = raised_cosine_log_eval(-x2, history_window, n_basis_funcs)

    phi_products = phi_ts1[:, :, None] * phi_ts2[:, None, :]

    return simpson(phi_products, x=x1, axis=0)
    # return phi_products.sum(0)


# this can be parallelized possibly but we only compute it once
# takes <10 seconds for 40-50 delta_x values
def precompute_Phi_x(x, ws, n_basis_funcs, history_window):
    """precompute M_ts_ts' for all possible deltas"""
    M_x = []

    for delta_x in range(ws):
        M_x.append(phi_product_int(delta_x, x, ws, n_basis_funcs, history_window))

    return jnp.stack(M_x)


def compute_M_block(x, Phi_x, tot_spikes, max_window, max_spikes, n_batches_scan, pair):
    def compute_Mn_half(pair):
        def valid_idx(M_scan, idx):
            spk_in_window = slice_array(
                tot_spikes, idx, max_window
            )
            dts = spk_in_window[0] - jax.lax.dynamic_slice(tot_spikes, (0, idx), (1, 1))
            dts_valid = jnp.where(spk_in_window[1] == pair[0], dts[0], invalid_dts)
            dts_idx = jnp.argmin(jnp.abs(x[None, :] - jnp.abs(dts_valid[:, None])), axis=1)
            cross_prod_sum = jnp.sum(Phi_x[dts_idx], axis=0)
            M_scan = M_scan + cross_prod_sum
            return M_scan

        def invalid_idx(M_scan, idx):
            M_scan += jnp.zeros_like(Phi_x[0])
            return M_scan

        def scan_fn(M_scan, idx):
            M_scan = jax.lax.cond(idx > tot_spikes.shape[1], invalid_idx, valid_idx, M_scan, idx)
            return M_scan, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.zeros_like(Phi_x[0]), idxs))

        post_idx = jnp.nonzero(tot_spikes[1] == pair[1], size=max_spikes, fill_value=tot_spikes.shape[1] + 5)[0]
        post_idx_array, padding = reshape_for_vmap(post_idx, n_batches_scan)
        invalid_dts = jnp.tile(x[-1] + 1, max_window)
        out, _ = scan_vmap(post_idx_array)
        sub, _ = scan_vmap(padding[:, None])

        return jnp.sum(out, 0) - jnp.sum(sub, 0)

    pair_double = jnp.stack((pair, pair[::-1]))
    pair_M = jax.vmap(compute_Mn_half)(pair_double)

    return pair_M[0] + pair_M[1].T


@jax.jit
def construct_M(M_self, M_cross):
    N, J = M_self.shape[0], M_self.shape[1]
    M_full = jnp.zeros((N * J, N * J), dtype=(M_self.dtype))
    diag_starts = jnp.arange(N) * J
    rows = jnp.arange((N * (N - 1)) // 2) // (N - 1)
    cols = jnp.arange((N * (N - 1)) // 2) % (N - 1) + 1 + rows
    i_starts, j_starts = rows * J, cols * J

    def insert_block(matrix, idx, block):
        return jax.lax.dynamic_update_slice(matrix, block, idx)

    insert_block_vmap = jax.vmap(lambda idxs, blocks: insert_block(M_full, idxs, blocks), in_axes=(0, 0))
    M_full = insert_block_vmap((diag_starts, diag_starts), M_self).sum(0)
    M_full = insert_block_vmap((i_starts, j_starts), M_cross).sum(0)
    M_full = insert_block_vmap((j_starts, i_starts), jnp.transpose(M_cross, (0, 2, 1))).sum(0)

    return M_full


def compute_M(
        max_window,
        spikes_n,
        n_batches_scan,
        x,
        Phi_x,
        tot_spikes,
        neuron_ids,
):
    self_pairs = jnp.array(list(zip(neuron_ids, neuron_ids)))
    cross_pairs = jnp.array(list(combinations(neuron_ids, 2)))

    self_products = jnp.einsum("i,jk->ijk", spikes_n, Phi_x[0])

    M_block_pair = lambda p: compute_M_block(x, Phi_x, tot_spikes, max_window, spikes_n.max(), n_batches_scan, p)

    M_self = jax.vmap(M_block_pair)(self_pairs)
    M_self += self_products
    M_cross = jax.vmap(M_block_pair)(cross_pairs)

    M_full = construct_M(M_self, M_cross)
    return M_full

def run_scan_fn_k(
        tot_spikes,
        target_idx,
        max_window,
        basis_fn,
        n_batches_scan,
        k_init,
):

    def scan_k(k_sum, i):
        spk_in_window = slice_array(
            tot_spikes, i + 1, max_window + 1
        )
        dts = spk_in_window[0] - jax.lax.dynamic_slice(tot_spikes, (0, i), (1, 1))
        eval = basis_fn(dts[0])

        k_sum = k_sum.at[spk_in_window[1].astype(int)].add(eval)

        return k_sum, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_k, k_init, idxs))
    target_idx_array, padding = reshape_for_vmap(target_idx, n_batches_scan)
    k, _ = scan_vmap(target_idx_array)
    sub, _ = scan_vmap(padding[:, None])

    return jnp.sum(k,0).ravel() - jnp.sum(sub,0).ravel()


def compute_chebyshev(f, xlim, power=2, dx=0.01):
    """jax only implementation"""
    xx = jnp.arange(xlim[0] + dx / 2.0, xlim[1], dx)
    nx = xx.shape[0]
    xxw = jnp.arange(-1.0 + 1.0 / nx, 1.0, 1.0 / (0.5 * nx))
    Bx = jnp.zeros([nx, power + 1])
    for i in range(0, power + 1):
        Bx = Bx.at[:, i].set(jnp.power(xx, i))
    errwts_cheby = 1.0 / jnp.sqrt(1 - xxw**2)
    Dx = jnp.diag(errwts_cheby)
    fx = f(xx)
    what_cheby = jnp.linalg.lstsq(Bx.T @ Dx @ Bx, Bx.T @ Dx @ fx, rcond=None)[0]
    return what_cheby

def compute_paglm_weights(suff,interval,f=jnp.exp,Cinv=None):
    """
    This function computes the paglm weights given a nonlinearity, sufficient
    statistics, bin size, and optional prior over an approximation interval.
    suff = [m,M,k]
    """

    # if Cinv is none, make prior a matrix of zeros
    if Cinv is None:
        Cinv = jnp.zeros_like(suff[2])

    # compute Chebyshev approximation and paGLM weights
    a0,a1,a2 = compute_chebyshev(f,interval,power=2,dx=0.01)
    b = (a1 / 2*a2) * (suff[2]-suff[0])
    w_paglm =  jnp.linalg.lstsq(suff[1]+Cinv,b,rcond=True)[0]
    return w_paglm
