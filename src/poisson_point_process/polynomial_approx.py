import jax
import jax.numpy as jnp
from scipy.integrate import simpson

from .basis import raised_cosine_log_eval
from .utils import reshape_for_vmap, slice_array


def phi_product_int(delta_idx, x, ws, n_basis_funcs, history_window):
    """compute int(phi(tau)phi(tau-delta_x))dtau"""
    # set bounds to the overlap interval
    x1 = x[delta_idx:ws]
    x2 = x[0 : ws - delta_idx]

    # assert x1.size == x2.size

    phi_ts1 = raised_cosine_log_eval(-x1, history_window, n_basis_funcs)
    phi_ts2 = raised_cosine_log_eval(-x2, history_window, n_basis_funcs)

    phi_products = phi_ts1[:, :, None] * phi_ts2[:, None, :]

    return simpson(phi_products, x=x1, axis=0)


# this can be parallelized possibly but we only compute it once
# takes <10 seconds for 40-50 delta_x values
def precompute_Phi_x(x, ws, n_basis_funcs, history_window):
    """precompute M_ts_ts' for all possible deltas"""
    M_x = []

    for delta_x in range(ws):
        M_x.append(phi_product_int(delta_x, x, ws, n_basis_funcs, history_window))

    return jnp.stack(M_x)


def compute_M_blocks(
        max_window,
        n_basis_funcs,
        n_batches_scan,
        x,
        Phi_x,
        tot_spikes,
        pairs,
        ):
    def compute_Mn(x, Phi_x, tot_spikes, pair):
        def valid_idx(M_scan, idx):
            spk_in_window = slice_array(
                tot_spikes, idx + 1, max_window + 1
            )
            dts = spk_in_window[0] - jax.lax.dynamic_slice(tot_spikes, (0, idx), (1, 1))
            dts_valid = jnp.where(spk_in_window[1] == pair[0], dts, invalid_dts)
            dts_idx = jnp.argmin(jnp.abs(x[None, :] - jnp.abs(dts_valid.T)), axis=1)
            cross_prod_sum = jnp.sum(Phi_x[dts_idx], axis=0)
            M_scan = M_scan + Phi_x[0] + 2 * cross_prod_sum
            return M_scan
        def invalid_idx(M_scan, idx):
            M_scan += jnp.zeros((n_basis_funcs, n_basis_funcs))
            return M_scan
        def scan_fn(M_scan, idx):
            M_scan = jax.lax.cond(idx > len(tot_spikes[0]), invalid_idx, valid_idx, M_scan, idx)
            return M_scan, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.zeros((n_basis_funcs, n_basis_funcs)), idxs))

        post_idx = jnp.nonzero(tot_spikes[1] == pair[1], size=len(tot_spikes[0]), fill_value=len(tot_spikes[0]) + 5)[0]
        post_idx_array, padding = reshape_for_vmap(post_idx, n_batches_scan)
        invalid_dts = jnp.tile(x[-1]+1, max_window+1)
        out, _ = scan_vmap(post_idx_array)
        sub, _ = scan_vmap(padding[:, None])

        return jnp.sum(out,0) - jnp.sum(sub,0)

    compute_Mn_vmap = jax.vmap(lambda pair: compute_Mn(x, Phi_x, tot_spikes, pair))

    return compute_Mn_vmap(pairs)

@jax.jit
def construct_M(M_self, M_cross):
    def insert_diag(n, matrix):
        i_start = n * J
        block = jax.lax.dynamic_slice_in_dim(M_self, n, 1, 0)[0]
        return jax.lax.dynamic_update_slice(matrix, block, (i_start, i_start))
    def insert_off_diag(idx, matrix):
        row = jax.lax.div(idx, N - 1)
        col = jax.lax.rem(idx, N - 1) + 1 + row
        i_start, j_start = row * J, col * J
        block = jax.lax.dynamic_slice(M_cross, (idx, 0, 0), (1, J, J)).squeeze(0)
        matrix = jax.lax.dynamic_update_slice(matrix, block, (i_start, j_start))
        matrix = jax.lax.dynamic_update_slice(matrix, block, (j_start, i_start))
        return matrix
    N, J = M_self.shape[0], M_self.shape[1]
    M_full = jnp.zeros((N*J, N*J))
    upper = (N * (N - 1)) // 2
    M_full = jax.lax.fori_loop(0, N, insert_diag, M_full)
    M_full = jax.lax.fori_loop(0, upper, insert_off_diag, M_full)
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
        print(dts[0].shape)
        print(eval.shape)

        k_sum = k_sum.at[spk_in_window[1].astype(int)].add(eval)

        return k_sum, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_k, k_init, idxs))
    target_idx_array, padding = reshape_for_vmap(target_idx, n_batches_scan)
    k, _ = scan_vmap(target_idx_array)
    sub, _ = scan_vmap(padding[:, None])

    return jnp.sum(k,0).flatten() - jnp.sum(sub,0).flatten()


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
