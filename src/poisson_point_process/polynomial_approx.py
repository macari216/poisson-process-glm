import jax
import jax.numpy as jnp
from scipy.integrate import simpson

from .basis import raised_cosine_log_eval
from .utils import slice_array


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


def run_scan_fn_M(x, Phi_x, pres_spikes_new, max_window, delta_idx, n_pres_spikes):

    def scan_fn_M(M_sum, i):
        dts = slice_array(pres_spikes_new, i, max_window) - jax.lax.dynamic_slice(
            pres_spikes_new, (i,), (1,)
        )
        dts_idx = jnp.argmin(jnp.abs(x[None, :] - jnp.abs(dts[:, None])), axis=1)
        cross_prod_sum = jnp.sum(Phi_x[dts_idx], axis=0)

        M_sum = M_sum + Phi_x[0] + 2 * cross_prod_sum

        return M_sum, None

    M, _ = jax.lax.scan(
        scan_fn_M, jnp.zeros((5, 5)), jnp.arange(delta_idx, n_pres_spikes + delta_idx)
    )
    return M


def run_scan_fn_k(
    pres_spikes,
    n_posts_spikes,
    delta_idx,
    posts_spikes,
    pres_spikes_new,
    max_window,
    history_window,
    n_basis_funcs,
):

    def scan_fn_k(lam_s, i):
        pre, post = i
        dts = slice_array(pres_spikes_new, pre, max_window) - jax.lax.dynamic_slice(
            posts_spikes, (post,), (1,)
        )

        ll = raised_cosine_log_eval(jnp.abs(dts), history_window, n_basis_funcs)

        lam_s += jnp.sum(ll)
        return jnp.sum(lam_s), None

    k_scan_idx = jnp.vstack(
        (
            jnp.searchsorted(pres_spikes, posts_spikes, "right") + delta_idx,
            jnp.arange(n_posts_spikes),
        )
    ).T
    k, _ = jax.lax.scan(scan_fn_k, jnp.array(0), k_scan_idx)
    return k


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
