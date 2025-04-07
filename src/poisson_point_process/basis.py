from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import gamma, gammainc, factorial
from scipy.special import laguerre

import numpy as np
from sklearn.decomposition import PCA

from .utils import comb

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def raised_cosine_log_eval(x, ws, n_basis_funcs, width=2.0, time_scaling=50.0):
    """jax only raised cosine log."""
    last_peak = 1 - width / (n_basis_funcs + width - 1)
    peaks = jnp.linspace(0, last_peak, n_basis_funcs)
    delta = peaks[1] - peaks[0]

    x = -x / ws

    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    x = jnp.log(time_scaling * x + 1) / jnp.log(time_scaling + 1)

    basis_funcs = 0.5 * (
        jnp.cos(
            jnp.clip(
                jnp.pi * (x[:, None] - peaks[None]) / (delta * width),
                -jnp.pi,
                jnp.pi,
            )
        )
        + 1
    )
    return basis_funcs

def raised_cosine_log_int(n_basis_funcs, history_window, width=2.0, time_scaling=50.0):
    last_peak = 1 - width / (n_basis_funcs + width - 1)
    peaks = jnp.linspace(0, last_peak, n_basis_funcs)
    delta = peaks[1] - peaks[0]

    x0 = jnp.clip(
        jnp.pi * (0 - peaks) / (delta * width),
        -jnp.pi,
        jnp.pi,
    )
    x1 = jnp.clip(
        jnp.pi * (1 - peaks) / (delta * width),
        -jnp.pi,
        jnp.pi,
    )

    # compute constants
    a = time_scaling + 1
    b = jnp.pi / (delta * width)
    A = jnp.log(a) / b
    c_I = A * jnp.power(a, peaks) / (2 * time_scaling)
    # evaluate the integral of the form \int exp(ax) * (cos(x) + 1) dx
    I = lambda x: jnp.exp(A * x) / A + (jnp.exp(A * x) * (A * jnp.cos(x) + jnp.sin(x))) / (A ** 2 + 1)
    return c_I * (I(x1) - I(x0)) * history_window


def interpolate_to_chebyshev(pts, y, power=None):
    if power is None:
        power = len(pts) - 1
    cheb_nodes = jnp.cos((2 * jnp.arange(power + 1) + 1) * jnp.pi / (2 * (power + 1)))
    interpolated_vals = jax.vmap(lambda fn: jnp.interp(jnp.flip(cheb_nodes), 2*pts-1, fn), in_axes=1, out_axes=1)(y)

    return interpolated_vals, cheb_nodes

def compute_barycentric_weights(y, t=None, n=None):
    ''''
    compute barycentric weights for polynomial interpolation
    assumes chebyshev nodes for interpolation by default (numerically stable and scalable for large N)
    '''
    if n is None:
        n = y.shape[0]-1

    # default to chebyshev nodes
    if t is None:
        w = jnp.power(-1, jnp.arange(n + 1)) * jnp.sin(((2*jnp.arange(n + 1) + 1)*jnp.pi)/(2*n + 2))

    else:
        C = (t[n] - t[0]) / 4
        tc = t / C
        omega = jnp.ones(n + 1)
        for m in range(n):
            d = tc[:m + 1] - tc[m + 1]
            omega = omega.at[:m + 1].multiply(d)
            omega = omega.at[m + 1].set(jnp.prod(-d))
        w = 1.0 / omega

    return w

@partial(jax.jit, static_argnums=(1,))
def polynomial_eval(dts, ws, w, t, y):
    dts = -dts / ws
    x = jnp.clip(dts, 0, 1)

    terms = w / (x[:, None] - t)
    f_values = (y.T[:,None,:] * terms[None,:,:]).sum(2) / jnp.sum(terms, axis=1, keepdims=True).T
    basis_funcs = jnp.where(jnp.isinf(terms).any(axis=1), y[jnp.argmax(x[:, None] == t, axis=1)].T, f_values)

    return jnp.where(dts > 1, 0, basis_funcs).T

def gp_basis(ws, n_fun, gamma=50, seed=0, rh=1, len_sc=0.12, a=50., b=0.001, w=0.065, delay=0.0):
    np.random.seed(seed)
    x = np.linspace(0,1,ws)
    log_spaced_x = np.log(gamma * x + 1) / np.log(gamma + 1)
    x_log = (np.array([log_spaced_x]) - np.transpose(np.array([log_spaced_x])))
    K_log = rh*np.exp(-(np.square(x_log)/(2*np.square(len_sc))))
    alpha = a*((x-delay)**2)*np.exp(-(x-delay) / w) + b
    alpha[x<delay] = b
    synapse_K = K_log*alpha[None,:]
    samp_GP = np.array(np.random.multivariate_normal(np.zeros(ws), synapse_K,10000)).T
    pca = PCA(n_components=n_fun)
    pca.fit(samp_GP.T)
    dx = 1 / (ws - 1)  # Grid spacing
    norm_factor = np.sqrt(dx)
    basis_kernel = pca.components_.T / norm_factor
    var_expl = pca.explained_variance_ * dx
    return basis_kernel, var_expl, x

def tau(t, a, W):
    """
    Time warping function mapping clock time `t` with
    warping parameter `a` and a time window `W` (meaning
    that 0 <= t <= W is the window of time we are
    considering). When `a=0` there is no warping. By eye,
    a choice of `a=1` seems like a reasonable default.
    """
    return W * norm.cdf(a + norm.ppf(t/W))

def phi_eval(x,n_fun,ws,a=1.):
    """
    Evaluate basis function `n` with time warping parameter `a`
    and window size `W` at time `t`.
    Satisfies psi(0)=0 and psi(W)=0.
    """
    n = np.arange(n_fun)+1

    # map to [0,1]
    x = -x /ws
    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    return jnp.sqrt(2 / ws) * jnp.sin(jnp.pi * n * tau(x, a, ws)[:,None] / ws)

def phi_self_eval(x,n_fun,ws,a=1.):
    """
    Evaluate basis function `n` with time warping parameter `a`
    and window size `W` at time `t`.
    Satisfies psi'(0)=0 and psi(W)=0.
    """
    n = np.arange(n_fun)+1

    # map to [0,1]
    x = -x /ws
    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    return jnp.sqrt(2/ws) * jnp.cos(jnp.pi*(2*n+1)*tau(x, a, ws)[:,None] / (2*ws))

def LaplacianWarpedEval(ws, n_basis_funcs, a=1.):
    eval_func = lambda dts: phi_eval(dts, n_basis_funcs, ws, a=a)
    return eval_func

def LaguerreEval(ws, n_basis_funcs, c=1.0, max_x=30.):
    P = np.zeros((n_basis_funcs, n_basis_funcs))
    for n in range(n_basis_funcs):
        P[n, :(n+1)] = laguerre(n).coef[::-1]
    P = jnp.array(P)

    def bf_eval(x, ws, p, c, max_x):
        x = -x / ws
        x = jnp.where(jnp.abs(x) > 1, 1, x)
        x *= max_x
        out = jnp.exp(-c * x/2) * jnp.polyval(p[::-1], c * x)
        return out.T
    basis_funcs = jax.vmap(bf_eval, in_axes=(None, None, 0, None, None), out_axes=1)

    eval_func = lambda dts: basis_funcs(dts, ws, P, c, max_x)
    return eval_func

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def laguerre_int(x, ws, n_basis_funcs, c=1.0, max_x=30):
    """Compute the integral I_n(x) = ∫_0^x L_n(t) * exp(-ct) dt using incomplete gamma function."""
    x = -x / ws
    x = jnp.where(jnp.abs(x) > 1, 1, x)
    x *= max_x

    n = jnp.arange(n_basis_funcs)[:, None]
    k = jnp.arange(n_basis_funcs + 1)[None, :]

    combs = comb(n, k) * (k <= n)
    comb_nk = combs[n, k]
    coeffs = ((-1) ** (k)) / (factorial(k)) * comb_nk
    powers = k + 1
    terms = lambda x: gammainc(powers, c/2 * x) * gamma(powers) / ((c/2) ** powers)

    integrals = coeffs * (terms(x) - terms(0))
    return integrals.sum(1) / max_x

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def laguerre_prod_int(xs, ws, n_basis_funcs, c=1.0, max_x=30.):
    def single_product_integral(x):
        x = -x / ws
        x = jnp.where(jnp.abs(x) > 1, 1, x)
        x *= max_x

        idx = jnp.arange(n_basis_funcs)
        n = idx.reshape(n_basis_funcs, 1, 1, 1, 1)
        k = idx.reshape(1, n_basis_funcs, 1, 1, 1)
        m = idx.reshape(1, 1, n_basis_funcs, 1, 1)
        j = idx.reshape(1, 1, 1, n_basis_funcs, 1)
        r = idx.reshape(1, 1, 1, 1, n_basis_funcs)
        comb_nk = jnp.where(k <= n, comb(n, k), 0.0)
        comb_mj = jnp.where(j <= m, comb(m, j), 0.0)
        comb_kr = jnp.where(r <= k, comb(k, r), 0.0)

        coeffs = (
                jnp.exp(-c * x / 2)
                * ((-1) ** (k + j))
                / (factorial(k) * factorial(j))
                * comb_nk
                * comb_mj
                * comb_kr
                * jnp.where((k - r) >= 0, jnp.power(x, k - r), 0.0)
        )
        powers = r + j + 1
        terms = gammainc(powers, c * (max_x - x)) * gamma(powers) / (c ** powers)
        integrals = coeffs * terms
        return integrals.sum(axis=(1, 3, 4)) / max_x

    return jax.vmap(single_product_integral)(xs)


def ChebyshevInterpolation(eval_pts, basis_kernels, history_window, power=None):
    if power is None:
        power = len(eval_pts) - 1
    interpolated_kernels, nodes = interpolate_to_chebyshev(eval_pts, basis_kernels, power)
    barycentric_weights = jnp.power(-1, jnp.arange(power + 1)) * jnp.sin(((2*jnp.arange(power + 1) + 1)*jnp.pi)/(2*power + 2))
    eval_func = lambda dts: polynomial_eval(dts, history_window, barycentric_weights, -((nodes-1)/2), interpolated_kernels)
    return eval_func

def RaisedCosineLogEval(ws, n_basis_funcs, width=2.0, time_scaling=50.0):
    eval_func = lambda dts: raised_cosine_log_eval(dts, ws, n_basis_funcs, width, time_scaling)
    return eval_func