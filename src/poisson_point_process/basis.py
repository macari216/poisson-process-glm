from functools import partial

import jax
import jax.numpy as jnp

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


def interpolate_to_chebyshev(pts, y, power=None):
    if power is None:
        power = len(pts) - 1
    cheb_nodes = jnp.flip(jnp.cos((2 * jnp.arange(power + 1) + 1) * jnp.pi / (2 * (power + 1))))
    interpolated_vals = jax.vmap(lambda fn: jnp.interp(cheb_nodes, 2*pts-1, fn), in_axes=1, out_axes=1)(y)

    return interpolated_vals

def polynomial_interpolation(y, t=None, n=None):
    ''''
    compute basis functions interpolation using barycentric formula
    and return a function for evaluating the polynomial at any x
    assumes chebyshev nodes for interpolation by default (numerically stable and scalable for large N)
    '''
    if n is None:
        n = y.shape[0]-1
    # default to chebyshev nodes
    if t is None:
        t = jnp.cos((2 * jnp.arange(n + 1) + 1) * jnp.pi / (2 * (n + 1)))
        # map to [0,1]
        t = -((t-1)/2)
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

    @partial(jax.jit, static_argnums=(1,))
    def polynomial_eval(dts, ws, w):
        dts = -dts / ws
        x = jnp.clip(dts, 0, 1)

        terms = w / (x[:, None] - t)
        f_values = (y.T[:,None,:] * terms[None,:,:]).sum(2) / jnp.sum(terms, axis=1, keepdims=True).T
        basis_funcs = jnp.where(jnp.isinf(terms).any(axis=1), y[jnp.argmax(x[:, None] == t, axis=1)].T, f_values)

        return jnp.where(dts > 1, 0, basis_funcs).T
    basis_fn = lambda x, h: polynomial_eval(x, h, w)

    return basis_fn

def ChebyshevInterpolation(eval_pts, basis_kernels, power=None):
    interpolated_kernels = interpolate_to_chebyshev(eval_pts, basis_kernels, power)
    eval_func = polynomial_interpolation(interpolated_kernels, n=power)
    return eval_func
