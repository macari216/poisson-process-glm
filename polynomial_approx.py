import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from time import perf_counter

import nemos as nmo
import pynapple as nap

from scipy.integrate import simpson

import matplotlib.pyplot as plt

### functions
# compute max window
@jax.jit
def tot_spk_in_window(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single neuron"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)

@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    return jax.lax.dynamic_slice(array, (i - window_size,), (window_size,))

def raised_cosine_log_eval(x, ws, n_basis_funcs, width=2., time_scaling=50.):
    """jax only raised cosine log."""
    last_peak = 1 - width / (n_basis_funcs + width - 1)
    peaks = jnp.linspace(0, last_peak, n_basis_funcs)
    delta = peaks[1] - peaks[0]

    x = x / ws

    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    x = jnp.log(time_scaling * x + 1) / jnp.log(
        time_scaling + 1
    )


    basis_funcs = 0.5 * (
            jnp.cos(
                jnp.clip(
                    np.pi * (x[:, None] - peaks[None]) / (delta * width),
                    -np.pi,
                    np.pi,
                )
            )
            + 1
    )
    return basis_funcs

def phi_product_int(delta_idx, x, ws, n_basis_funcs):
    """compute int(phi(tau)phi(tau-delta_x))dtau"""
    # set bounds to the overlap interval
    x1 = x[delta_idx : ws]
    x2 = x[0 : ws-delta_idx]

    assert x1.size == x2.size

    phi_ts1 = raised_cosine_log_eval(x1, history_window, n_basis_funcs)
    phi_ts2 = raised_cosine_log_eval(x2, history_window, n_basis_funcs)

    phi_products = phi_ts1[:, :, None] * phi_ts2[:, None, :]

    return simpson(phi_products, x=x1, axis=0)

# this can be parallelized possibly but we only compute it once
# takes <10 seconds for 40-50 delta_x values
def precompute_Phi_x(x, ws, n_basis_funcs):
    """precompute M_ts_ts' for all possible deltas"""
    M_x = []

    for delta_x in range(ws):
        M_x.append(phi_product_int(delta_x, x, ws, n_basis_funcs))

    return jnp.stack(M_x)

def scan_fn_M(M_sum, i):
    dts = slice_array(pres_spikes_new, i, max_window) - jax.lax.dynamic_slice(pres_spikes_new, (i,), (1,))
    dts_idx = jnp.argmin(jnp.abs(x[None,:] - jnp.abs(dts[:,None])), axis=1)
    cross_prod_sum = jnp.sum(Phi_x[dts_idx], axis=0)

    M_sum = M_sum + Phi_x[0] + 2*cross_prod_sum

    return M_sum, None

def scan_fn_k(lam_s, i):
    pre, post = i
    dts = slice_array(pres_spikes_new, pre, max_window) - jax.lax.dynamic_slice(posts_spikes, (post,), (1,))

    ll = raised_cosine_log_eval(jnp.abs(dts), history_window, n_basis_funcs)

    lam_s += jnp.sum(ll)
    return jnp.sum(lam_s), None

def compute_chebyshev(f,xlim,power=2,dx=0.01):
    """jax only implementation"""
    xx = jnp.arange(xlim[0]+dx/2.0,xlim[1],dx)
    nx = xx.shape[0]
    xxw = jnp.arange(-1.0+1.0/nx,1.0,1.0/(0.5*nx))
    Bx = jnp.zeros([nx,power+1])
    for i in range(0,power+1):
        Bx = Bx.at[:,i].set(jnp.power(xx,i))
    errwts_cheby = 1.0 / jnp.sqrt(1-xxw**2)
    Dx = jnp.diag(errwts_cheby)
    fx = f(xx)
    what_cheby = jnp.linalg.lstsq(Bx.T @ Dx @ Bx,Bx.T @ Dx @ fx, rcond = None)[0]
    fhat_cheby = Bx @ what_cheby
    return what_cheby

### main

# generate data
n_neurons = 2
spk_hz = 200
tot_time_sec = 1000
tot_spikes_n = int(tot_time_sec * spk_hz * n_neurons)
history_window = 0.004
binsize = 0.0001
window_size = int(history_window/binsize)
n_basis_funcs = 5
print(f"total spikes: {tot_spikes_n}")

np.random.seed(123)

# full dataset
tot_spikes = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
neuron_ids = np.random.choice(n_neurons, size=len(tot_spikes))

# postsynaptic neuron
target_neu_id = 1
posts_spk_idx = jnp.arange(len(tot_spikes))[neuron_ids == target_neu_id]
posts_spikes = tot_spikes[posts_spk_idx]
n_posts_spikes = posts_spikes.size

#presynaptic neuron
pres_spk_idx = jnp.arange(len(tot_spikes))[neuron_ids != target_neu_id]
pres_spikes = tot_spikes[pres_spk_idx]
n_pres_spikes = pres_spikes.size
print(f"presynaptic spikes: {n_pres_spikes}")

# compute sufficient statistics
# basis functions binning
# (this will be useful with nonparamatric basis functions e.g. GP basis)
x = jnp.linspace(0, history_window, window_size)

# m linear cumulative contributions
# m = n_pres_spikes * raised_cosine_log_eval(x, history_window, n_basis_funcs).sum(0)
m = n_pres_spikes * simpson(raised_cosine_log_eval(x, history_window, n_basis_funcs), x=x, axis=0)

# M interaction matrix
max_window = tot_spk_in_window(jnp.array([-history_window,0]), tot_spikes, tot_spikes)
max_window = int(max_window)

delta_idx = jax.nn.relu(max_window - pres_spk_idx[0])
# pres_idx_new = pres_spk_idx + delta_idx
pres_spikes_new = jnp.hstack((jnp.full(delta_idx, history_window+1), pres_spikes))
posts_idx_new = posts_spk_idx + delta_idx
tot_spikes_new = np.hstack((jnp.full(delta_idx, history_window+1), tot_spikes))
neuron_ids_new = np.hstack((jnp.full(delta_idx, -1), neuron_ids))

Phi_x = precompute_Phi_x(x, window_size, n_basis_funcs)

M, _ = jax.lax.scan(scan_fn_M, jnp.zeros((5,5)), jnp.arange(delta_idx,n_pres_spikes+delta_idx))

# ## test for scan (it works)
# M_loop = np.zeros((n_basis_funcs, n_basis_funcs))
# for idx in range(delta_idx, n_pres_spikes+delta_idx):
#     dts = pres_spikes_new[idx-max_window: idx] - pres_spikes_new[idx]
#     dts_idx = jnp.argmin(jnp.abs(x[None,:] - jnp.abs(dts[:,None])), axis=1)
#     cross_prod_sum = jnp.sum(Phi_x[dts_idx],axis=0)
#     M_loop = M_loop + Phi_x[0] + 2*cross_prod_sum
#
# print(jnp.allclose(M_loop,M))

# k event cumulative contributions
k_scan_idx =jnp.vstack((jnp.searchsorted(pres_spikes, posts_spikes, 'right') + delta_idx,
                        jnp.arange(n_posts_spikes))).T
k, _ =  jax.lax.scan(scan_fn_k, jnp.array(0), k_scan_idx)

# compute coefficients
bounds = [-2,2]
f = jnp.exp
nonlin_center = [np.log(n_posts_spikes/tot_time_sec)+bounds[0], np.log(n_posts_spikes/tot_time_sec)+bounds[1]]
coefs = compute_chebyshev(f, nonlin_center, power=2, dx=binsize)

# model params
# here we only fit the weights from presynaptic neuron (0)
# to postsynaptic neuron (1) ignoring the self (1 to 1) weights
# w = jnp.zeros(n_basis_funcs)
w = jnp.array(0.01 * np.random.randn(n_basis_funcs))

# integral
linear_term = jnp.dot(m,w)
quadratic_term = np.dot(w, np.dot(M, w))
approx_integral = coefs[0] + coefs[1]*linear_term + coefs[2]*quadratic_term

### test: compare to the exact integral of a quadratic nonlinearity
spikes_dict = {0: pres_spikes, 1: posts_spikes}
spikes_tsgroup = nap.TsGroup(spikes_dict, nap.IntervalSet(0, tot_time_sec))
y_count = spikes_tsgroup.count(binsize)
rc_basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs, 'eval')
X = rc_basis.compute_features(y_count[:,0])
X = jnp.array(X)
dot_prod = jnp.dot(X,w)[40:]
exact_integral = tot_time_sec*coefs[0] + coefs[1]*simpson(dot_prod,x=y_count.t[40:]) + coefs[2]*simpson(jnp.square(dot_prod),x=y_count.t[40:])

# print(approx_integral)
# print(exact_integral)
print(np.allclose(approx_integral, exact_integral))