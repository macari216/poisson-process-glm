import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pynapple as nap
from scipy.integrate import simpson

from poisson_point_process import raised_cosine_log_eval
from poisson_point_process.polynomial_approx import (
    compute_chebyshev,
    precompute_Phi_x,
    run_scan_fn_k,
    run_scan_fn_M,
)
from poisson_point_process.utils import tot_spk_in_window

### mainf

# generate data
n_neurons = 2
spk_hz = 200
tot_time_sec = 1000
tot_spikes_n = int(tot_time_sec * spk_hz * n_neurons)
history_window = 0.004
binsize = 0.0001
window_size = int(history_window / binsize)
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

# presynaptic neuron
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
m = n_pres_spikes * simpson(
    raised_cosine_log_eval(x, history_window, n_basis_funcs), x=x, axis=0
)

# M interaction matrix
max_window = tot_spk_in_window(jnp.array([-history_window, 0]), tot_spikes, tot_spikes)
max_window = int(max_window)

delta_idx = jax.nn.relu(max_window - pres_spk_idx[0])
# pres_idx_new = pres_spk_idx + delta_idx
pres_spikes_new = jnp.hstack((jnp.full(delta_idx, history_window + 1), pres_spikes))
posts_idx_new = posts_spk_idx + delta_idx
tot_spikes_new = np.hstack((jnp.full(delta_idx, history_window + 1), tot_spikes))
neuron_ids_new = np.hstack((jnp.full(delta_idx, -1), neuron_ids))

Phi_x = precompute_Phi_x(x, window_size, n_basis_funcs, history_window)

M = run_scan_fn_M(x, Phi_x, pres_spikes_new, max_window, delta_idx, n_pres_spikes)

# ## test for scan (it works)
# M_loop = np.zeros((n_basis_funcs, n_basis_funcs))
# for idx in range(delta_idx, n_pres_spikes+delta_idx):
#     dts = pres_spikes_new[idx-max_window: idx] - pres_spikes_new[idx]
#     dts_idx = jnp.argmin(jnp.abs(x[None,:] - jnp.abs(dts[:,None])), axis=1)
#     cross_prod_sum = jnp.sum(Phi_x[dts_idx],axis=0)
#     M_loop = M_loop + Phi_x[0] + 2*cross_prod_sum
#
# print(jnp.allclose(M_loop,M))

# # k event cumulative contributions
# k_scan_idx = jnp.vstack(
#     (
#         jnp.searchsorted(pres_spikes, posts_spikes, "right") + delta_idx,
#         jnp.arange(n_posts_spikes),
#     )
# ).T
k = run_scan_fn_k(
    pres_spikes,
    n_posts_spikes,
    delta_idx,
    posts_spikes,
    pres_spikes_new,
    max_window,
    history_window,
    n_basis_funcs,
)

# compute coefficients
bounds = [-2, 2]
f = jnp.exp
nonlin_center = [
    np.log(n_posts_spikes / tot_time_sec) + bounds[0],
    np.log(n_posts_spikes / tot_time_sec) + bounds[1],
]
coefs = compute_chebyshev(f, nonlin_center, power=2, dx=binsize)

# model params
# here we only fit the weights from presynaptic neuron (0)
# to postsynaptic neuron (1) ignoring the self (1 to 1) weights
# w = jnp.zeros(n_basis_funcs)
w = jnp.array(0.01 * np.random.randn(n_basis_funcs))

# integral
linear_term = jnp.dot(m, w)
quadratic_term = np.dot(w, np.dot(M, w))
approx_integral = coefs[0] + coefs[1] * linear_term + coefs[2] * quadratic_term

### test: compare to the exact integral of a quadratic nonlinearity
spikes_dict = {0: pres_spikes, 1: posts_spikes}
spikes_tsgroup = nap.TsGroup(spikes_dict, nap.IntervalSet(0, tot_time_sec))
y_count = spikes_tsgroup.count(binsize)
rc_basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs, "eval")
X = rc_basis.compute_features(y_count[:, 0])
X = jnp.array(X)
dot_prod = jnp.dot(X, w)[40:]
exact_integral = (
    tot_time_sec * coefs[0]
    + coefs[1] * simpson(dot_prod, x=y_count.t[40:])
    + coefs[2] * simpson(jnp.square(dot_prod), x=y_count.t[40:])
)

# print(approx_integral)
# print(exact_integral)
print(np.allclose(approx_integral, exact_integral))
