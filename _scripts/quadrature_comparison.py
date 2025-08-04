import numpy as np
import jax.numpy as jnp
import jax
from jaxopt import GradientDescent
import matplotlib.pyplot as plt
from time import perf_counter
from functools import partial
from poisson_point_process import utils, simulate
from poisson_point_process.poisson_process_glm import ContinuousMC
from poisson_point_process.poisson_process_obs_model import MonteCarloApproximation
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral


def glnodes(N):
    """
    Computes the Legendre-Gauss-Lobatto (LGL) nodes and weights.
    Parameters:
        N (int): The number of intervals (number of nodes is N+1)
    Returns:
        x (np.ndarray): LGL nodes (size N+1)
        w (np.ndarray): Corresponding weights for integration (size N+1)
        P (np.ndarray): Vandermonde matrix of Legendre polynomials evaluated at nodes (shape (N+1, N+1))
    """
    N1 = N + 1
    x = jnp.cos(jnp.pi * jnp.arange(N1) / N)  # Chebyshev-Gauss-Lobatto initial guess
    P = jnp.zeros((N1, N1))
    xold = 2 * jnp.ones_like(x)
    eps = jnp.finfo(float).eps
    # Newton-Raphson method
    while jnp.max(jnp.abs(x - xold)) > eps:
        xold = x.copy()
        # P[:, 0] = 1
        # P[:, 1] = x
        P = P.at[:, 0].set(1)
        P = P.at[:, 1].set(x)
        for k in range(1, N):
            P = P.at[:, k + 1].set(((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1))
        x = xold - (x * P[:, N] - P[:, N - 1]) / (N1 * P[:, N])
    w = 2.0 / (N * N1 * P[:, N] ** 2)
    return x, w, P

def precompute_quad_nodes(y, M, T, M_min=3):
    K = y.shape[1]
    M_res = M - K * M_min
    interval_lengths = (jnp.concatenate([y[0], jnp.array([T])]) -
                        jnp.concatenate([jnp.array([0]), y[0]]))
    weights = interval_lengths / jnp.sum(interval_lengths.clip(1e-12))
    residual_nodes = jnp.floor(weights * M_res).astype(int)
    nodes_per_interval = residual_nodes + M_min
    top_k_indices = jnp.argsort(-(weights * M_res - residual_nodes))[: M - nodes_per_interval.sum()]
    nodes_per_interval = nodes_per_interval.at[top_k_indices].add(1)
    quad_nodes = {}
    for n_nodes in jnp.unique(nodes_per_interval):
        x, w, _ = glnodes(n_nodes)
        ref_mask = jnp.concatenate([jnp.ones(len(x) - 1), jnp.array([0.0])])
        log_mask = jnp.concatenate([jnp.array([1.0]), jnp.zeros(len(x) - 1)])
        quad_nodes[int(n_nodes)] = (x, w, ref_mask, log_mask)
    return tuple(int(x) for x in nodes_per_interval), quad_nodes

def compute_lam_tilde(dts, weights, bias):
    fx = basis_fn(dts)
    return jnp.sum(fx * weights) + bias

def intensity_fun(X, i, params, r_t):
    weights = params[0].reshape(n_neurons, n_basis_funcs)
    bias = params[1]
    spk_in_window = utils.slice_array(X, i[-1].astype(int), max_window)
    dts = spk_in_window[0] - i[0]
    lam_tilde = compute_lam_tilde(dts, weights[spk_in_window[1].astype(int)], bias)
    return jnp.exp(lam_tilde) * r_t

# @partial(jax.jit, static_argnums=(3,4))
def compute_quadrature_nll(
        X,
        y,
        params,
        nodes_per_interval,
        quad_nodes,
):
    tau = 0.00005
    r = lambda t: jnp.where(t > tau, 1.0, 0.0)
    K = y.shape[1]
    cif = 0.0
    log_lam = 0.0
    t_prev = 0.0
    # Loop over spikes
    for k in range(K):
        m = nodes_per_interval[k]#.astype(int)
        x, w, ref_mask, log_mask = quad_nodes[m]
        t = y[:, k]
        r_t = r(t[0] - t_prev)
        a = t_prev + tau
        b = t[0]
        if a >= b:
            print(f"refractory period is too long, interval {t[0]-t_prev}")
            t_prev = t[0]
            continue
        # Map quadrature nodes
        t_nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
        idx_nodes = jnp.searchsorted(X[0], t_nodes)
        nodes = jnp.vstack((t_nodes, idx_nodes))
        # Loop over nodes
        for i, n in enumerate(nodes.T):
            lam = intensity_fun(X, n, params, r_t)
            log_lam += jnp.log(lam) * log_mask[i]  # current spike's contribution
            cif += (0.5 * (b - a) * w[i]) * lam * ref_mask[i]
        t_prev = t[0]
    print(cif)
    print(log_lam)
    return cif - log_lam


def make_compute_nll(quad_nodes):
    @partial(jax.jit, static_argnums=(3,))
    def compute_quadrature_nll(params, X, y, nodes_per_interval):
        tau = 0.00005
        r = lambda t: jnp.where(t > tau, 1.0, 0.0)
        K = y.shape[1]
        cif = 0.0
        log_lam = 0.0
        t_prev = 0.0
        # Loop over spikes
        for k in range(K):
            m = nodes_per_interval[k]
            x, w, ref_mask, log_mask = quad_nodes[m]
            t = y[:, k]
            r_t = r(t[0] - t_prev)
            a = t_prev + tau
            b = t[0]
            # if a >= b:
            #     print(f"refractory period is too long, interval {t[0] - t_prev}")
            #     t_prev = t[0]
            #     continue
            # Map quadrature nodes
            t_nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
            idx_nodes = jnp.searchsorted(X[0], t_nodes)
            nodes = jnp.vstack((t_nodes, idx_nodes))
            # Loop over nodes
            for i, n in enumerate(nodes.T):
                lam = intensity_fun(X, n, params, r_t)
                log_lam += jnp.log(lam) * log_mask[i]  # current spike's contribution
                cif += (0.5 * (b - a) * w[i]) * lam * ref_mask[i]
            t_prev = t[0]
        # print(cif)
        # print(log_lam)
        return jnp.sum(cif - log_lam).squeeze()
    return compute_quadrature_nll

n_neurons = 8
T = 100
history_window = 0.005
binsize = 0.0001
n_bins_tot = int(T/binsize)
window_size = int(history_window/binsize)
n_basis_funcs = 4
c = 1.5
basis_fn = GenLaguerreEval(history_window, n_basis_funcs)
time = jnp.linspace(0,history_window,window_size)
kernels = basis_fn(-time)
int_fn = GenLaguerreInt(history_window, n_basis_funcs)
prod_int_fn = GenLaguerreProdIntegral(history_window, n_basis_funcs)
# phi = jax.nn.softplus
# phi_inverse = lambda x: jnp.log(jnp.exp(x) - 1)
phi = jnp.exp
phi_inverse = jnp.log

M = 50 * T

np.random.seed(216)

pres_rate_hz = 10
posts_rate_hz = 2

# inverse firing rate per bin
bias_true = phi_inverse(posts_rate_hz*binsize)
weights_true = jnp.array(np.random.normal(0.1, 0.3, n_neurons * n_basis_funcs))
filters_true = phi(np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T) + bias_true) / binsize

_, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_hz, bias_true, binsize,
                                                           n_bins_tot, n_neurons, weights_true, window_size, kernels,
                                                           phi)
spike_times, spike_ids = simulate.poisson_times(X_counts, T, binsize)
spike_times_y, _ = simulate.poisson_times(y_counts[:, None], T, binsize)
X_spikes = jnp.vstack((spike_times, spike_ids))
target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)
y_spikes = jnp.vstack((spike_times_y, target_idx))
# y_spikes = jnp.vstack((y_spikes[0], jnp.zeros(y_spikes.shape[1]), y_spikes[1]))

# params_true = (weights_true, jnp.atleast_1d(bias_true))
# params = (jnp.array(np.random.normal(0.1, 0.3, n_neurons * n_basis_funcs)), jnp.atleast_1d(phi_inverse(y_spikes.shape[1]/T)))

# baseline_fr = 2
# biases = jnp.log(jnp.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons))) + jnp.log(binsize)
# weights_true = jnp.array(np.random.normal(0,0.5, size=(n_neurons, n_neurons, n_basis_funcs)))
# params = (weights_true, biases)
# spike_counts, rates = simulate.poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, kernels, params, phi)
# print(spike_counts.sum())
# spike_times, spike_ids = simulate.poisson_times(spike_counts, T, binsize)
# X_spikes = jnp.vstack((spike_times,spike_ids))
# y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))

max_window = int(utils.compute_max_window_size(jnp.array([-history_window, 0]), X_spikes[0], X_spikes[0]))
X_shift, y_shift = utils.adjust_indices_and_spike_times(X_spikes, history_window, max_window, y_spikes)

nodes_per_interval, quad_nodes = precompute_quad_nodes(y_spikes, M, T)

# compute_quadrature_nll(X_shift, y_shift, params, nodes_per_interval, quad_nodes)



# loss_fn = lambda X, y, p: compute_quadrature_nll(X, y, p, nodes_per_interval, quad_nodes)
compute_nll = make_compute_nll(quad_nodes)
loss_fn = lambda p, X, y: compute_nll(p, X, y, nodes_per_interval)

# t0 = perf_counter()
# print(compute_quadrature_nll(X_shift, y_shift, params))
# print(perf_counter()-t0)

solver = GradientDescent(fun=loss_fn, maxiter=10, stepsize=-1)
params = (jnp.zeros_like(weights_true), jnp.atleast_1d(phi_inverse(y_spikes.shape[1]/T)))
state = solver.init_state(params, X_shift, y_shift)
for i in range(1000):
    t0 = perf_counter()
    params, state = solver.update(params, state, X_shift, y_shift)
    print(f"iter {i}, loss={state.error}, time {perf_counter()-t0}")