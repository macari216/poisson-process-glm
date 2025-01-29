import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from poisson_point_process import simulate

import nemos as nmo
import pynapple as nap

import paglm.core as paglm

"""Code from: https://github.com/davidzoltowski/paglm/tree/master"""

def generate_spike_times(tot_time_sec, tot_spikes_n, n_neurons):
    tot_spikes = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
    neuron_ids = np.random.choice(n_neurons, size=len(tot_spikes))
    spike_dict = {key: nap.Ts(tot_spikes[np.arange(len(tot_spikes))[neuron_ids == key]]) for key in range(n_neurons)}
    spikes_tsgroup = nap.TsGroup(spike_dict, nap.IntervalSet(0, tot_time_sec))

    return spikes_tsgroup

def times_to_counts(spikes, basis, posts_n):
    y = np.array(spikes.count(binsize)).squeeze()
    X = basis.compute_features(y)
    y, X = y[window_size:], X[window_size:]
    y = y[:, posts_n]

    return X, y

def generate_poisson_counts(mean_per_sec, binsize, n_bins_tot, n_pres, n_basis_funcs, ws, basis):
    lam_pres = np.abs(np.random.normal(mean_per_sec, mean_per_sec/10, n_pres))

    rate_per_bin = lam_pres * binsize
    weights_true = np.random.normal(0, 1, n_pres * n_basis_funcs)
    pres_spikes = np.random.poisson(lam=rate_per_bin, size=(n_bins_tot, n_pres))
    X = basis.compute_features(pres_spikes)
    X = X[ws:]
    lam_posts = f(np.dot(X, weights_true))
    y = np.random.poisson(lam=lam_posts, size=len(lam_posts))

    return weights_true.reshape(n_neurons,n_basis_funcs), X, y

def batched_sufficient_stats(X, y, n_batches):
    T = y.shape[0]
    batch_size = int(np.ceil(T/n_batches))
    append_X = np.zeros((-T % batch_size, X.shape[1]))
    append_y = np.zeros(-T % batch_size)
    X, y = np.vstack((X, append_X)), np.hstack((y, append_y))

    sum_x = np.sum(X, axis=0)

    sum_yx = sum(
        np.sum((y[i: i + batch_size])[:, np.newaxis] * X[i: i + batch_size], axis=0)
        for i in range(0, T, batch_size)
    )
    sum_xxT = sum(
        X[i: i + batch_size].T @ X[i: i + batch_size]
        for i in range(0, T, batch_size)
    )
    sym_yxxT = sum(
        X[i: i + batch_size].T @ ((y[i: i + batch_size])[:, np.newaxis] * X[i: i + batch_size])
        for i in range(0, T, batch_size)
    )

    return [sum_x, sum_yx, sum_xxT, sym_yxxT]


def batched_suff_stats(spikes, batch_size, T, binsize, window_size, n_neurons, n_basis_funcs, posts_n):
    n_batches = int(np.ceil(T / batch_size))

    D = n_neurons * n_basis_funcs + 1
    sum_x = np.zeros(D)
    sum_yx = np.zeros(D)
    sum_xxT = np.zeros((D, D))
    sym_yxxT = np.zeros((D, D))

    start = 0.
    for i in range(n_batches):
        end = start + batch_size
        ep = nap.IntervalSet(start, end + window_size)
        yb = np.array(spikes.count(ep=ep, bin_size=binsize))
        Xb = rc_basis.compute_features(yb)

        yb, Xb = yb[window_size:], Xb[window_size:]
        yb = yb[:, posts_n]
        Xb = np.concatenate((np.ones([Xb.shape[0], 1]), Xb), axis=1)

        sum_x += np.sum(Xb, axis=0)
        sum_yx += np.sum(yb[:, np.newaxis] * Xb, axis=0)
        sum_xxT += Xb.T @ Xb
        sym_yxxT += Xb.T @ (yb[:, np.newaxis] * Xb)

        start = end
    return [sum_x, sum_yx, sum_xxT, sym_yxxT], Xb, yb

def sufficient_stats(X, y):
    sum_x = np.sum(X, axis=0)
    sum_yx = np.sum(y[:, np.newaxis] * X, axis=0)
    sum_xxT = X.T @ X
    sym_yxxT = X.T @ (y[:, np.newaxis] * X)

    return [sum_x, sum_yx, sum_xxT, sym_yxxT]

def population_suff_stats(X,y):
    sum_x = np.sum(X, axis=0)
    sum_yx = np.dot(X.T,y)
    sum_xxT = X.T @ X
    sym_yxxT = np.einsum('ij,ik,il->jlk', X, y, X)

    return [sum_x, sum_yx, sum_xxT, sym_yxxT]

# generate data
n_neurons = 10
posts_n = 0
spk_per_sec = 10
history_window = 0.3
tot_time_sec = 100+history_window
tot_spikes_n = int(tot_time_sec * spk_per_sec * n_neurons)
binsize = 0.01
window_size = int(history_window / binsize)
n_basis_funcs = 10
n_bins_tot = int(tot_time_sec / binsize)
n_batches = 10
test_size = int(n_bins_tot/10)

softplus = lambda x: np.log(1 + np.exp(x))

np.random.seed(123)

rc_basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs, "conv", window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window

# set inverse link
# f = np.exp
# inv_link = jnp.exp
f = softplus
inv_link = jax.nn.softplus

# all-to-one connectivity (weights only)
weights_true, X, y =  generate_poisson_counts(spk_per_sec, binsize, n_bins_tot, n_neurons, n_basis_funcs, window_size, rc_basis)

# uniform (bias only)
# spike_times = generate_spike_times(tot_time_sec, tot_spikes_n, n_neurons)
# X, y = times_to_counts(spike_times, rc_basis)

# X_bias = np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
# Xs = X_bias[:test_size]
Xs = X[:test_size]
ys = y[:test_size]
suff = batched_sufficient_stats(X, y, n_batches)



# model params
bounds = [[-5,20], [-5,25],[-1,15],[-1,25]]
mean_fr = np.log(y.sum() / y.shape[0])
nonlin_center = [
    np.array([
        mean_fr + b[0],
        mean_fr + b[1]
    ]) for b in bounds
]

# # prior
# lambda_ridge = np.power(2.0,4)
# Cinv = lambda_ridge*np.eye(suff[0].shape[0])
# Cinv[0,0] = 0.0 # no prior on bias

weights, interval = paglm.fit_paglm(f, suff, binsize, nonlin_center, Xs, ys)
print(f"optimal interval: {interval[0]-mean_fr, interval[1]-mean_fr}")

# in nemos
obs_model = nmo.observation_models.PoissonObservations(inv_link)
model = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model).fit(X,y)

weights_nemos, bias_nemos = model.coef_.reshape(n_neurons,n_basis_funcs), model.intercept_
# weights_paglm, bias_paglm = weights[1:].reshape(n_neurons,n_basis_funcs), weights[0]
weights_paglm, bias_paglm = weights.reshape(n_neurons,n_basis_funcs), 0

print(f"bias (nemos): {bias_nemos[0]}, bias (paglm): {bias_paglm}")
print(f"weights sum (nemos): {weights_nemos.sum()}, weights sum (paglm): {weights_paglm.sum()}")
print(f"true weights sum: {weights_true.sum()}")

filters_nemos = np.dot(weights_nemos, kernels.T)
filters_paglm = np.dot(weights_paglm, kernels.T)
filters_true = np.dot(weights_true, kernels.T)

sc_f = np.max(np.abs(filters_paglm)) / np.max(np.abs(filters_nemos))

# n = 3
# plt.figure()
# plt.plot( np.exp(sc_f*filters_nemos[n]), c='k', label='exact')
# plt.plot( np.exp(filters_paglm[n]), c='r', label='paGLM')
# plt.legend()

fig, axs = plt.subplots(2,5,figsize=(15,5))
for n, ax in enumerate(axs.flat):
    ax.plot(softplus(filters_nemos[n]), c='k', label='exact')
    ax.plot(softplus(filters_paglm[n]), c='r', label='paGLM')
    ax.plot(softplus(filters_true[n]), c='g', label='true')
plt.legend()
plt.tight_layout()

plt.figure()
coefs = paglm.compute_chebyshev(f, interval)
x = np.arange(interval[0], interval[1], 0.01)
plt.plot(f(x), c='k', label="exact nonlinearity")
plt.plot(coefs[0]+coefs[1]*x + coefs[2]*(x**2), c='r', label="approximation")
plt.legend()
plt.show()
