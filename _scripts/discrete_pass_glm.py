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

"""Code from: https://github.com/davidzoltowski/paglm/tree/master"""
import paglm.core as paglm

all_to_one = True

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

def generate_poisson_counts(mean_per_sec, weights, bias, binsize, n_bins_tot, n_pres, n_basis_funcs, ws, basis):
    lam_pres = np.abs(np.random.normal(mean_per_sec, mean_per_sec/10, n_pres))

    rate_per_bin = lam_pres * binsize
    pres_spikes = np.random.poisson(lam=rate_per_bin, size=(n_bins_tot, n_pres))
    X = basis.compute_features(pres_spikes)
    X = X[ws:]
    lam_posts = f(np.dot(X, weights) + bias)
    # plt.hist( np.log(np.exp(lam_posts) - 1), bins=50)
    # plt.hist(lam_posts, bins=50)
    y = np.random.poisson(lam=lam_posts, size=len(lam_posts))

    return X, y, lam_posts

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

def quadratic(x):
    coefs = paglm.compute_chebyshev(f, interval)
    return coefs[0]+coefs[1]*x + coefs[2]*(x**2)

def compute_quadratic_ll(w, suff, interval, dt, f):
    a0, a1, a2 = paglm.compute_chebyshev(f, interval, power=2, dx=0.01)
    b0, b1, b2 = paglm.compute_chebyshev(lambda x: np.log(f(x)), interval, power=2, dx=0.01)
    Xtyb = suff[1] * b1 - suff[0] * a1 * dt
    ll = np.dot(w.T, Xtyb) - np.dot(w.T, np.dot(a2 * dt * suff[2] - b2 * suff[3], w))

    return ll

# generate data
n_neurons = 4
posts_n = 1
history_window = 0.3
tot_time_sec = 1000
binsize = 0.01
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches = 1
test_size = int(n_bins_tot/20)

softplus = lambda x: np.log(1 + np.exp(x))

np.random.seed(216)

rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window

# set inverse link
f = np.exp
inv_link = jnp.exp
# f = softplus
# inv_link = jax.nn.softplus

if all_to_one:
    # all-to-one connectivity
    pres_rate_per_sec = 8
    posts_rate_per_sec = 2
    # rescaled proportionally to the binsize
    bias_true =  posts_rate_per_sec + np.log(binsize)
    weights_true = np.random.normal(0, 0.1, n_neurons * n_basis_funcs)
    X, y, pres_spikes, lam_posts =  simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                               n_bins_tot, n_neurons, weights_true, window_size, rc_basis, inv_link)
    inv_rates = np.log(np.exp(lam_posts) - 1)

    if posts_rate_per_sec == 0:
        X_bias = jnp.concatenate((jnp.zeros([X.shape[0],1]),X),axis=1)
    else:
        X_bias = jnp.concatenate((jnp.ones([X.shape[0], 1]), X), axis=1)

    Xs = X_bias[:test_size]
    ys = y[:test_size]

else:
    baseline_fr = 4
    biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr/10, n_neurons))) + np.log(binsize)
    bias_true = biases[posts_n]

    weights_true = jnp.array(np.random.normal(-0.1, 0.1, size=(n_neurons, n_neurons, n_basis_funcs)))
    params = (weights_true, biases)

    spike_counts, firing_rates = simulate.poisson_counts_recurrent(n_bins_tot+window_size,n_neurons, window_size, kernels, params, inv_link)
    inv_rates = np.log(np.exp(firing_rates[:,posts_n]) - 1)
    print(inv_rates.shape)

    print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
    print(f"max spikes per bin: {spike_counts.max()}")
    print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(spike_counts.sum(0))}")
    plt.hist(inv_rates, bins=50)
    plt.show()

    X = rc_basis.compute_features(spike_counts)
    y = spike_counts[:, posts_n]

    X, y = X[window_size:], y[window_size:]
    # Xs = X[:test_size]
    X_bias = np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
    Xs = X_bias[:test_size]

    ys = y[:test_size]


suff = batched_sufficient_stats(X_bias, y, n_batches)


# model params
# bounds = [[-5, 5], [-2,7], [-2,2], [-0,7]]
# # bounds = [[-1,1]]
# mean_fr = np.log(y.sum() / y.shape[0])
# nonlin_center = [
#     np.array([
#         mean_fr + b[0],
#         mean_fr + b[1]
#     ]) for b in bounds
# ]
nonlin_center = [[np.percentile(inv_rates, 2.5), np.percentile(inv_rates, 97.5)]]
# # prior
# lambda_ridge = np.power(2.0,4)
# Cinv = lambda_ridge*np.eye(suff[0].shape[0])
# Cinv[0,0] = 0.0 # no prior on bias

print("fitting paGLM...")
weights, interval = paglm.fit_paglm(f, suff, nonlin_center, Xs, ys)
# print(f"optimal interval: {interval[0]-mean_fr, interval[1]-mean_fr}")
print(f"optimal interval: {interval[0], interval[1]}")

# in nemos
obs_model = nmo.observation_models.PoissonObservations(inv_link)
model = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model).fit(X,y)

weights_nemos, bias_nemos = model.coef_.reshape(n_neurons,n_basis_funcs), model.intercept_
weights_paglm, bias_paglm = weights[1:].reshape(n_neurons,n_basis_funcs), weights[0]
# weights_paglm, bias_paglm = weights.reshape(n_neurons,n_basis_funcs), -4

# print(f"paglm ll with inferred weights: {paglm.poisson_log_like(weights_paglm.flatten(),ys,Xs,f=f,Cinv=None)}")
# print(f"paglm ll with true weights: {paglm.poisson_log_like(weights_true.flatten(),ys,Xs,f=f,Cinv=None)}")

print(f"bias (nemos): {bias_nemos[0]}, bias (paglm): {bias_paglm}")
print(f"true bias: {bias_true}")
print(f"weights sum (nemos): {weights_nemos.sum()}, weights sum (paglm): {weights_paglm.sum()}")
print(f"true weights sum: {weights_true.sum()}")

filters_nemos = np.dot(weights_nemos, kernels.T) + bias_nemos
filters_paglm = np.dot(weights_paglm, kernels.T) + bias_paglm
if len(weights_true.shape) == 3:
    weights_true = weights_true[posts_n]
filters_true = np.dot(weights_true.reshape(n_neurons,n_basis_funcs), kernels.T) + bias_true
# filters_true = np.dot(weights_true[posts_n], kernels.T) + bias_true

sc_f = np.max(np.abs(filters_paglm)) / np.max(np.abs(filters_true))
# print(sc_f)

# n = 3
# plt.figure()
# plt.plot( np.exp(sc_f*filters_nemos[n]), c='k', label='exact')
# plt.plot( np.exp(filters_paglm[n]), c='r', label='paGLM')
# plt.legend()

fig, axs = plt.subplots(3,2,figsize=(7,9))
for n, ax in enumerate(axs.flat):
    if n > 1:
        ax.plot(time, inv_link(filters_nemos[n-2]), c='k', label='exact')
        ax.plot(time, quadratic(filters_paglm[n-2]), c='r', label='paGLM')
        ax.plot(time, inv_link(filters_true[n-2]), c='g', label='true')
        ax.set_xlabel("time from spike")
        ax.set_ylabel("gain")
    elif n == 0:
        ax.hist(inv_rates, bins=50, color='k', alpha=0.4)
        ax.axvline(np.percentile(inv_rates, 2.5), 0,1, color="k")
        ax.axvline(np.percentile(inv_rates, 97.5), 0, 1, color="k")
        ax.set_xlabel("inverse firing rates")
    else:
        coefs = paglm.compute_chebyshev(f, interval)
        xtw = np.dot(X, weights_paglm.flatten())
        x = np.arange(np.percentile(inv_rates, 2.5), np.percentile(inv_rates, 97.5), 0.01)
        ax.plot(x,f(x), c='k', label="exact nonlinearity")
        ax.plot(x,quadratic(x), c='r', label="approximation")
        ax.set_xlabel("approximation range")
        ax.legend(loc="upper left")
plt.legend(loc="upper right")
fig.suptitle(f"binsize {binsize}")
plt.tight_layout()
plt.show()


# plt.figure()
# plt.hist(quadratic(np.dot(X, weights_paglm.flatten())), bins=100, alpha=0.5,label="paglm", color='r')
# plt.hist(softplus(np.dot(X, weights_true.flatten())), bins=100, alpha=0.8, label="true", color='g')
# plt.xlabel("predicted fr")
# plt.ylabel("count")
# plt.legend()

# plt.figure()
# plt.plot(softplus(filters_nemos[0]), c='k', label='exact')
# plt.plot(quadratic(filters_paglm[0]), c='r', label='paGLM')
# plt.plot(softplus(filters_true[0]), c='g', label='true')
# plt.legend()

