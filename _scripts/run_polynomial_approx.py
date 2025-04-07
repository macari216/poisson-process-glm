import os
import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import paglm.core as paglm

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import ContinuousPA
from poisson_point_process.poisson_process_obs_model import PolynomialApproximation
from poisson_point_process.utils import quadratic
from poisson_point_process.basis import RaisedCosineLogEval, LaguerreEval

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# generate data
n_neurons = 5
history_window = 0.004
tot_time_sec = 1000
binsize = 0.0001
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
# rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
# time, kernels = rc_basis.evaluate_on_grid(window_size)
# time *= history_window
# inverse_link = jax.nn.softplus
# link_f = lambda x: jnp.log(jnp.exp(x) -1.)
inverse_link = jnp.exp
link_f = jnp.log

basis_fn = LaguerreEval(history_window, n_basis_funcs)
time = jnp.linspace(0,history_window,window_size)
kernels = basis_fn(-time)

np.random.seed(123)

####
pres_rate_per_sec = 10
posts_rate_per_sec = 3
# rescaled proportionally to the binsize
bias_true = posts_rate_per_sec + np.log(binsize)
posts_rate_sim = inverse_link(posts_rate_per_sec + np.log(binsize)) / binsize
weights_true = np.random.normal(0, 0.3, n_neurons * n_basis_funcs)
# weights_true = jnp.array([0.5,0.3,0.2,0.4])
X, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                                           n_bins_tot, n_neurons, weights_true, window_size, kernels,
                                                           inverse_link)

print(f"X spikes: {X_counts.sum(0)}, y spikes: {y_counts.sum()}")
print(f"max spikes per bin: {jnp.hstack((y_counts[:, None], X_counts)).max()}")
print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(y_counts.sum(0))}")
# plt.plot(lam_posts)

spike_times, spike_ids = simulate.poisson_times(X_counts, tot_time_sec, binsize)
spike_times_y, _ = simulate.poisson_times(y_counts[:, None], tot_time_sec, binsize)
X_spikes = jnp.vstack((spike_times, spike_ids))
target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)
y_spikes = jnp.vstack((spike_times_y, target_idx))

X_bias = jnp.concatenate((X, jnp.ones([X.shape[0], 1])), axis=1)
X_discrete = X_bias
y_discrete = y_counts


# compute coefficients
bounds = [-0.5,0.5]

mean_rate = spike_times_y.size / tot_time_sec
# interval = [
#     float(link_f(mean_rate) + bounds[0]),
#     float(link_f(mean_rate) + bounds[1]),
# ]
# interval_discrete = [
#     float(link_f(mean_rate*binsize) + bounds[0]),
#     float(link_f(mean_rate*binsize) + bounds[1]),
# ]
interval = [np.percentile(link_f(lam_posts/binsize), 2.5), np.percentile(link_f(lam_posts/binsize), 97.5)]
interval_discrete = [np.percentile(link_f(lam_posts), 2.5), np.percentile(link_f(lam_posts), 97.5)]
# print([np.percentile(link_f(lam_posts/binsize), 2.5), np.percentile(link_f(lam_posts/binsize), 97.5)])
print(interval)
print(interval_discrete)

x = jnp.arange(interval[0], interval[1], 0.01)
plt.figure()
plt.plot(x, inverse_link(x), c='k', label="exact nonlinearity")
plt.plot(x, quadratic(x, inverse_link, interval), c='r', label="approximation")
plt.legend()
plt.show()

# interval_discrete = [np.percentile(inv_rates, 5), np.percentile(inv_rates, 95)]

# interval = interval_discrete

obs_model_pa = PolynomialApproximation(
    inverse_link_function=inverse_link,
    n_basis_funcs=n_basis_funcs,
    n_batches_scan=n_batches_scan,
    history_window=history_window,
    window_size=window_size,
    approx_interval=interval,
    eval_function=basis_fn,
)

tt0 = perf_counter()
model_pa = ContinuousPA(solver_name="LBFGS", observation_model=obs_model_pa, solver_kwargs={"tol":1e-12}).fit_closed_form(X_spikes, y_spikes)
print(f"fit continuous PA GLM model, {perf_counter() - tt0}")

obs_model_exact = nmo.observation_models.PoissonObservations(inverse_link)
tt0 = perf_counter()
model_exact = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model_exact,solver_kwargs={"tol":1e-12}).fit(X, y_discrete)
print(f"fit discrete GLM model, {perf_counter() - tt0}")

def sufficient_stats(X, y):
    sum_x = np.sum(X, axis=0)
    sum_yx = np.sum(y[:, np.newaxis] * X, axis=0)
    sum_xxT = X.T @ X
    sym_yxxT = X.T @ (y[:, np.newaxis] * X)

    return [sum_x, sum_yx, sum_xxT, sym_yxxT]

tt0 = perf_counter()
suff_discrete = sufficient_stats(X_bias, y_discrete)
weights_d, interval_d = paglm.fit_paglm(inverse_link, suff_discrete, [interval_discrete])
print(f"fit discrete PA GLM model, {perf_counter() - tt0}")

weights_paglm, bias_paglm = weights_d[:-1].reshape(n_neurons,n_basis_funcs), weights_d[-1]
weights_nemos, bias_nemos = model_exact.coef_.reshape(-1,n_basis_funcs), model_exact.intercept_
weights_pa, bias_pa = model_pa.coef_.reshape(-1,n_basis_funcs), model_pa.intercept_

filters_pa = np.dot(weights_pa, kernels.T) + bias_pa
filters_paglm = np.dot(weights_paglm, kernels.T) + bias_paglm
filters_nemos = np.dot(weights_nemos, kernels.T) + bias_nemos
filters_true = np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T) + bias_true

# scores
pred_pa = model_pa.predict(X_spikes, binsize)
pred_paglm = quadratic(np.dot(X, weights_d[:-1])+weights_d[-1], inverse_link, interval_discrete)
pred_exact = model_exact.predict(X)
print(f"continuous PA GLM pseudo-r2 score: {np.round(obs_model_exact.pseudo_r2(y_counts, pred_pa*binsize, score_type="pseudo-r2-Cohen"), 5)}")
print(f"discrete GLM pseudo-r2  score: {np.round(obs_model_exact.pseudo_r2(y_counts, pred_exact, score_type="pseudo-r2-Cohen"), 5)}")
print(f"discrete PA GLM pseudo-r2  score: {np.round(obs_model_exact.pseudo_r2(y_counts, pred_paglm, score_type="pseudo-r2-Cohen"), 5)}")

#compare predicted rate
def select_spk(spikes, ep):
    return spikes[0, (spikes[0]>ep[0]*binsize)&(spikes[0]<ep[1]*binsize)]/binsize
eps = [(260000,270000),(710000,720000)]
fig, axs = plt.subplots(2,1, figsize=(12,6))
for i, ax in enumerate(axs):
    ax.plot(np.arange(eps[i][0],eps[i][1]), lam_posts[eps[i][0]:eps[i][1]]/binsize, c='g', lw=0.5, label="true")
    ax.plot(np.arange(eps[i][0],eps[i][1]), pred_exact[eps[i][0]:eps[i][1]]/binsize, c='k', lw=0.5, label="exact")
    ax.plot(np.arange(eps[i][0], eps[i][1]), pred_paglm[eps[i][0]:eps[i][1]]/binsize, c='b', lw=0.5, label="pa discrete")
    ax.plot(np.arange(eps[i][0], eps[i][1]), pred_pa[eps[i][0]:eps[i][1]], c='r', lw=0.5, label="pa continuous")
    ax.vlines(select_spk(y_spikes, eps[i]), -10, pred_pa[eps[i][0]:eps[i][1]].min(), color='darkblue', lw=0.8, label="spikes")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (Hz)")
axs[0].legend(loc="upper right")
plt.tight_layout()

# compare filters
fig, axs = plt.subplots(3,2,figsize=(7,9))
axs = axs.flat
for n, ax in enumerate(axs):
    if n==0:
        x = jnp.arange(interval[0], interval[1], 0.01)
        ax.plot(x, inverse_link(x), c='k', label="exact nonlinearity")
        ax.plot(x,quadratic(x, inverse_link, interval), c='r', label="approximation")
        ax.legend()
    else:
        line1 = ax.plot(time, inverse_link(filters_true[n-1])/binsize, c='g', label='true')
        line2 = ax.plot(time, inverse_link(filters_nemos[n-1])/binsize, c='k', label='exact discrete')
        line3 = ax.plot(time, quadratic(filters_pa[n-1], inverse_link, interval), c='r', label='pa continuous')
        line4 = ax.plot(time, quadratic(filters_paglm[n-1], inverse_link, interval_discrete)/binsize, c='b', label='pa discrete')
        ax.set_xlabel("time from spike")
        ax.set_ylabel("gain")

        if n==1:
            lines = [line1[0], line2[0], line3[0], line4[0]]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")

plt.tight_layout()
plt.show()

