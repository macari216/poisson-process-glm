import os
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process import simulate
from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

all_to_all = True

# generate data
n_neurons = 9
target_neu_id = 0
history_window = 0.1
tot_time_sec = 1000
binsize = 0.001
window_size = int(history_window / binsize)
n_basis_funcs = 4
n_bins_tot = int(tot_time_sec / binsize)
n_batches_scan = 1
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window
inverse_link = jnp.exp

np.random.seed(216)

t0 = perf_counter()
if all_to_all:
    # all-to-all connectivity (bias and weights)

    # true parameters
    baseline_fr = 2.1
    biases = jnp.array(np.abs(np.random.normal(baseline_fr, baseline_fr / 10, n_neurons))) + np.log(binsize)
    bias_true = biases[target_neu_id]
    posts_rate_per_sec = biases[target_neu_id] - np.log(binsize)

    weights_true = jnp.array(np.random.normal(0, 0.1, size=(n_neurons, n_neurons, n_basis_funcs)))
    params = (weights_true, biases)

    spike_counts, rates = simulate.poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, kernels, params, inverse_link)

    print(f"total spikes: {spike_counts.sum()}, y spikes: {spike_counts[:, target_neu_id].sum()}")
    print(f"mean spikes per neuron: {jnp.mean(spike_counts.sum(0))}")
    print(f"max spikes per bin: {spike_counts.max()}")
    print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(spike_counts.sum(0))}")
    plt.plot(rates[:, target_neu_id])
    plt.show()

    spike_times, spike_ids = simulate.poisson_times(spike_counts, tot_time_sec, binsize)

    X_spikes = jnp.vstack((spike_times,spike_ids))
    spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
    y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

    rc_basis.set_input_shape(spike_counts)
    X = rc_basis.compute_features(spike_counts)
    y_counts = spike_counts[:, target_neu_id]
    lam_posts = rates[:, target_neu_id]

else:
    # all-to-one connectivity
    pres_rate_per_sec = 3
    posts_rate_per_sec = 2
    # rescaled proportionally to the binsize
    bias_true =  posts_rate_per_sec + np.log(binsize)
    # bias_true = jnp.log(jnp.exp(binsize*jax.nn.softplus(posts_rate_per_sec))-1)
    weights_true = jnp.array(np.random.normal(0, 0.5, n_neurons * n_basis_funcs))
    # weights_true = jnp.array([0.8, 0.5])
    X, y_counts, X_counts, lam_posts = simulate.poisson_counts(pres_rate_per_sec, posts_rate_per_sec, binsize,
                                               n_bins_tot, n_neurons, weights_true, window_size, rc_basis, inverse_link)
    inv_rates = np.log(np.exp(lam_posts) - 1)
    print(f"y spikes total: {y_counts.sum()}")
    print(f"X spikes total: {X_counts.sum()}")
    print(f"mean spikes per neuron: {jnp.mean(jnp.hstack((y_counts[:,None], X_counts)).sum(0))}")
    print(f"max spikes per bin: {jnp.hstack((y_counts[:,None], X_counts)).max()}")
    print(f"ISI (ms): {tot_time_sec*1000 / jnp.mean(jnp.hstack((y_counts[:,None], X_counts)).sum(0))}")
    # plt.hist(inv_rates, bins=50)
    plt.plot(lam_posts)
    plt.show()

    spike_times, spike_ids = simulate.poisson_times(X_counts, tot_time_sec, binsize)
    spike_times_y, _ = simulate.poisson_times(y_counts[:, None], tot_time_sec, binsize)
    X_spikes = jnp.vstack((spike_times, spike_ids))
    target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)-1
    y_spikes = jnp.vstack((spike_times_y, target_idx))

print(f"generated data: {np.round(perf_counter() - t0, 5)}")

obs_model_kwargs = {
    "n_basis_funcs": n_basis_funcs,
    "history_window": history_window,
    "inverse_link_function": inverse_link,
    "n_batches_scan": n_batches_scan,
    "mc_random_key": jax.random.PRNGKey(0),
    "mc_n_samples": y_spikes.shape[1]*10,
}
print(y_spikes.shape[1]*10)

model = ContinuousMC(solver_name="GradientDescent", obs_model_kwargs=obs_model_kwargs, solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 1e-7})
true_params = (weights_true[target_neu_id].ravel(), jnp.array([posts_rate_per_sec]))
params = model.initialize_params(X_spikes, y_spikes)
state = model.initialize_state(X_spikes, y_spikes, params)

num_iter = 1000
tt0 = perf_counter()
error = np.zeros(num_iter)
for step in range(num_iter):
    t0 = perf_counter()
    params, state = model.update(params, state, X_spikes, y_spikes)
    # error[step] = state.error
    error[step] = model._negative_log_likelihood(X_spikes, y_spikes, params, state.aux)
    t1 = perf_counter()
    if step % 50 == 0:
        print(f"step {step}, time: {t1 - t0}, error: {error[step]}")
print(f"fitted model, {perf_counter() - tt0}")

obs_model = nmo.observation_models.PoissonObservations(inverse_link)
model_exact = nmo.glm.GLM(solver_name="LBFGS",observation_model=obs_model,solver_kwargs={"tol":1e-12}).fit(X, y_counts)

weights_nemos, bias_nemos = model_exact.coef_.reshape(-1,n_basis_funcs), model_exact.intercept_
weights_mc, bias_mc = model.coef_.reshape(-1,n_basis_funcs), model.intercept_
filters_mc = np.dot(weights_mc, kernels.T) + bias_mc
filters_nemos = np.dot(weights_nemos, kernels.T) + bias_nemos -jnp.log(binsize)
if len(weights_true.shape) == 3:
    weights_true = weights_true[target_neu_id]
filters_true = np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T) + bias_true -jnp.log(binsize)

print(f"true weights: {weights_true, bias_true} ({posts_rate_per_sec})")
print(f"MC weights: {model.coef_, model.intercept_}")
print(f"nemos weights: {model_exact.coef_, model_exact.intercept_}")

results = {
    "error": error,
    "true_params": (weights_true, bias_true),
    "mc_params": (weights_mc, bias_mc),
    "exact_params": (weights_nemos, bias_nemos),
    "kernels": kernels,
    "time": time,
}

# np.save(f"/mnt/home/amedvedeva/ceph/MC_fit.npz", results)
# print("Script terminated")

# compare scores
pred_MC = model.predict(X_spikes, binsize)
pred_exact = model_exact.predict(X)
print(f"discrete (exact) pseudo-r2 score: {np.round(obs_model.pseudo_r2(y_counts, pred_exact), 5)}")
print(f"continuous (approx) pseudo-r2 score: {np.round(obs_model.pseudo_r2(y_counts, pred_MC*binsize), 5)}")
print(f"discrete (exact) log-likelohood score: {np.round(obs_model.log_likelihood(y_counts, pred_exact), 5)}")
print(f"continuous (approx) log-likelohood score: {np.round(obs_model.log_likelihood(y_counts, pred_MC*binsize), 5)}")

#compare predicted rate
def select_spk(spikes, ep):
    return spikes[0, (spikes[0] > ep[0] * binsize) & (spikes[0] < ep[1] * binsize)] / binsize
eps = [(260000,270000),(710000,720000)]
# eps = [(13000,16000),(71000,74000)]
fig, axs = plt.subplots(2,1, figsize=(12,6))
for i, ax in enumerate(axs):
    ax.plot(np.arange(eps[i][0],eps[i][1]), lam_posts[eps[i][0]:eps[i][1]]/binsize, c='g', lw=0.5, label="true")
    ax.plot(np.arange(eps[i][0],eps[i][1]), pred_exact[eps[i][0]:eps[i][1]]/binsize, c='k', lw=0.5, label="exact")
    ax.plot(np.arange(eps[i][0], eps[i][1]), pred_MC[eps[i][0]:eps[i][1]], c='r', lw=0.5, label="approx")
    ax.vlines(select_spk(y_spikes, eps[i]), -10, 0.8, color='darkblue', lw=0.8, label="spikes")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (Hz)")
axs[0].legend(loc="upper right")
plt.suptitle("predicted firing rate")
plt.tight_layout()


#compare filters
fig, axs = plt.subplots(5,2,figsize=(7,9))
# fig, axs = plt.subplots(1,2,figsize=(7,5))
axs = axs.flat
for n, ax in enumerate(axs):
    if n==0:
        ax.plot(np.arange(0, num_iter),error, c='k')
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    else:
        line1 = ax.plot(time, jnp.exp(filters_true[n-1]), c='g', label='true')
        line2 = ax.plot(time, jnp.exp(filters_nemos[n-1]), c='k', label='exact')
        line3 = ax.plot(time, jnp.exp(filters_mc[n-1]), c='r', label='approx')
        ax.set_xlabel("time from spike")
        ax.set_ylabel("gain")

        if n==len(axs)-1:
            lines = [line1[0], line2[0], line3[0]]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")
plt.tight_layout()
plt.show()