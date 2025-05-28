import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

import nemos as nmo

import pynapple as nap

import paglm.core as paglm

from poisson_point_process.poisson_process_glm import PopulationContinuousPA, PopulationContinuousMC
from poisson_point_process.poisson_process_obs_model import PopulationPolynomialApproximation, MonteCarloApproximation
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral
from poisson_point_process.utils import quadratic

# jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--StepSize", help="Set initial step size")
parser.add_argument("-e", "--Epochs", help="Provide number of epochs")
parser.add_argument("-k", "--KFold", help="Provide kfold index")
parser.add_argument("-t", "--TimeRec", help="Provide recording time index")
parser.add_argument("-m", "--ModelName", help="Provide model name")
parser.add_argument("-b", "--Batches", help="Provide the number of batches for scan")
args = parser.parse_args()
init_lr = float(args.StepSize)
n_epochs = int(args.Epochs)
k = int(args.KFold)
tm = int(args.TimeRec)
model_name = str(args.ModelName)
n_b_scan = int(args.Batches)

# load data
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')[
    'c57LightOffTime'].squeeze().astype(float)
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']
audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
audio_segm = nap.IntervalSet(start=audio_segm[:, 0], end=audio_segm[:, 1])

test_sets = np.load("/mnt/home/amedvedeva/ceph/songbird_data/test_sets_5f.npz", mmap_mode='r', allow_pickle=True)

n_neurons = spikes_quiet.shape[0]
n_basis_funcs = 4
history_window = 0.004
# nonlinearity
inverse_link = jnp.exp
link_f = jnp.log
# need >1 if the number of spikes is too large to parallelize scan on GPU
binsize = 0.0001
window_size = int(history_window / binsize)
basis = GenLaguerreEval(history_window, n_basis_funcs)
time = jnp.linspace(0, history_window, window_size)
kernels = basis(-time)

obs_model_exact = nmo.observation_models.PoissonObservations(inverse_link)

spikes_quiet_ei = np.vstack((spikes_quiet[np.argwhere(ei_labels.squeeze() == 1).squeeze()],
                             spikes_quiet[np.argwhere(ei_labels.squeeze() == -1).squeeze()]))

ts_dict_quiet = {key: nap.Ts(spikes_quiet_ei[key, 0].flatten()) for key in range(spikes_quiet_ei.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

kfolds = 5
n_times = 5

test_duration = off_time * 0.15
off_interval = nap.IntervalSet(0, off_time)
test_intervals = [nap.IntervalSet(i * test_duration, i * test_duration + test_duration).set_diff(audio_segm) for i in
                  range(kfolds)]
train_intervals = [off_interval.set_diff(test_intervals[i]).set_diff(audio_segm) for i in range(kfolds)]
if n_times > 1:
    train_length = np.logspace(np.log10(10), np.log10(off_time- test_duration), n_times)
else:
    train_length = np.array([off_time - test_duration])

times = np.zeros((kfolds, n_times, 4))
scores_ll = np.zeros((kfolds, n_times, 4))
scores_r2 = np.zeros((kfolds, n_times, 4))
errors = np.zeros((kfolds, n_times, n_epochs))
weights_pad = []
weights_pa = []
weights_mc = []

pred_pad_l = []
pred_pa_l = []
pred_mc_l = []

# k = 0
# tm = 0
# model_name = "PAC"
print(f"K={k}, TM={tm}")
print(f"Model = {model_name}")
train_dur = nap.IntervalSet(train_intervals[k].start[0], train_intervals[k].start[0] + train_length[tm])
train_int = train_intervals[k].intersect(train_dur)
spikes_train = spike_times_quiet.restrict(train_int)
spikes_n = jnp.array([spikes_train[n].shape[0] for n in range(n_neurons)])
spike_ids = jnp.repeat(jnp.arange(n_neurons), spikes_n)
spike_times = jnp.concatenate([spikes_train.data[n].t for n in range(n_neurons)])
sorted_indices = jnp.argsort(spike_times)
spike_times = spike_times[sorted_indices]
spike_ids = spike_ids[sorted_indices]
X_spikes = jnp.vstack((spike_times, spike_ids))
y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))

# test set
spikes_test = spike_times_quiet.restrict(test_intervals[k])
spikes_n = jnp.array([spikes_test[n].shape[0] for n in range(n_neurons)])
spike_ids = jnp.repeat(jnp.arange(n_neurons), spikes_n)
spike_times = jnp.concatenate([spikes_test.data[n].t for n in range(n_neurons)])
sorted_indices = jnp.argsort(spike_times)
spike_times = spike_times[sorted_indices]
spike_ids = spike_ids[sorted_indices]
X_test = jnp.vstack((spike_times, spike_ids))

# counts_test = test_sets[f"test_set_{k}"].item()

# get approx inv fr interval (can try moving this outside to fix)
# approx_rates = (spikes_test.count(bin_size=0.05).to_numpy() + 0.05)  / 0.05
# interval = [np.percentile(link_f(approx_rates), 0.5, axis=0),
#             np.percentile(link_f(approx_rates), 99.5, axis=0)]
# interval_discrete = [np.percentile(link_f(approx_rates*binsize), 0.5, axis=0),
#             np.percentile(link_f(approx_rates*binsize), 99.5, axis=0)]

bounds = [-1.5, 0.7]
mean_rates = spikes_train.rates.to_numpy()
interval = [
    link_f(mean_rates) + bounds[0],
    link_f(mean_rates) + bounds[1],
]
interval_discrete = [
    link_f(mean_rates * binsize) + bounds[0],
    link_f(mean_rates * binsize) + bounds[1],
]

# get closed-form PA-c
if model_name == "PAC":
    n_batches_scan = n_b_scan
    obs_model_pa = PopulationPolynomialApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        approx_interval=interval,
        eval_function=basis,
        int_function=GenLaguerreInt(history_window, n_basis_funcs),
        prod_int_function=GenLaguerreProdIntegral(history_window, n_basis_funcs),
    )

    tt0 = perf_counter()
    model_pa = PopulationContinuousPA(
        solver_name="LBFGS",
        observation_model=obs_model_pa,
        solver_kwargs={"tol": 1e-12}
    ).fit_closed_form(X_spikes, y_spikes)
    times[k, tm, 2] = perf_counter() - tt0
    print(f"fit PA model, time: {times[k, tm, 2]}")

    # test_set = jnp.array(counts_test.toarray())
    # pred_pa = model_pa.predict(X_test, 10000, binsize, test_duration)
    # scores_r2[k, -1, 2] = obs_model_exact.pseudo_r2(counts_test, pred_pa * binsize)
    # scores_ll[k, -1, 2] = obs_model_exact.log_likelihood(counts_test, pred_pa * binsize)
    weights_pa.append(np.vstack((model_pa.coef_, model_pa.intercept_)))


# MONTE CARLO
elif model_name == "MC":
    def lr_schedule_mc(step):
        initial_lr = init_lr
        decay_rate = 0.999
        return initial_lr * decay_rate ** step

    n_batches_scan = n_b_scan
    obs_model = MonteCarloApproximation(
        inverse_link_function=inverse_link,
        n_basis_funcs=n_basis_funcs,
        n_batches_scan=n_batches_scan,
        history_window=history_window,
        eval_function=basis,
        mc_n_samples=2000000,
    )
    model_mc = PopulationContinuousMC(
        solver_name="GradientDescent",
        observation_model=obs_model,
        random_key=jax.random.PRNGKey(0),
        solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": lr_schedule_mc})

    tt0 = perf_counter()
    # initialize the model
    # pa_params = (model_pa.coef_, jnp.atleast_1d(model_pa.intercept_))
    params = model_mc.initialize_params(X_spikes, y_spikes)
    state = model_mc.initialize_state(X_spikes, y_spikes, params)
    # run updates
    num_iter = n_epochs
    error = np.zeros(num_iter)

    for step in range(num_iter):
        t0 = perf_counter()
        params, state = model_mc.update(params, state, X_spikes, y_spikes)
        error[step] = state.error
        if step % 100 == 0:
            print(f"step {step}, error: {error[step]}, time: {perf_counter() - t0}")
    times[k, tm, 3] = (perf_counter() - tt0) + times[k, tm, 2]
    print(f"fit MC model, time: {times[k, tm, 3]}")
    errors[k, tm] = error

    # test_set = jnp.array(counts_test.toarray())
    # pred_mc = model_mc.predict(X_test, 5000, binsize, test_duration)
    # scores_r2[k, -1, 3] = obs_model_exact.pseudo_r2(counts_test, pred_mc * binsize)
    # scores_ll[k, -1, 3] = obs_model_exact.log_likelihood(counts_test, pred_mc * binsize)
    weights_mc.append(np.vstack((model_mc.coef_, model_mc.intercept_)))


# BATCHED POLYNOMIAL APPROXIMATION DISCRETE
else:
    def sufficient_stats(spike_times, kernels, batch_size, ws, binsize):
        n_fun = kernels.shape[1]
        n_neurons = len(spike_times)
        N = n_fun * n_neurons
        n_batches = int(
            jnp.ceil((spike_times.time_support.end[-1] - spike_times.time_support.start[0]) / batch_size))
        batch_intervals = [
            nap.IntervalSet(spike_times.time_support.start[0] + i * batch_size,
                            spike_times.time_support.start[0] + i * batch_size + batch_size + history_window)
            for i in range(n_batches)
        ]
        sum_x = jnp.zeros(N + 1)
        sum_yx = jnp.zeros((N + 1, n_neurons))
        sum_xxT = jnp.zeros((N + 1, N + 1))
        sum_yxxT = jnp.zeros((N + 1, N + 1, n_neurons))

        for i, batch in enumerate(batch_intervals):
            yb = jnp.array(
                spike_times.count(ep=spike_times.time_support.intersect(batch), bin_size=binsize).to_numpy())
            Xb = jnp.array(nmo.convolve.create_convolutional_predictor(kernels, yb).reshape(-1, N))
            yb, Xb = yb[int(ws / binsize):], Xb[int(ws / binsize):]
            Xb = np.concatenate((Xb, np.ones([Xb.shape[0], 1])), axis=1)

            sum_x = sum_x + jnp.sum(Xb, axis=0)
            sum_yx = sum_yx + Xb.T @ yb
            sum_xxT = sum_xxT + Xb.T @ Xb
            print(f"{i + 1}/{len(batch_intervals)}")
            # sum_yxxT = sum_yxxT + jnp.einsum('ti,tj,tn->ijn', Xb, Xb, yb)

        return [sum_x, sum_yx, sum_xxT, sum_yxxT]

    tt0 = perf_counter()
    with jax.default_device(cpu):
        suff_discrete = sufficient_stats(spikes_train, kernels, 60., history_window, binsize)
    weights_d, interval_d = paglm.fit_paglm_population(inverse_link, suff_discrete, interval_discrete)
    times[k, tm, 1] = perf_counter() - tt0
    print(f"fit PA-d model, time: {times[k, tm, 1]}")

    # compute test log likelihood
    def batched_predict(spike_times, kernels, weights, batch_size, ws, binsize):
        n_fun = kernels.shape[1]
        n_neurons = len(spike_times)
        N = n_fun * n_neurons
        n_batches = int(
            np.ceil((spike_times.time_support.end[-1] - spike_times.time_support.start[0]) / batch_size))
        batch_intervals = [
            nap.IntervalSet(spike_times.time_support.start[0] + i * batch_size,
                            spike_times.time_support.start[0] + i * batch_size + batch_size + history_window)
            for i in range(n_batches)
        ]
        pred_paglm = []
        for i, batch in enumerate(batch_intervals):
            Xb = spike_times.count(ep=spike_times.time_support.intersect(batch), bin_size=binsize).to_numpy()
            Xb = nmo.convolve.create_convolutional_predictor(kernels, Xb).reshape(-1, N)
            Xb = Xb[int(ws / binsize):]
            pred_paglm.append(
                quadratic(np.dot(Xb, weights[:-1]) + weights[-1], inverse_link, interval_discrete))
            print(f"{i+1}/{len(batch_intervals)}")
        return jnp.concatenate(pred_paglm)

    # test_set = jnp.array(counts_test.toarray())
    # pred_paglm = batched_predict(spikes_test, kernels, weights_d, 60., history_window, binsize)
    # scores_r2[k, -1, 1] = obs_model_exact.pseudo_r2(counts_test[window_size:], pred_paglm)
    # scores_ll[k, -1, 1] = obs_model_exact.log_likelihood(counts_test[window_size:], pred_paglm)
    weights_pad.append(weights_d)


# save params and kernels (not filters)
results = {
    "params_paD": weights_pad,
    "params_paC": weights_pa,
    "params_mc": weights_mc,
    "times": times,
    "scores_ll": scores_ll,
    "scores_r2": scores_r2,
    "error_mc": errors,
    "kernels": kernels,
    "binsize": binsize,
}

np.savez(f"/mnt/home/amedvedeva/ceph/real_data_{k}k_{tm}_tm_{model_name}.npz", **results)
print("Script terminated")

# plots
# results = np.load("/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbird_output/MC_fit_83.npz")
# for key in results.files:
#     globals()[key] = results[key]
# audio_segm = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57AudioSegments.mat')['c57AudioSegments']
# off_time = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57LightOffTime.mat')['c57LightOffTime']
# ei_labels = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57EI.mat')['c57EI']
# spikes_quiet = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
#
# ts_dict_all = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
# spike_times_all = nap.TsGroup(ts_dict_all)
# spike_times_all["EI"] = ei_labels
# inh = spike_times_all.getby_category("EI")[-1]
# exc = spike_times_all.getby_category("EI")[1]
# spike_times_sorted = nap.TsGroup.merge(exc, inh, reset_index=True)
#
# weights, bias = model.coef_, model.intercept_
# filters = np.dot(weights.reshape(195,-1), kernels.T) + bias
# filters_raw = np.dot(weights.reshape(195,-1), kernels.T)
#
# def plot_coupling(responses, cmap_name="bwr",
#                       figsize=(8, 6), fontsize=15, alpha=0.5, cmap_label="hsv"):
#     # plot heatmap
#     sum_resp = np.sum(responses.reshape(196,-1), axis=1)
#     # normalize by cols (for fixed receiver neuron, scale all responses
#     # so that the strongest peaks to 1)
#     sum_resp_n = sum_resp/ sum_resp.max()
#     # scale to 0,1
#     color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()
#     color = color.reshape(14,14,-1)
#     cmap = plt.get_cmap(cmap_name)
#     n_row, n_col, n_tp = responses.shape
#     time = np.arange(n_tp)
#     fig, axs = plt.subplots(n_row, n_col, figsize=figsize, sharey=True)
#     for rec, rec_resp in enumerate(responses):
#         for send, resp in enumerate(rec_resp):
#             axs[rec, send].plot(time, responses[rec, send], color="k")
#             axs[rec, send].spines["left"].set_visible(False)
#             axs[rec, send].spines["bottom"].set_visible(False)
#             axs[rec, send].set_xticks([])
#             axs[rec, send].set_yticks([])
#             axs[rec, send].axhline(0, color="k", lw=0.5)
#     for rec, rec_resp in enumerate(responses):
#         for send, resp in enumerate(rec_resp):
#             xlim = axs[rec, send].get_xlim()
#             ylim = axs[rec, send].get_ylim()
#             rect = plt.Rectangle(
#                 (xlim[0], ylim[0]),
#                 xlim[1] - xlim[0],
#                 ylim[1] - ylim[0],
#                 alpha=alpha,
#                 color=cmap(color[rec, send]),
#                 zorder=1
#             )
#             axs[rec, send].add_patch(rect)
#             axs[rec, send].set_xlim(xlim)
#             axs[rec, send].set_ylim(ylim)
#     plt.suptitle(f"Filters to neuron {target_neu_id}", fontsize=fontsize)
#     return fig
# filter_plot = np.append(filters_raw, filters_raw[-1]).reshape(14,14,40)
# plot_coupling(filter_plot)
#
# ref_spikes = nap.TsGroup({target_neu_id: spike_times_sorted[target_neu_id]})
# ccg = nap.compute_crosscorrelogram((spike_times_sorted,ref_spikes), 0.0001, 0.004, norm=False)
# acg = nap.compute_autocorrelogram(ref_spikes, 0.0001, 0.004, norm=False)
# ccg[(target_neu_id,target_neu_id)] = acg[target_neu_id]
# pres = np.sort(np.flip(np.argsort(np.abs(filters_raw).max(1)))[:12])
# fig, axs = plt.subplots(4,3, figsize=(12,9))
# axs = axs.flat
# for i, ax in enumerate(axs):
#     pair = (pres[i],target_neu_id)
#     ax2=ax.twinx()
#     bar = ax2.bar(ccg[pair].index, ccg[pair], width=0.0001, alpha=0.3, color='lightsteelblue',
#                       label="CCG")
#     ax2.set_xlim(0, 0.004)
#     # ax2.set_yscale("log")
#     if i%3!=0:
#         ax2.set_ylabel("spikes count")
#     filter = inverse_link(filters[pair[0]])
#     # filter = filters_raw[pair[0]]
#     line2 = ax.plot(time, filter, c='darkred', lw=2, label=f"from n{pres[i]}")
#     if i%3==0:
#         ax.set_ylabel("filter")
#     ax.set_xlabel("time (ms)")
#     lines = [bar, line2[0]]
#     labels = [l.get_label() for l in lines]
#     ax.axhline(0, 0, 1, color='k', lw=0.5)
#     ax.legend(lines, labels, loc="upper right")
# fig.suptitle(f"filters to n{target_neu_id}")
# plt.tight_layout()