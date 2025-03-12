import os
import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

import nemos as nmo

import pynapple as nap
import matplotlib.pyplot as plt

from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

t0 = perf_counter()
# off_time = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57LightOffTime.mat')['c57LightOffTime'][0]
# spikes_quiet = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
# ei_labels = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57EI.mat')['c57EI']

off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime'][0]
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--TargetID", help="Provide postsynaptic neuron id")
parser.add_argument("-e", "--Epochs", help="Provide number of epochs")
args = parser.parse_args()
target_neu_id = int(args.TargetID)
n_epochs = int(args.Epochs)

# sort by exc/inh
spikes_quiet_ei = np.vstack((spikes_quiet[np.argwhere(ei_labels.squeeze()==1).squeeze()],spikes_quiet[np.argwhere(ei_labels.squeeze()==-1).squeeze()]))
# sort spike times and create marked X and y spike arrays
spikes_n = jnp.array([spikes_quiet_ei[n][0][spikes_quiet_ei[n][0] < off_time].size for n in range(195)])
spike_ids = jnp.repeat(jnp.arange(195), spikes_n)
spike_times = jnp.concatenate([spikes_quiet_ei[n][0][spikes_quiet_ei[n][0] < off_time[0]] for n in range(195)])
sorted_indices = jnp.argsort(spike_times)
spike_times = spike_times[sorted_indices]
spike_ids = spike_ids[sorted_indices]
X_spikes = jnp.vstack((spike_times, spike_ids))
spike_idx_target = jnp.arange(len(spike_times))[spike_ids == target_neu_id]
y_spikes = jnp.vstack((spike_times[spike_idx_target], spike_idx_target))

print(f"loaded data, {perf_counter() - t0}")

# set model params
n_neurons = spikes_quiet.shape[0]
n_basis_funcs = 8
history_window = 0.004
# nonlinearity
inverse_link = jnp.exp
# need >1 if the number of spikes is too large to parallelize scan on GPU
n_batches_scan = 1

# for plotting and predicting rates
binsize = 0.0001
window_size = int(history_window / binsize)
rc_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs, window_size=window_size)
time, kernels = rc_basis.evaluate_on_grid(window_size)
time *= history_window

obs_model_kwargs = {
    "n_basis_funcs": n_basis_funcs,
    "history_window": history_window,
    "inverse_link_function": inverse_link,
    "n_batches_scan": n_batches_scan,
    "mc_random_key": jax.random.PRNGKey(0),
    "mc_n_samples": y_spikes.shape[1]*2,
}

model = ContinuousMC(
    solver_name="GradientDescent",
    obs_model_kwargs=obs_model_kwargs,
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": 1e-7}
)

# initialize the model
params = model.initialize_params(X_spikes, y_spikes)
state = model.initialize_state(X_spikes, y_spikes, params)
# run updates
num_iter = n_epochs
tt0 = perf_counter()
error = np.zeros(num_iter)
loss = np.zeros(num_iter)
for step in range(num_iter):
    t0 = perf_counter()
    params, state = model.update(params, state, X_spikes, y_spikes)
    error[step] = state.error
    t1 = perf_counter()
    loss[step] = model._negative_log_likelihood(X_spikes, y_spikes, params, state.aux)
    if step % 10 == 0:
        print(f"step {step}, time: {t1-t0}, error: {error[step]}")
print(f"fit model, time: {perf_counter() - tt0}")

results = {
    "target_neu_id": target_neu_id,
    "error": error,
    "loss": loss,
    "weights": model.coef_,
    "bias": model.intercept_,
    "binsize": binsize,
    "kernels": kernels,
    "time": time,
}

np.savez(f"/mnt/home/amedvedeva/ceph/MC_fit_{target_neu_id}.npz", **results)
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