import os
import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

import nemos as nmo

import matplotlib.pyplot as plt

from poisson_point_process.poisson_process_glm import ContinuousMC

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

t0 = perf_counter()
# off_time = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57LightOffTime.mat')['c57LightOffTime'][0]
# spikes_quiet = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
# ei_labels = sio.loadmat('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57EI.mat')['c57EI']

off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime'][0]
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

# y spikes
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--TargetID", help="Provide postsynaptic neuron id")
args = parser.parse_args()
target_neu_id = int(args.TargetID)

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
n_basis_funcs = 4
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
num_iter = 500
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
print(f"fitted model, time: {perf_counter() - tt0}")

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