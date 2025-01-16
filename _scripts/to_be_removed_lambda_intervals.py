### this is an old idea that didn't work I'll move it somewhere else later

import os
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import scipy.io as sio


def poisson_nll(predicted_rate, spike_indices, binsize):
    lambda_st = jnp.array(
        [
            sum(predicted_rate[spike_indices[k], k])
            for k in range(predicted_rate.shape[1])
        ]
    )
    lambda_int = predicted_rate.sum(0) * binsize

    return jnp.mean(lambda_int - jnp.log(lambda_st))


@jax.jit
def update_lambda(lambda_hat, d_lams, all_k_indices):
    for i in range(len(d_lams)):
        all_indices = all_k_indices[i]
        valid_mask = all_indices < lambda_hat.shape[0]
        all_indices = jnp.where(
            valid_mask, all_indices, 0
        )  # Replace out-of-bounds with index 0 ???
        updates = d_lams[i] * valid_mask[..., None]
        lambda_hat = lambda_hat.at[all_indices].add(updates)

    return lambda_hat


def compute_lam_k(spikes_k):
    delta = nap.IntervalSet(t_k - ws_sec, t_k)
    S_delta = spikes_tsd.restrict(delta)
    taus = jnp.round(t_k - S_delta.t).astype(int)
    lam_k = jnp.sum(jnp.log(inverse_link_fun(d_lams[S_delta.d, taus, k])))
    return lam_k


def comp_tau(y_k, t_k, binsize=0.0001):
    return jnp.floor((y_k - t_k) / binsize).astype(int)


@jax.jit
def comp_lam_yk(log_lambda, spikes, y_times, n_indices, ws_sec):
    for y_k in spikes:
        delta_mask = (y_times >= y_k - ws_sec) & (y_times < y_k)
        S_delta = jnp.where(delta_mask)[0]
        S_delta_t = jnp.take(y_times, S_delta)
        S_delta_n = jnp.take(n_indices, S_delta)

        y_k_sum = sum(
            jnp.dot(basis_taus[comp_tau(y_k, t_k)], w[i, :, n])
            for t_k, i in zip(S_delta_t, S_delta_n)
        )

        log_lambda = log_lambda.at[0].add(jnp.log(inverse_link_fun(y_k_sum)))

        return log_lambda


def compute_intervals_and_mappings(spikes_array, ws_sec):
    spike_times = np.concatenate([spikes for spikes in spikes_array]).squeeze()
    spike_indices = np.concatenate(
        [np.tile(k, len(spikes_array[k])) for k in range(spikes_array.shape[0])]
    ).squeeze()
    sorted_indices = np.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    spike_indices = spike_indices[sorted_indices]

    merged_intervals = []
    spike_mappings = {}

    current_start = spike_times[0]
    current_end = current_start + ws_sec

    interval_spikes = [[] for _ in range(spikes_array.shape[0])]
    interval_spikes[spike_indices[0]].append(current_start)

    for idx, spike_time in enumerate(spike_times[1:]):
        if spike_time <= current_end:
            k = spike_indices[idx]
            interval_spikes[k].append(spike_time)
            current_end = max(current_end, spike_time + ws_sec)
        else:
            merged_intervals.append((float(current_start), float(current_end)))
            interval_spikes = [
                np.array(interval_spikes[k]) for k in range(len(interval_spikes))
            ]
            spike_mappings[merged_intervals[-1]] = interval_spikes
            interval_spikes = [[] for _ in range(spikes_array.shape[0])]
            k = spike_indices[idx]
            interval_spikes[k].append(spike_time)
            current_start = spike_time
            current_end = spike_time + ws_sec
    merged_intervals.append((float(current_start), float(current_end)))
    interval_spikes = [
        np.array(interval_spikes[k]) for k in range(len(interval_spikes))
    ]
    spike_mappings[merged_intervals[-1]] = interval_spikes

    return merged_intervals, spike_mappings


os.environ["JAX_PLATFORM_NAME"] = "cpu"

# load data
# basis_gp = np.load('/mnt/home/amedvedeva/ceph/songbird_data/gp_basis.npz', allow_pickle=True)
# spikes_quiet_sorted = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/spikes_quiet_adj.mat')['spikes_quiet_adj'].squeeze()
# audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
# off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
# spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
# ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

basis_gp = np.load("/songbirds_data/gp_basis.npz", allow_pickle=True)
spikes_quiet_sorted = np.load("/songbirds_data/spikes_quiet_adj.npy", allow_pickle=True)
audio_segm = sio.loadmat("/songbirds_data/c57AudioSegments.mat")["c57AudioSegments"]
ei_labels = sio.loadmat("/songbirds_data/c57EI.mat")["c57EI"]
spikes_quiet = sio.loadmat("/songbirds_data/c57SpikeTimesQuiet.mat")[
    "c57SpikeTimesQuiet"
]
audio_segm = nap.IntervalSet(start=audio_segm[:, 0], end=audio_segm[:, 1])

# set params
ws = 40
binsize = 0.0001
n_fun = 10
n_neurons = 15
kernels = basis_gp["gp_var"] * basis_gp["gp_kernels"]
inverse_link_fun = jax.nn.softplus

# construct binned counts and convolved X
ts_dict_quiet = {
    key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(n_neurons)
}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)
spike_times_quiet["EI"] = ei_labels[:n_neurons]
inh = spike_times_quiet.getby_category("EI")[-1]
exc = spike_times_quiet.getby_category("EI")[1]
spike_times_sorted = nap.TsGroup.merge(exc, inh, reset_index=True)
tot_time = nap.IntervalSet(0, 31).set_diff(audio_segm)

y = np.array(spike_times_sorted.count(ep=tot_time, bin_size=binsize))
X = nmo.convolve.create_convolutional_predictor(kernels, y).reshape(
    -1, n_fun * n_neurons
)
print(X.shape)

# construct jax compatible list os spikes
spikes_quiet_sorted = spikes_quiet_sorted[:n_neurons]
for i in range(n_neurons):
    spikes_quiet_sorted[i] = spikes_quiet_sorted[i][
        spikes_quiet_sorted[i] < 31 - 0.83586667
    ]
spikes_quiet_ex_jax = [jnp.array(arr) for arr in spikes_quiet_sorted]

last_sp = max(
    [
        jnp.maximum(spikes_quiet_ex_jax[0].max(), spikes_quiet_ex_jax[1].max())
        for i in range(14)
    ]
)
n_bins = int(round(last_sp / binsize + ws))

# train glm to obtain model params
mask = np.zeros((n_neurons, n_neurons * n_fun))
for i in range(n_neurons):
    mask[i, i * n_fun : i * n_fun + n_fun] = np.ones(n_fun)
obs_model = nmo.observation_models.PoissonObservations(inverse_link_fun)
regularizer = nmo.regularizer.GroupLasso(mask=mask)
model = nmo.glm.PopulationGLM(
    observation_model=obs_model,
    solver_name="ProximalGradient",
    regularizer=regularizer,
    solver_kwargs=dict(tol=10**-12),
    regularizer_strength=1e-7,
)
print("fitting glm")
model.fit(X, y)
w_hat = model.coef_.reshape(n_neurons, n_fun, n_neurons)

# standard nemos (compute X and compute ll per bin)
t0 = perf_counter()
X = nmo.convolve.create_convolutional_predictor(kernels, y).reshape(
    -1, n_fun * n_neurons
)
nemos_lambda = model.predict(X)
print(f"standard: {perf_counter() - t0}")

y = y[~jnp.isnan(nemos_lambda).any(axis=1)]
nemos_lambda = nemos_lambda[~jnp.isnan(nemos_lambda).any(axis=1)]
neg_ll_nemos = model._observation_model._negative_log_likelihood(y, nemos_lambda)


# don't compute X and compute ll per spike
update_indices = jnp.arange(ws)

spike_indices = np.array(
    [
        np.array(
            [np.round(ts / binsize).astype(int) + 1 for ts in spikes_quiet_sorted[i]]
        )
        for i in range(n_neurons)
    ],
    dtype=object,
)
all_indices = [np.add.outer(k_indices, update_indices) for k_indices in spike_indices]
all_indices = [jnp.array(arr) for arr in all_indices]

t0 = perf_counter()
d_lams = [jnp.dot(kernels, w_hat[i]) for i in range(n_neurons)]
loop_lambda = jnp.tile(model.intercept_, (n_bins, 1))

loop_lambda = inverse_link_fun(update_lambda(loop_lambda, d_lams, all_indices))
print(f"looped: {perf_counter() - t0}")

neg_ll = poisson_nll(loop_lambda, spike_indices, binsize)
print(neg_ll_nemos)
print(neg_ll)

plt.figure()
plt.plot(nemos_lambda[:, 6])
plt.plot(loop_lambda[:, 6])


n_neurons = 15
ws_sec = 0.004
ws = 40
spike_times = np.concatenate([spikes for spikes in spikes_quiet_sorted]).squeeze()
spike_indices = np.concatenate(
    [
        np.tile(k, len(spikes_quiet_sorted[k]))
        for k in range(spikes_quiet_sorted.shape[0])
    ]
).squeeze()
sorted_indices = np.argsort(spike_times)
spike_times = spike_times[sorted_indices]
spike_indices = spike_indices[sorted_indices]
spikes_tsd = nap.Tsd(t=spike_times, d=spike_indices)

# d_lams = jnp.stack([jnp.dot(kernels, w_hat[i]) for i in range(n_neurons)])

basis_rc = nmo.basis.RaisedCosineBasisLog(n_fun, "eval")
time, basis_taus = basis_rc.evaluate_on_grid(ws)
lam_k = jnp.zeros(n_neurons)
for k in range(n_neurons):
    spikes_k = spikes_tsd.t[spikes_tsd.d == k].squeeze()
    for y_k in spikes_k:
        delta = nap.IntervalSet(y_k - ws_sec, y_k)
        S_delta = spikes_tsd.restrict(delta)
        for t_k, n in zip(S_delta.t, S_delta.d):
            tau = round(y_k - t_k)
            lam_k = lam_k.at[k].add(
                jnp.log(inverse_link_fun(jnp.dot(basis_taus[tau], w_hat[n, :, k])))
            )


t0 = perf_counter()
d_lams = jnp.stack([jnp.dot(kernels, w_hat[i]) for i in range(n_neurons)])
lam_k = jnp.zeros(n_neurons)
for k in range(n_neurons):
    spikes_k = spikes_tsd.t[spikes_tsd.d == k].squeeze()
    lam_k = lam_k.at[k].add(jax.jit(compute_lam_k)(spikes_k))
print(perf_counter() - t0)


#### simulated data

n_neurons = 10
n_spikes = 1000
n_fun = 10
T = 10  # sec
ws_sec = 0.004
binsize = 0.0001
ws = int(ws_sec / binsize)
inverse_link_fun = jax.nn.softplus

# simulate X
y_times = np.sort(T * np.random.random_sample(n_spikes))
n_indices = np.random.randint(0, n_neurons, n_spikes)
spikes_x = nap.Tsd(t=y_times, d=n_indices)

# generate some weights and basis functions
w = np.random.normal(0, 0.03, size=(n_neurons, n_fun, n_neurons))
basis_rc = nmo.basis.RaisedCosineBasisLog(n_fun, "eval")
time, basis_taus = basis_rc.evaluate_on_grid(ws)

# compute the fist logl term for each postsynaptic neuron
# for each postsynaptic neuron, sum_yk(log(softplus(sum_tk(w dot phi(yk-tk)))))


log_lambda_n = jnp.zeros(n_neurons)
for n in range(n_neurons):
    spikes_n = spikes_x.t[spikes_x.d == n].squeeze()
    for y_k in spikes_n:
        S_delta = spikes_x.restrict(nap.IntervalSet(y_k - ws_sec, y_k))
        y_k_sum = sum(
            jnp.dot(basis_taus[comp_tau(y_k, t_k)], w[i, :, n])
            for t_k, i in zip(S_delta.t, S_delta.d)
        )
        log_lambda_n = log_lambda_n.at[n].add(jnp.log(inverse_link_fun(y_k_sum)))


###########


log_lambda_n = jnp.zeros(n_neurons)
for n in range(n_neurons):
    spikes_n = y_times[n_indices == n]
    log_lambda_n = log_lambda_n.at[n].add(
        comp_lam_yk(jnp.zeros(1), spikes_n, y_times, n_indices, ws_sec)
    )


# spikes_quiet_sorted = np.load('/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/spikes_quiet_adj.npy', allow_pickle=True)
spikes_quiet_sorted = sio.loadmat(
    "/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/c57SpikeTimesQuiet.mat"
)["c57SpikeTimesQuiet"].squeeze()
# for i in range(195):
#     spikes_quiet_sorted[i] = spikes_quiet_sorted[i][spikes_quiet_sorted[i]<31-0.83586667]
# spikes_quiet_ex_jax = [jnp.array(arr) for arr in spikes_quiet_sorted]

ws_sec = 0.004
spikes_quiet_sorted = np.array(
    [
        spikes_quiet_sorted[k].squeeze()[spikes_quiet_sorted[k].squeeze() < 9108]
        for k in range(195)
    ],
    dtype=object,
)
merged_intervals, mapped_spikes = compute_intervals_and_mappings(
    spikes_quiet_sorted, ws_sec
)
np.save(
    "/Users/amedvedeva/Simons Foundation Dropbox/Arina Medvedeva/glm_songbirds/songbirds_data/merged_intervals.npy",
    np.array(merged_intervals),
)
print("Merged Intervals:", len(merged_intervals))
# mapped_spikes = map_spikes_to_intervals(spikes_quiet_ex_jax, merged_intervals)

int_lengths = np.array([end - start for start, end in merged_intervals])
spk_per_int = np.array(
    [sum(len(k) for k in mapped_spikes[interval]) for interval in merged_intervals]
)
neur_per_int = np.array(
    [sum(k.size > 0 for k in mapped_spikes[interval]) for interval in merged_intervals]
)
print(int_lengths.sum())
plt.hist(int_lengths, bins=1000)
plt.show()
# print("Spike mapping:", mapped_spikes)
