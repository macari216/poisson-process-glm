import jax.numpy as jnp
import numpy as np
import pynapple as nap

def generate_spike_times_uniform(tot_time_sec, tot_spikes_n, n_neurons, seed=216):
    np.random.seed(seed)
    tot_spikes = np.sort(np.random.uniform(0, tot_time_sec, size=tot_spikes_n))
    neuron_ids = np.random.choice(n_neurons, size=len(tot_spikes))
    spike_dict = {key: nap.Ts(tot_spikes[np.arange(len(tot_spikes))[neuron_ids == key]]) for key in range(n_neurons)}
    spikes_tsgroup = nap.TsGroup(spike_dict, nap.IntervalSet(0, tot_time_sec))

    return spikes_tsgroup

def times_to_xy(spikes, basis, posts_n, binsize, window_size):
    y = jnp.array(spikes.count(binsize)).squeeze()
    X = basis.compute_features(y)
    y, X = y[window_size:], X[window_size:]
    y = y[:, posts_n]

    return X, y

def generate_poisson_counts(mean_per_sec, binsize, n_bins_tot, n_pres, n_basis_funcs, ws, basis, nonlin, seed=216):
    np.random.seed(seed)
    lam_pres = np.abs(np.random.normal(mean_per_sec, mean_per_sec/10, n_pres))

    rate_per_bin = lam_pres * binsize
    weights_true = np.random.normal(0, 1, n_pres * n_basis_funcs)
    pres_spikes = jnp.array(np.random.poisson(lam=rate_per_bin, size=(n_bins_tot, n_pres)))
    X = basis.compute_features(pres_spikes)
    X = X[ws:]
    lam_posts = nonlin(np.dot(X, weights_true))
    y = jnp.array(np.random.poisson(lam=lam_posts, size=len(lam_posts)))

    return weights_true, X, y