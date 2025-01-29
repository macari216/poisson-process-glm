import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import nemos as nmo
from time import perf_counter

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

    return weights_true, X, pres_spikes, y

def generate_poisson_counts_recurrent(tot_time_sec, binsize, n_neurons, basis_kernels, params, init_spikes, inv_link):
    # parameters for simulator
    n_bins_tot = int(tot_time_sec / binsize)
    feedforward_input = np.zeros((n_bins_tot, n_neurons, 1))
    feedforward_coef = np.zeros((n_neurons, 1))
    random_key = jax.random.key(123)
    coefs, intercepts = params

    # generate poisson firing rates per bin
    spikes, _ = nmo.simulation.simulate_recurrent(coupling_coef = coefs,
                                                 feedforward_coef = feedforward_coef,
                                                 intercepts = intercepts,
                                                 random_key = random_key,
                                                 feedforward_input = feedforward_input,
                                                 init_y = init_spikes,
                                                 coupling_basis_matrix = basis_kernels,
                                                 inverse_link_function = inv_link

    )

    return spikes

def generate_poisson_times(counts, tot_time_sec, binsize, random_key=jax.random.PRNGKey(0)):
    """generate poisson process spike times
    since the counts are provided, we assume spike times
    are uniformly distributed within bins (memoryless property of poisson process)"""

    n_bins_tot, n_neurons = counts.shape
    bin_starts = jnp.linspace(0, tot_time_sec, n_bins_tot, endpoint=False)

    repeated_bins = jnp.repeat(bin_starts, counts.sum(1))
    random_offsets = jax.random.uniform(
        random_key,
        shape=(repeated_bins.size,),
        minval=0,
        maxval=binsize,
    )

    spike_times = repeated_bins + random_offsets
    neuron_indices = jnp.repeat(jnp.arange(n_neurons), counts.sum(0))

    #sort by time
    sorted_indices = jnp.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    neuron_indices = neuron_indices[sorted_indices]

    return spike_times, neuron_indices