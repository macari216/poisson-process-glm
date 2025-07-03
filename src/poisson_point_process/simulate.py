import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import nemos as nmo
from scipy.optimize import bisect

def poisson_counts(mean_per_sec, bias_posts, binsize, n_bins_tot, n_pres, weights_true, ws, basis, nonlin, seed=216):
    n_bins_tot += ws

    np.random.seed(seed)
    lam_pres = np.abs(np.random.normal(mean_per_sec, mean_per_sec/10, n_pres))

    weights_true = jnp.array(weights_true)
    bias_posts = jnp.array(bias_posts)

    rate_per_bin = lam_pres * binsize
    pres_spikes = jnp.array(np.random.poisson(lam=rate_per_bin, size=(n_bins_tot, n_pres)))

    X = nmo.convolve.create_convolutional_predictor(basis, jnp.array(pres_spikes)).reshape(n_bins_tot, -1)
    X = X[ws:]
    lam_posts = nonlin(np.dot(X, weights_true) + bias_posts)
    posts_spikes = jnp.array(np.random.poisson(lam=lam_posts, size=len(lam_posts)))

    return X, posts_spikes, jnp.array(pres_spikes)[ws:], lam_posts

def poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, basis_kernels, params, inv_link, init_spikes=None, seed=123):
    # parameters for simulator
    feedforward_input = np.zeros((n_bins_tot, n_neurons, 1))
    feedforward_coef = np.zeros((n_neurons, 1))
    if init_spikes is None:
        init_spikes = np.zeros((window_size, n_neurons))
    random_key = jax.random.key(seed)
    coefs, intercepts = params

    # generate poisson firing rates per bin
    spikes, firing_rates = nmo.simulation.simulate_recurrent(coupling_coef = coefs,
                                                 feedforward_coef = feedforward_coef,
                                                 intercepts = intercepts,
                                                 random_key = random_key,
                                                 feedforward_input = feedforward_input,
                                                 init_y = init_spikes,
                                                 coupling_basis_matrix = basis_kernels,
                                                 inverse_link_function = inv_link

    )

    return spikes, firing_rates

def poisson_times(counts, tot_time_sec, binsize, random_key=jax.random.PRNGKey(0)):
    """generate poisson process spike times
    since the counts are provided, we assume spike times
    are uniformly distributed within bins (memoryless property of poisson process)"""

    n_bins_tot, n_neurons = counts.shape
    bin_starts = jnp.linspace(binsize, tot_time_sec, int(tot_time_sec / binsize)) - binsize/2

    repeated_bins = jnp.repeat(bin_starts, counts.sum(1).astype(int))
    random_offsets = jax.random.uniform(
        random_key,
        shape=(repeated_bins.size,),
        minval=0,
        maxval=binsize,
    )

    spike_times = repeated_bins + random_offsets
    neuron_ids = jnp.tile(jnp.arange(n_neurons), n_bins_tot)
    neuron_indices = jnp.repeat(neuron_ids, counts.flatten().astype(int))

    #sort by time
    sorted_indices = jnp.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    neuron_indices = neuron_indices[sorted_indices]

    return spike_times, neuron_indices


def inhomogeneous_process(t_max, b, w, ws, seed=123):
    """
    Exact simulation of inhomogeneous Poisson Point process
    by temporal re-scaling method. Cinlar 1975.
    """
    # intensity = lambda t, l0, w, ws: l0 + w * ((t % (2 * ws)) > ws).astype(float)
    cumulative_intensity = lambda t, l0, w, ws: l0 * t + w * (ws * (t // (2 * ws)) + jax.nn.relu((t % (2 * ws)) - ws))
    cumul_intensity = lambda x: cumulative_intensity(x, b, w, ws)
    np.random.seed(seed)
    spike_times = []
    s = 0
    step_for_bisect = t_max / 100.
    t0 = 0
    upper = step_for_bisect
    while t0 < t_max:
        uni = np.random.uniform()
        s = s - np.log(uni)
        # very ugly way to be over the optimum
        # but bisect is very fast and accurate.
        while cumul_intensity(upper) < s:
            upper = upper + step_for_bisect
        spike_times.append(
            bisect(lambda x: cumul_intensity(x) - s, upper - step_for_bisect, upper, xtol=10**-14)
        )
        upper = upper - step_for_bisect
        t0 = spike_times[-1]
    spike_times = np.array(spike_times)
    return spike_times[spike_times < t_max]