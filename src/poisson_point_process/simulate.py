from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import nemos as nmo
from scipy.optimize import bisect

from nemos import validation
from nemos.pytrees import FeaturePytree

def poisson_counts(
        pres_rate_per_bin, bias_posts, n_bins_tot, n_pres, weights_true,
                   window_size, basis_kernels, phi, seed=216,
):
    n_bins_tot += window_size

    np.random.seed(seed)

    weights_true = jnp.array(weights_true)
    bias_posts = jnp.array(bias_posts)

    pres_spikes = jnp.array(np.random.poisson(lam=pres_rate_per_bin, size=(n_bins_tot, n_pres)))

    X = nmo.convolve.create_convolutional_predictor(basis_kernels, jnp.array(pres_spikes)).reshape(n_bins_tot, -1)
    X = X[window_size:]
    lam_posts = phi(np.dot(X, weights_true) + bias_posts)
    posts_spikes = jnp.array(np.random.poisson(lam=lam_posts, size=len(lam_posts)))

    return X, posts_spikes, jnp.array(pres_spikes)[window_size:], lam_posts

def simulate_recurrent(
    coupling_coef: NDArray,
    feedforward_coef: NDArray,
    intercepts: NDArray,
    random_key: jax.Array,
    feedforward_input: Union[NDArray, jnp.ndarray],
    coupling_basis_matrix: Union[NDArray, jnp.ndarray],
    init_y: Union[NDArray, jnp.ndarray],
    inverse_link_function: Callable = jax.nn.softplus,
):
    """
    Simulate neural activity using the GLM as a recurrent network.

    This function projects neural activity into the future, employing the fitted
    parameters of the GLM. It is capable of simulating activity based on a combination
    of historical activity and external feedforward inputs like convolved currents, light
    intensities, etc.

    Parameters
    ----------
    coupling_coef :
        Coefficients for the coupling (recurrent connections) between neurons.
        Expected shape: (n_neurons (receiver), n_neurons (sender), n_basis_coupling).
    feedforward_coef :
        Coefficients for the feedforward inputs to each neuron.
        Expected shape: ``(n_neurons, n_basis_input)``.
    intercepts :
        Bias term for each neuron. Expected shape: ``(n_neurons,)``.
    random_key :
        jax.random.key for seeding the simulation.
    feedforward_input :
        External input matrix to the model, representing factors like convolved currents,
        light intensities, etc. When not provided, the simulation is done with coupling-only.
        Expected shape: ``(n_time_bins, n_neurons, n_basis_input)``.
    init_y :
        Initial observation (spike counts for PoissonGLM) matrix that kickstarts the simulation.
        Expected shape: ``(window_size, n_neurons)``.
    coupling_basis_matrix :
        Basis matrix for coupling, representing between-neuron couplings
        and auto-correlations. Expected shape: ``(window_size, n_basis_coupling)``.
    inverse_link_function :
        The inverse link function for the observation model.

    Returns
    -------
    simulated_activity :
        Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
        Shape, ``(n_time_bins, n_neurons)``.
    firing_rates :
        Simulated rates for each neuron over time. Shape, ``(n_time_bins, n_neurons,)``.

    Raises
    ------
    ValueError
        If there's an inconsistency between the number of neurons in model parameters.
    ValueError
        If the number of neurons in input arguments doesn't match with model parameters.

    Examples
    --------
    .. plot::
        :include-source: True
        :caption: Recurrently connected GLM simulations.

        >>> import numpy as np
        >>> import jax
        >>> import matplotlib.pyplot as plt
        >>> from nemos.simulation import simulate_recurrent
        >>> np.random.seed(42)
        >>> n_neurons = 2
        >>> coupling_duration = 100
        >>> feedforward_input = np.random.normal(size=(1000, n_neurons, 1))
        >>> coupling_basis = np.random.normal(size=(coupling_duration, 10))
        >>> coupling_coef = 0.5*np.random.normal(size=(n_neurons, n_neurons, 10))
        >>> intercept = -9 * np.ones(n_neurons)
        >>> init_spikes = np.zeros((coupling_duration, n_neurons))
        >>> random_key = jax.random.key(123)
        >>> spikes, rates = simulate_recurrent(
        ...     coupling_coef=coupling_coef,
        ...     feedforward_coef=np.ones((n_neurons, 1)),
        ...     intercepts=intercept,
        ...     random_key=random_key,
        ...     feedforward_input=feedforward_input,
        ...     coupling_basis_matrix=coupling_basis,
        ...     init_y=init_spikes
        ... )
        >>> _ = plt.figure()
        >>> _ = plt.plot(rates[:, 0], label="Neuron 0 rate")
        >>> _ = plt.plot(rates[:, 1], label="Neuron 1 rate")
        >>> _ = plt.legend()
        >>> _ = plt.title("Simulated firing rates")
        >>> _ = plt.show()
    """
    if isinstance(feedforward_input, FeaturePytree):
        raise ValueError(
            "simulate_recurrent works only with arrays. "
            "FeaturePytree provided instead!"
        )
    # convert to jnp.ndarray of floats
    coupling_basis_matrix = jnp.asarray(coupling_basis_matrix, dtype=float)
    coupling_coef = jnp.asarray(coupling_coef, dtype=float)
    feedforward_coef = jnp.asarray(feedforward_coef, dtype=float)
    intercepts = jnp.asarray(intercepts, dtype=float)
    feedforward_input = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=float), feedforward_input
    )
    init_y = jnp.asarray(init_y, dtype=float)

    # check that n_neurons is consistent
    n_neurons = intercepts.shape[0]
    if (
        feedforward_input.shape[1] != n_neurons
        or feedforward_coef.shape[0] != n_neurons
        or init_y.shape[1] != n_neurons
        or coupling_coef.shape[0] != n_neurons
        or coupling_coef.shape[1] != n_neurons
    ):
        raise ValueError(
            "The number of neurons provided in the inputs is inconsistent!"
        )

    # checks the input size
    validation.check_tree_leaves_dimensionality(
        feedforward_input,
        expected_dim=3,
        err_message="`feedforward_input` must be three-dimensional, with shape "
        "(n_timebins, n_neurons, n_features) or pytree of the same shape.",
    )
    validation.check_tree_axis_consistency(
        feedforward_coef,
        feedforward_input,
        axis_1=1,
        axis_2=2,
        err_message="Inconsistent number of features. "
        f"spike basis coefficients has {jax.tree_util.tree_map(lambda p: p.shape[0], feedforward_coef)} features, "
        f"X has {jax.tree_util.tree_map(lambda x: x.shape[2], feedforward_input)} features instead!",
    )

    validation.error_invalid_entry(feedforward_input)

    # validate y
    validation.check_tree_leaves_dimensionality(
        init_y,
        expected_dim=2,
        err_message="`init_y` must be two-dimensional, with shape (n_timebins, ).",
    )
    n_basis = coupling_coef.shape[-1]
    coupling_coef = coupling_coef.reshape(n_neurons, -1)

    if coupling_basis_matrix.shape[1] * n_neurons != coupling_coef.shape[1]:
        raise ValueError(
            f"Inconsistent number of features. `coupling_basis_matrix` assumes "
            f"{coupling_basis_matrix.shape[1]} basis functions for the coupling filters, "
            f"`coupling_coef` assumes {n_basis} basis functions instead."
        )

    if init_y.shape[0] != coupling_basis_matrix.shape[0]:
        raise ValueError(
            "`init_y` and `coupling_basis_matrix`"
            " should have the same window size! "
            f"`init_y` window size: {init_y.shape[0]}, "
            f"`coupling_basis_matrix` window size: {coupling_basis_matrix.shape[0]}"
        )

    subkeys = jax.random.split(random_key, num=feedforward_input.shape[0])
    # Pre-compute feedforward contribution: (n_samples, n_neurons)
    feed_forward_contrib = jnp.einsum("ik,tik->ti", feedforward_coef, feedforward_input)

    # Pre-flip the basis to match convolution behavior (jnp.convolve flips the kernel)
    coupling_basis_flipped = coupling_basis_matrix[::-1]

    def scan_fn(
        activity: jnp.ndarray, inputs: Tuple[jax.Array, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Optimized scan over time steps.

        Improvements over original:
        - Direct iteration over feedforward input (no dynamic_slice)
        - Simple einsum for convolution (no nested scan)
        """
        key, ff_input = inputs

        # Simple einsum convolution: activity (window, n_neurons) @ basis_flipped (window, n_basis)
        # Flipping matches jnp.convolve behavior used in _tensor_convolve
        # Result: (n_neurons, n_basis) -> flattened to (n_neurons * n_basis,)
        conv_act = jnp.einsum("wn,wb->nb", activity, coupling_basis_flipped).reshape(-1)

        # Predict firing rate
        firing_rate = inverse_link_function(
            coupling_coef.dot(conv_act) + ff_input + intercepts
        )

        # Simulate activity
        new_act = jax.random.poisson(key, firing_rate)

        activity = jnp.vstack((activity[1:], new_act))

        return activity, (new_act, firing_rate)

    # Iterate over (subkeys, feed_forward_contrib) together
    _, outputs = jax.lax.scan(scan_fn, init_y, (subkeys, feed_forward_contrib))
    simulated_activity, firing_rates = outputs
    return simulated_activity, firing_rates

def poisson_counts_recurrent(n_bins_tot, n_neurons, window_size, basis_kernels, params, inv_link,
                             feedforward_input=None, feedforward_coef=None, init_spikes=None, seed=123):
    # parameters for simulator
    if feedforward_input is None:
        feedforward_input = np.zeros((n_bins_tot, n_neurons, 1))
        feedforward_coef = np.zeros((n_neurons, 1))
    if init_spikes is None:
        init_spikes = np.zeros((window_size, n_neurons))
    random_key = jax.random.key(seed)
    coefs, intercepts = params

    # generate poisson firing rates per bin
    spikes, firing_rates = simulate_recurrent(coupling_coef = coefs,
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

def sim_real_jax(sim_time, binsize, n, W, key=jax.random.PRNGKey(0), b=0.011, b_std=0.6, thres=0.03, rise=0.0015,
                 fall=0.002, ref=0.001, cond=1):
    """
    Simulate a recurrent conductance-based threshold spiking network with alpha function synapses
    Each neuron:
    - receives random external input, scaled by gaussian noise and a bernoulli mask
    - receives recurrent input from other neurons
    - spikes if input > threshold
    - opens a synaptic channel for a fixed duration
    - updates activity with decay, producing an alpha function response.
    Parameters
    ----------
    sim_time : float
        total simulation time (seconds)
    binsize : float
        time step size
    n : int
        number of neurons
    W : shape (n, n)
        recurrent weight matrix
    key : PRNGKey
        JAX random key
    b, b_std, thres, rise, fall, cond : float
       model parameters controlling external input, synaptic dynamics, and channel conductance
    Returns
    -------
    spikes : (t_steps, n)
        binary spike train
    act : (t_steps, n)
        synaptic activity
    """
    def scan_fn(carry, t):
        act, ch_counter, ref_counter, key = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # input from external and synaptic activity
        xi = jax.random.normal(subkey1, (n,)) * b_std
        b_t = (b * (1 + xi)) * jax.random.bernoulli(subkey2, 0.07, (n,))
        I_t = W @ act.T + b_t

        # spike and open synaptic channel
        ref_counter = jnp.maximum(ref_counter - 1, 0)
        spikes = jnp.where((I_t > thres) & (ref_counter==0), 1., 0.)
        ref_counter = jnp.where(spikes > 0, r, ref_counter)
        ch_counter = jnp.where(spikes > 0, o, ch_counter)
        g = jnp.where(ch_counter > 0, cond, 0)
        ch_counter = jnp.maximum(ch_counter - 1, 0)

        # update activity
        act_next = act - ((act + g * (act - 1)) * binsize / fall)
        return (act_next, ch_counter, ref_counter, key), (spikes, act_next)

    t_steps = int(sim_time / binsize)
    o = int(rise / binsize)
    r = int(ref/ binsize)
    init_arrays = (jnp.zeros(n, ), jnp.zeros(n, ), jnp.zeros(n,), key)

    _, (spikes, act) = jax.lax.scan(scan_fn, init_arrays, jnp.arange(t_steps))

    return spikes, act


def sim_real(tot_time, binsize, n, W, thres=0.03, b=0.001, b_std=0.6, rise=0.0015, fall=0.002, ref=0.001, cond=1):
    """
    NumPy version
    """
    t = int(tot_time / binsize)
    o = int(rise / binsize)  # interval for opening synaptic channels
    r = int(ref / binsize)  # absolute refractory period
    # initialize synaptic activity, spikes, and channel conductances
    # s0 = np.abs(np.random.normal(0, 1, n))
    s0 = np.zeros(n)
    s = np.concatenate((s0[None, :], np.zeros((t, n))), axis=0)
    spikes_ring = np.zeros((t, n))
    # g = np.zeros((t, n))
    ch_counter = np.zeros(n)
    ref_counter = np.zeros(n)
    for t in range(t):
        # external input
        xi = np.random.normal(0, b_std, n)
        b_t = (b * (1 + xi)) * np.random.binomial(1, 0.07, n)
        # print(b_t.sum())
        # summed input from synaptic and external activity
        I_t = (W @ s[t].T) + b_t
        ref_counter = jnp.maximum(ref_counter - 1, 0)
        thres_mask = (I_t > thres) & (ref_counter==0)
        spikes_ring[t, thres_mask] = 1
        ch_counter = jnp.where(spikes_ring[t] > 0, o, ch_counter)
        ref_counter = jnp.where(spikes_ring[t] > 0, r, ref_counter)
        # print(ch_counter.sum())
        g_t = jnp.where(ch_counter > 0, 1, 0)
        ch_counter = jnp.maximum(ch_counter - 1, 0)
        # g[t:t + o, thres_mask] = 1
        s[t + 1] = s[t] - ((s[t] + cond * g_t * (s[t] - 1)) * binsize / fall)
    return spikes_ring, s[1:]