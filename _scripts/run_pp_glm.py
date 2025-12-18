import os

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import pynapple as nap

from poisson_point_process import simulate
from poisson_point_process.utils import quadratic
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral
from poisson_point_process.poisson_process_glm import ContinuousPA, ContinuousMC
from poisson_point_process.poisson_process_obs_model import PolynomialApproximation, MonteCarloApproximation

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

#### EXAMPLE CODE FOR FITTING A SINGLE POSTSYNAPTIC NEURON
#### SIMULATION SIZE AND OPTIMIZATION PARAMS ARE CHOSEN TO BE FEASIBLE ON CPU

# jax will prefer GPU if available
print("JAX is using:", jax.default_backend())

## generate data from all-to-one coupled GLM
n_neurons = 8
sim_time = 500
history_window = 0.005
# required for simulation
binsize = 0.00005
n_bins_tot = int(sim_time/binsize)
window_size = int(history_window/binsize)

# create a set of basis functions
n_basis_funcs = 4
basis_fn = GenLaguerreEval(history_window, n_basis_funcs)
time = jnp.linspace(0,history_window,100)
kernels = basis_fn(-time)
int_fn = GenLaguerreInt(history_window, n_basis_funcs)
prod_int_fn = GenLaguerreProdIntegral(history_window, n_basis_funcs)

# set inverse link function Phi
phi = jnp.exp
phi_inverse = jnp.log

# simulate data
print("simulating spike data...")

np.random.seed(216)

#set generative model parameters
# baseline firing rate in Hz
pres_rate_hz = 5
posts_rate_hz = 2

# inverse firing rate per bin
bias_true = phi_inverse(posts_rate_hz * binsize)
# generative weights
weights_true = np.random.normal(-0.2, 0.5, n_neurons * n_basis_funcs)

# true filters in Hz
filters_true = phi(np.dot(weights_true.reshape(-1,n_basis_funcs), kernels.T) + bias_true) / binsize

# step 1: simulate counts
# post- and presynaptic spike counts and postsynaptic firing rates per bin
_, y_counts, X_counts, rates = simulate.poisson_counts(pres_rate_hz, bias_true, binsize,
                                                           n_bins_tot, n_neurons, weights_true, window_size, kernels,
                                                           phi)

# step 2: convert to spike times
spike_times, spike_ids = simulate.poisson_times(X_counts, sim_time, binsize)
spike_times_y, _ = simulate.poisson_times(y_counts[:, None], sim_time, binsize)

# X_spikes contains all presynaptic spike times and corresponding neuron IDs
X_spikes = jnp.vstack((spike_times, spike_ids))

# y_spikes contains postsynaptic spike times, IDs, and their insertion indices into
# the presynaptic spike times array, preserving temporal order (required to perform a scan over spikes)
# In the single postsynaptic neuron case, neuron ID is always set to 0
target_idx = jnp.searchsorted(X_spikes[0], spike_times_y)
y_spikes = jnp.vstack((spike_times_y, jnp.zeros(target_idx.size), target_idx))

## fit continuous PA model
print("fitting PA-c model...")

# set approximation range based on binned firing rates
approx_interval = [
    np.percentile(phi_inverse(rates/binsize), 2.5),
    np.percentile(phi_inverse(rates/binsize), 97.5)
]

# initialize PA-c observation model (computes sufficient statistics)
# Here we apply Ridge regularization to reduce approximation error
obs_model_pa = PolynomialApproximation(
    n_basis_funcs=n_basis_funcs,
    n_batches_scan=1,
    n_batches_pa=1,
    history_window=history_window,
    eval_function=basis_fn,
    int_function = int_fn,
    prod_int_function= prod_int_fn,
)

# initialize and fit PA-c model (closed form solution)
tt0 = perf_counter()
model_pa = ContinuousPA(
    regularizer="Ridge",
    regularizer_strength = 100,
    observation_model=obs_model_pa,
    inverse_link_function=phi,
    approx_interval=approx_interval,
    recording_time=nap.IntervalSet(0, sim_time),
    reset_suff_stats=True,
    solver_kwargs={"tol":1e-12}
).fit_closed_form(X_spikes, y_spikes)
time_pa = perf_counter() - tt0
print(f"PA-c fit time: {time_pa}")

# construct estimated filters
# to avoid model mismatch, we use the approximated quadratic nonlinearity to construct the filters
weights_pa, bias_pa = model_pa.coef_.reshape(-1,n_basis_funcs), model_pa.intercept_
filters_pa = quadratic(np.dot(weights_pa, kernels.T)  + bias_pa, phi, approx_interval)

#compute MSE
mse_pa = np.mean((filters_true - filters_pa) ** 2)


## fit continuous MC model
print("fitting MC model...")

# initialize MC observation model (computes nll)
# draw 500,000 MC samples per gradient step
obs_model_mc = MonteCarloApproximation(
    n_basis_funcs=n_basis_funcs,
    n_batches_scan=1,
    history_window=history_window,
    M_samples=int(5e5),
    eval_function=basis_fn,
    int_function=int_fn,
    control_var=True,
)

# initialize MC model
model_mc = ContinuousMC(
    solver_name="ProxStochasticAdamRoP",
    observation_model=obs_model_mc,
    recording_time=nap.IntervalSet(0, sim_time),
    inverse_link_function=phi,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"stepsize": 1e-4}
    )

tt0 = perf_counter()
model_mc.fit(X_spikes, y_spikes)
time_mc = perf_counter() - tt0
print(f"MC fit time: {time_mc}")

# construct estimated filters
weights_mc, bias_mc = model_mc.coef_.reshape(-1,n_basis_funcs), model_mc.intercept_
filters_mc = phi(np.dot(weights_mc, kernels.T) + bias_mc)
#compute MSE
mse_mc = np.mean((filters_true - filters_mc) ** 2)


## fit hybrid PA-MC model
print("fitting hybrid model...")
# initialize hybrid model
# models parameters are initializes at the PA estimate
model_h = ContinuousMC(
    solver_name="StochasticAdamRoP",
    observation_model=obs_model_mc,
    inverse_link_function=phi,
    recording_time=nap.IntervalSet(0, sim_time),
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"stepsize": 1e-4}
)

tt0 = perf_counter()
pa_params = (model_pa.coef_.squeeze(), jnp.atleast_1d(model_pa.intercept_))
model_h.fit(X_spikes, y_spikes, init_params=pa_params)
time_h = perf_counter() - tt0
print(f"Hybrid fit time: {time_h}")

# construct estimated filters
weights_h, bias_h = model_h.coef_.reshape(-1,n_basis_funcs), model_h.intercept_
filters_h = phi(np.dot(weights_h, kernels.T) + bias_h)
#compute MSE
mse_h = np.mean((filters_true - filters_h) ** 2)

# print scores
print()
print(f"MSE against true filters: \nPa-c {mse_pa} \nMC {mse_mc} \nHybrid {mse_h}")
print()

# save results
results = {
    "pa-c": {
        "fit_time": time_pa,
        "filters": filters_pa,
        "mse": mse_pa,
    },
    "monte-carlo": {
        "fit_time": time_mc,
        "filters": filters_mc,
        "mse": mse_mc,
    },
    "hybrid": {
        "fit_time": time_h,
        "filters": filters_h,
        "mse": mse_h,
    },
    "true": {
        "filters": filters_true,
        "basis_kernels": kernels,
        "weights": weights_true,
        "bias": bias_true,
    }
}

save_path = "../_results/pp_glm_results.npz"
os.makedirs("../_results", exist_ok=True)
np.savez(save_path, **results)
print("results saved to _results/pp_glm_results.npz")

print("script terminated")