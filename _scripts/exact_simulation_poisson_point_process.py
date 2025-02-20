from copy import deepcopy
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from numba.np.npyfuncs import np_real_mod_impl
from scipy.integrate import simpson
from scipy.optimize import bisect
from jaxopt import ProjectedGradient, projection

"""
Simulate the simplest inhomogeneous poisson point process that is non-trivial:

1 - A single pre-synaptic neuron
2 - Fixed time window of time ws
3 - Spikes of pre-synaptic neuron spaced out every 2*ws
4 - Every spikes contributes to the rate with a rectangular contribution u(s) = 1 if 0 < s <= ws and 0 otherwise
    The intensity is then l(t) = l0 + sum_j u(t - 2 * ws * j)
5 - The cumulative intensity is L(t) = \int_0^t l(s) ds = l0 * t + ws * (t // (2 * ws)) + ws + relu(t - ws * (t // 20) - ws)
6 - The simulation follows Cinlar 1975, (the non-homogeneous process is a rescaled homogeneous one)
"""

def simulate_process(t_max, cumul_intensity, seed=123):
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


# params
l0 = 0.1
ws = 10
T = 5000

n_samples = 1000000
time = np.linspace(0, T, n_samples)

intensity = lambda t, l0, ws: l0 + ((t % (2 * ws)) > ws).astype(float)
cumulative_intensity = lambda t, l0, ws: l0*t + ws * (t // (2 * ws)) + jax.nn.relu(t - 2 * ws * (t // 20) - ws)



plt.plot(time, intensity(time, l0, ws))
plt.xlim(0,55)
plt.show()
approx = simpson(intensity(time, l0, ws), x=time)
true = cumulative_intensity(time[-1], l0, ws)
print(approx - true)

# simulate spikes
spike_times = simulate_process(T, lambda x: cumulative_intensity(x, l0, ws))
pre_synaptic = np.arange(ws, T, 2*ws).astype(float)

parametric_intensity = lambda t, params, ws: params[0] + params[1] * jnp.asarray((t % (2 * ws)) > ws, dtype=float)
parametric_cumul = lambda t, params, ws: params[0] * t + params[1] * (ws * (t // (2 * ws)) + jax.nn.relu(t - 2 * ws * (t // 20) - ws))

jit_parametric_intensity = jax.jit(parametric_intensity)
jit_parametric_cumul = jax.jit(parametric_cumul)

@jax.jit
def neg_log_likelihood(params, window_size, post_synaptic, tmax):
    log_lam = jnp.log(jit_parametric_intensity(post_synaptic, params, window_size))
    return  jit_parametric_cumul(tmax, params, window_size) / log_lam.shape[0]  - log_lam.mean()

p0 = jnp.asarray([1, 2], dtype=float)
pg = ProjectedGradient(fun=neg_log_likelihood, projection=projection.projection_box)
out = pg.run(p0, (1E-6, jnp.inf), window_size=ws, post_synaptic=spike_times, tmax=T)

print("recovered params", out[0])

# add std glm
import pynapple as nap
import nemos as nmo
bin_size = 0.001
count = nap.Ts(spike_times).count(bin_size).astype(float)
feature = count.value_from(nap.Tsd(time, intensity(time, l0, ws)))[:, None].astype(float)
feature = feature - feature[0,0] + 1 # make non-zero (log(feature  *w) in the poisson)

jax.config.update("jax_enable_x64", True)
# risky, no projection available but should work on fake data
model = nmo.glm.GLM(observation_model=nmo.observation_models.PoissonObservations(lambda x:x)).fit(feature, count)

pred_rate = model.predict(feature)/bin_size
plt.figure()
plt.plot(time, intensity(time, l0, ws), label="actual rate")
plt.plot(pred_rate, label="regular glm rate prediction")
plt.plot(time, jit_parametric_intensity(time, out[0], ws), label="point-process GLM")
plt.xlim(0, 50)
plt.ylim(0, 2)
plt.legend()
plt.show()