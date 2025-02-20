from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
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
    step_for_bisect = 10
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
plt.show()
approx = simpson(intensity(time, l0, ws), x=time)
true = cumulative_intensity(time[-1], l0, ws)
print(approx - true)

# simulate spikes
spike_times = simulate_process(5000, lambda x: cumulative_intensity(x, l0, ws))
pre_synaptic = np.arange(2*ws, 5000, 2*ws).astype(float)

parametric_intensity = lambda t, params, ws: params[0] + params[1] * jnp.asarray((t % (2 * ws)) > ws, dtype=float)
parametric_cumul = lambda t, params, ws: params[0] * t + params[1] * (ws * (t // (2 * ws)) + jax.nn.relu(t - 2 * ws * (t // 20) - ws))

jit_parametric_intensity = jax.jit(parametric_intensity)
jit_parametric_cumul = jax.jit(parametric_cumul)

@jax.jit
def log_likelihood(params, window_size, post_synaptic, tmax):
    log_lam = jnp.log(jit_parametric_intensity(post_synaptic, params, window_size))
    return  jit_parametric_cumul(tmax, params, window_size) - log_lam.sum()

p0 = jnp.asarray([0.05, 2])
pg = ProjectedGradient(fun=log_likelihood, projection=projection.projection_box)
pg.run(p0, (1E-6, jnp.inf), window_size=ws, post_synaptic=spike_times, tmax=T)







