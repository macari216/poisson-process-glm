from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pynapple as nap

from nemos.regularizer import GroupLasso

from poisson_point_process import utils, simulate
from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt, GenLaguerreProdIntegral
from poisson_point_process.poisson_process_obs_model import MonteCarloApproximation, PolynomialApproximation
from poisson_point_process.poisson_process_glm import ContinuousMC, PopulationContinuousMC, PopulationContinuousPA

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score

jax.config.update("jax_enable_x64", True)

def rsn_w(key, p=0.15, neg=0.3, m=0., v=1., n=100, bf=10):
    n_conx = int(n * n * p)
    print(n_conx)
    n_neg = int(n_conx * neg)
    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)
    indices = jax.random.choice(subkey1, n * n, (n_conx,), replace=False)
    row_idx = indices // n
    col_idx = indices % n
    weights = jnp.abs(jax.random.normal(subkey2, (n_conx, bf)) * jnp.sqrt(v) + m)
    neg_idx = jax.random.choice(subkey3, n_conx, (n_neg,), replace=False)
    weights = weights.at[neg_idx].set(-weights[neg_idx])
    # Initialize W with zeros
    W = jnp.zeros((n, n, bf))
    W = W.at[row_idx, col_idx].set(weights)
    self_coupling = -1 * jnp.abs(jax.random.normal(subkey4, (n, bf)) * jnp.sqrt(v) + m)
    W = W.at[jnp.arange(n), jnp.arange(n)].set(self_coupling)
    return W

def compute_roc_curve(true_conn, filters):
    abs_argmax = np.argmax(np.abs(filters), axis=1)
    peak_filt = np.array(np.take_along_axis(filters, abs_argmax[:, np.newaxis], axis=1).squeeze(axis=1))
    scores = np.abs(peak_filt)
    fpr, tpr, roc_thresh = roc_curve(true_conn, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(true_conn, scores)

    f1s = []
    for t in roc_thresh:
        preds = (scores >= t).astype(int)
        f1s.append(f1_score(true_conn, preds))
    best_t = roc_thresh[np.argmax(f1s)]
    pred_conn = (scores >= best_t).astype(int)

    return fpr, tpr, roc_auc, ap, pred_conn


n_neurons = 25
target_n = 2
sim_time = 1000
binsize = 0.0001
history_window = 0.005
window_size = int(history_window/binsize)
n_basis_funcs = 4
c = 1.3
basis_fn = GenLaguerreEval(history_window, n_basis_funcs, c=c)
int_fn = GenLaguerreInt(history_window, n_basis_funcs, c=c)
time = jnp.linspace(0,history_window,window_size)
kernels = basis_fn(-time)
phi = jnp.exp
phi_inverse = jnp.log
# phi = jax.nn.softplus
# phi_inverse = lambda x: jnp.log(jnp.exp(x) - 1)
recording_time = nap.IntervalSet(0, sim_time)

weights_true = rsn_w(jax.random.PRNGKey(0), p=0.08, neg=0.25, m=0, v=0.0017, n=n_neurons, bf=1).squeeze()
true_conn = (weights_true[target_n]!=0).astype(int)

spikes, act = simulate.sim_real_jax(sim_time, binsize, n_neurons, weights_true, thres=0.03, b=0.011, b_std=0.6, rise=0.0015, fall=0.002, cond=1)

spike_times, spike_ids = simulate.poisson_times(spikes, sim_time, binsize)
X_spikes = jnp.vstack((spike_times,spike_ids))
y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))
y_spikes = y_spikes[:,y_spikes[1]==target_n]
y_spikes = y_spikes.at[1].set(jnp.zeros(y_spikes.shape[1]))

obs_model_mc = MonteCarloApproximation(
    n_basis_funcs=n_basis_funcs,
    n_batches_scan=1,
    history_window=history_window,
    mc_n_samples=int(1e4),
    eval_function=basis_fn,
    inverse_link_function=phi,
)

# unregularized
model_mc = ContinuousMC(
    solver_name="GradientDescent",
    observation_model=obs_model_mc,
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 1000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 100 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')

weights_mc, bias_mc = model_mc.coef_.reshape(n_neurons, -1), model_mc.intercept_
filters_mc_raw = jnp.dot(weights_mc, kernels.T)
filters_mc = phi(filters_mc_raw + bias_mc)

fpr_unreg, tpr_unreg, roc_auc_unreg, ap_unreg, pred_conn_unreg = compute_roc_curve(true_conn, filters_mc_raw)

# ridge
model_mc = ContinuousMC(
    solver_name="GradientDescent",
    observation_model=obs_model_mc,
    regularizer="Ridge",
    regularizer_strength=100,
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 1000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 100 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')

weights_mc, bias_mc = model_mc.coef_.reshape(n_neurons, -1), model_mc.intercept_
filters_mc_raw = jnp.dot(weights_mc, kernels.T)
filters_mc = phi(filters_mc_raw + bias_mc)

fpr_ridge, tpr_ridge, roc_auc_ridge, ap_ridge, pred_conn_ridge = compute_roc_curve(true_conn, filters_mc_raw)

# group lasso
n_groups = n_neurons
n_features = n_groups * n_basis_funcs
mask = np.zeros((n_groups, n_features))
for i in range(n_groups):
    mask[i, i * n_basis_funcs:i * n_basis_funcs + n_basis_funcs] = np.ones(n_basis_funcs)

model_mc = ContinuousMC(
    solver_name="ProximalGradient",
    observation_model=obs_model_mc,
    regularizer=GroupLasso(mask=mask),
    regularizer_strength=100,
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 1000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 100 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')

weights_mc, bias_mc = model_mc.coef_.reshape(n_neurons, -1), model_mc.intercept_
filters_mc_raw = jnp.dot(weights_mc, kernels.T)
filters_mc = phi(filters_mc_raw + bias_mc)

frac_zero = weights_mc.sum(1)[weights_mc.sum(1) == 0].size / weights_mc.sum(1).size
print(f"fraction set to 0: {frac_zero}, true: {1 - (true_conn.sum() / true_conn.size)}")
pred_conn_gl = (weights_mc.sum(1)!=0).astype(int)

fpr_gl, tpr_gl, roc_auc_gl, ap_gl, _ = compute_roc_curve(true_conn, filters_mc_raw)

# plot roc curve
plt.figure(figsize=(7,5))
plt.plot(fpr_unreg, tpr_unreg, label=f"UnReg (AUC = {roc_auc_unreg:.2f})")
plt.plot(fpr_ridge, tpr_ridge, label=f"Ridge (AUC = {roc_auc_ridge:.2f})")
plt.plot(fpr_gl, tpr_gl, label=f"GL (AUC = {roc_auc_gl:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("false positives")
plt.ylabel("true positives")
plt.legend()

print(f"average precision: unreg {ap_unreg}, ridge {ap_ridge}, gl {ap_gl}")

# plot true vs predicted connections
mat = np.vstack([true_conn, pred_conn_unreg, pred_conn_ridge, pred_conn_gl])
binary_cmap = ListedColormap(['white', 'black'])
plt.figure(figsize=(7,2))
plt.imshow(mat, aspect="auto", cmap=binary_cmap)
plt.yticks([0,1], ["true", "predicted UnReg", "predicted Ridge", "predicted GL"])
plt.xticks(np.arange(mat.shape[1]), np.arange(mat.shape[1]))
plt.xlabel("presynaptic neuron")
plt.tight_layout()

plt.show()