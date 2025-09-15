import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np

import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

import pynapple as nap

from nemos.regularizer import GroupLasso

from poisson_point_process.basis import GenLaguerreEval, GenLaguerreInt
from poisson_point_process.poisson_process_obs_model import MonteCarloApproximation
from poisson_point_process.poisson_process_glm import PopulationContinuousMC

from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score

jax.config.update("jax_enable_x64", True)
os.environ["JAX_PLATFORM_NAME"] = "gpu"

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

def preprocess_spikes(spikes, n_neurons):
    spikes_n = jnp.array([spikes[n].shape[0] for n in range(n_neurons)])
    spike_ids = jnp.repeat(jnp.arange(n_neurons), spikes_n)
    spike_times = jnp.concatenate([spikes.data[n].t for n in range(n_neurons)])
    sorted_indices = jnp.argsort(spike_times)
    spike_times = spike_times[sorted_indices]
    spike_ids = spike_ids[sorted_indices]
    X_spikes = jnp.vstack((spike_times, spike_ids))
    y_spikes = jnp.vstack((X_spikes, jnp.arange(spike_times.size)))

    return X_spikes, y_spikes

def compute_reg_strength_gl(scalar_alpha, mean_rates, n_neurons):
    base_alphas_post = mean_rates / mean_rates.mean()
    per_post_alphas = base_alphas_post * scalar_alpha
    alphas = jnp.tile(per_post_alphas[:, None], (1, n_neurons))
    alphas = alphas.at[jnp.arange(n_neurons), jnp.arange(n_neurons)].set(0.0)
    return alphas

def compute_reg_strength_ridge(scalar_alpha, mean_rates, n_features):
    base_alphas_post = mean_rates / mean_rates.mean()
    per_post_alphas = base_alphas_post * scalar_alpha
    alphas = jnp.tile(per_post_alphas[None, :], (n_features, 1))
    return alphas

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

sim_time = 40 # 400, 1000...
n_neurons = 100
binsize = 0.0001
history_window = 0.007
window_size = int(history_window / binsize)
n_basis_funcs = 4
c = 1.3
# basis_fn = RaisedCosineLogEval(history_window, n_basis_funcs)
basis_fn = GenLaguerreEval(history_window, n_basis_funcs, c=c)
int_fn = GenLaguerreInt(history_window, n_basis_funcs, c=c)
time = jnp.linspace(0, history_window, window_size)
kernels = basis_fn(-time)
phi = jnp.exp
phi_inverse = jnp.log
# phi = jax.nn.softplus
# phi_inverse = lambda x: jnp.log(jnp.exp(x) - 1)

# load data
true_conn = np.load('conn_matrix.npy')
true_conn = true_conn + np.eye(n_neurons)
spikes = pickle.load(open('spikes1.pckl', 'rb'))
spikes_tsgroup = nap.TsGroup({n: np.array(spikes[n])/1000 for n in range(len(spikes))})

# training set
train_dur = (sim_time) * 0.8
train_int = nap.IntervalSet(2, 2 + train_dur)
spikes_train = spikes_tsgroup.restrict(train_int)
X_spikes, y_spikes = preprocess_spikes(spikes_train, n_neurons)
mean_rates = jnp.unique(y_spikes[1], return_counts=True)[1] / train_dur
recording_time = train_int

# testing set
test_int = nap.IntervalSet(2 + train_dur, 2 + sim_time)
spikes_test = spikes_tsgroup.restrict(test_int)
X_spikes_test, y_spikes_test = preprocess_spikes(spikes_test, n_neurons)
recording_time_test = test_int

# initialize MC model
obs_model_mc = MonteCarloApproximation(
    n_basis_funcs=n_basis_funcs,
    n_batches_scan=1,
    history_window=history_window,
    control_var=True,
    int_function=int_fn,
    mc_n_samples=int(5e5),
    eval_function=basis_fn,
    inverse_link_function=phi,
)

# Unregularized model fit
model_mc = PopulationContinuousMC(
    solver_name="GradientDescent",
    observation_model=obs_model_mc,
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 2000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 200 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')
plt.plot(error_mc[error_mc!=0])
plt.yscale("log")
plt.show()

# record params and filters
weights_unreg, bias_unreg = model_mc.coef_.reshape(n_neurons, n_basis_funcs, n_neurons), model_mc.intercept_
filters_unreg_raw = np.einsum("jki,tk->ijt", weights_unreg, kernels)
filters_unreg = phi(filters_unreg_raw + bias_unreg[None, :, None])

fpr_unreg, tpr_unreg, roc_auc_unreg, ap_unreg, pred_conn_unreg = compute_roc_curve(
    true_conn.reshape(n_neurons ** 2, -1), filters_unreg_raw.reshape(n_neurons ** 2, -1)
)
# set the thresholded weigths to 0 (this is for test ll computation)
masked_weights_unreg = (weights_unreg * pred_conn_unreg.reshape(-1, n_neurons)[:, None, :]).reshape(-1, n_neurons)
masked_params_unreg = (masked_weights_unreg, jnp.atleast_1d(bias_unreg))
params_unreg = (weights_unreg.reshape(-1, n_neurons), jnp.atleast_1d(bias_unreg))

# Ridge
model_mc = PopulationContinuousMC(
    solver_name="GradientDescent",
    observation_model=obs_model_mc,
    regularizer="Ridge",
    regularizer_strength=compute_reg_strength_ridge(1000, mean_rates, n_neurons * n_basis_funcs),
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 2000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 200 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')
plt.plot(error_mc[error_mc!=0])
plt.yscale("log")
plt.show()

# record params and filters
weights_ridge, bias_ridge = model_mc.coef_.reshape(n_neurons, n_basis_funcs, n_neurons), model_mc.intercept_
filters_ridge_raw = np.einsum("jki,tk->ijt", weights_ridge, kernels)
filters_ridge = phi(filters_ridge_raw + bias_ridge[None, :, None])

fpr_ridge, tpr_ridge, roc_auc_ridge, ap_ridge, pred_conn_ridge = compute_roc_curve(
    true_conn.reshape(n_neurons ** 2, -1), filters_ridge_raw.reshape(n_neurons ** 2, -1)
)
masked_weights_ridge = (weights_ridge * pred_conn_ridge.reshape(-1, n_neurons)[:, None, :]).reshape(-1, n_neurons)
masked_params_ridge = (masked_weights_ridge, jnp.atleast_1d(bias_ridge))
params_ridge = (weights_ridge.reshape(-1, n_neurons), jnp.atleast_1d(bias_ridge))

# Group Lasso
n_groups = n_neurons
n_features = n_groups * n_basis_funcs
mask = np.zeros((n_groups, n_features))
for i in range(n_groups):
    mask[i, i * n_basis_funcs:i * n_basis_funcs + n_basis_funcs] = np.ones(n_basis_funcs)

model_mc = PopulationContinuousMC(
    solver_name="ProximalGradient",
    observation_model=obs_model_mc,
    regularizer=GroupLasso(mask=mask),
    regularizer_strength=compute_reg_strength_gl(100, mean_rates, n_neurons),
    recording_time=recording_time,
    random_key=jax.random.PRNGKey(0),
    solver_kwargs={"has_aux": True, "acceleration": False, "stepsize": -1})
params = model_mc.initialize_params(X_spikes, y_spikes)
state = model_mc.initialize_state(X_spikes, y_spikes, params)
num_iter = 2000
error_mc = np.zeros(num_iter)
for step in range(num_iter):
    params, state = model_mc.update(params, state, X_spikes, y_spikes)
    error_mc[step] = state.error
    if step % 200 == 0:
        print(f'step {step}, error {error_mc[step]}, stepsize {state.stepsize}')
plt.plot(error_mc[error_mc!=0])
plt.yscale("log")
plt.show()

# record params and filters
weights_gl, bias_gl = model_mc.coef_.reshape(n_neurons, n_basis_funcs, n_neurons), model_mc.intercept_
filters_gl_raw = np.einsum("jki,tk->ijt", weights_gl, kernels)
filters_gl = phi(filters_gl_raw + bias_gl[None, :, None])

# check the fraction of connections set to 0
frac_zero = weights_gl.sum(1)[weights_gl.sum(1) == 0].size / weights_gl.sum(1).size
print(f"fraction set to 0: {frac_zero}, true: {1 - (true_conn.sum() / true_conn.size)}")

# predicted connections can be the raw model output or masked with the optimal threshold (like UnReg and Ridge)
# pred_conn_gl = (weights_gl.sum(1) != 0).astype(int)
fpr_gl, tpr_gl, roc_auc_gl, ap_gl, pred_conn_gl = compute_roc_curve(true_conn.reshape(n_neurons ** 2, -1),
                                                         filters_gl_raw.reshape(n_neurons ** 2, -1))

masked_weights_gl = (weights_gl * pred_conn_gl.reshape(-1, n_neurons)[:, None, :]).reshape(-1, n_neurons)
masked_params_gl = (masked_weights_gl, jnp.atleast_1d(bias_gl))
params_gl = (weights_gl.reshape(-1, n_neurons), jnp.atleast_1d(bias_gl))

# AUROC and precision plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# ROC curves
axs[0].plot(fpr_unreg, tpr_unreg, label=f"UnReg (AUC = {roc_auc_unreg:.2f})", c="#ff4dc4")
axs[0].plot(fpr_ridge, tpr_ridge, label=f"Ridge (AUC = {roc_auc_ridge:.2f})", c="#ffa64d")
axs[0].plot(fpr_gl, tpr_gl, label=f"GL (AUC = {roc_auc_gl:.2f})", c="#4d94ff")
axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].set_xlabel("False Positives")
axs[0].set_ylabel("True Positives")
axs[0].legend()
axs[0].set_title("ROC Curves")

# average precision
regularizers = ["UnReg", "Ridge", "GL"]
ap_values = [ap_unreg, ap_ridge, ap_gl]
colors = ["#ff4dc4", "#ffa64d", "#4d94ff"]

axs[1].bar(regularizers, ap_values, color=colors)
axs[1].set_ylabel("average precision")
axs[1].set_title("average precision scores")
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# visualize connections
# white: true negative
# black: false negative
# green: true positive
# red: false positive

binary_cmap = ListedColormap(['white', 'black'])

# initialize and color in the predictions
pred_unreg = np.ones((n_neurons, n_neurons, 3))
pred_ridge = np.ones((n_neurons, n_neurons, 3))
pred_gl = np.ones((n_neurons, n_neurons, 3))

pred_unreg[~pred_conn_unreg.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = [0, 0, 0]
pred_unreg[pred_conn_unreg.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = to_rgb("#9e9d24")
pred_unreg[pred_conn_unreg.reshape(n_neurons, -1).astype(bool) & ~true_conn.astype(bool)] = to_rgb("#ff4d4d")

pred_ridge[~pred_conn_ridge.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = [0, 0, 0]
pred_ridge[pred_conn_ridge.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = to_rgb("#9e9d24")
pred_ridge[pred_conn_ridge.reshape(n_neurons, -1).astype(bool) & ~true_conn.astype(bool)] = to_rgb("#ff4d4d")

pred_gl[~pred_conn_gl.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = [0, 0, 0]
pred_gl[pred_conn_gl.reshape(n_neurons, -1).astype(bool) & true_conn.astype(bool)] = to_rgb("#9e9d24")
pred_gl[pred_conn_gl.reshape(n_neurons, -1).astype(bool) & ~true_conn.astype(bool)] = to_rgb("#ff4d4d")

# plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(true_conn, cmap=binary_cmap)
plt.title("true")

plt.subplot(1, 4, 2)
plt.imshow(pred_unreg, cmap=binary_cmap)
plt.title("UnReg")

plt.subplot(1, 4, 3)
plt.imshow(pred_ridge, cmap=binary_cmap)
plt.title("Ridge")

plt.subplot(1, 4, 4)
plt.imshow(pred_gl, cmap=binary_cmap)
plt.title("Group Lasso")

plt.tight_layout()
plt.show()

# test ll
mc_ll = MonteCarloApproximation(
n_basis_funcs = n_basis_funcs,
n_batches_scan = 3,
history_window = history_window,
mc_n_samples = int(3e6),
control_var = True,
int_function = int_fn,
eval_function = basis_fn,
inverse_link_function = phi,

)
mc_ll._initialize_data_params(recording_time_test, model_mc.max_window, X_spikes_test)
mc_ll._set_ll_function()

mean_params = (jnp.zeros_like(params[0]), jnp.atleast_1d(phi_inverse(mean_rates)))
mean_ll = -1 * mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, mean_params, jax.random.PRNGKey(2))[0]

# raw model output
xval_ll_vals = np.array([
    -1 * mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, params_unreg, jax.random.PRNGKey(2))[0],
    -1 * mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, params_ridge, jax.random.PRNGKey(2))[0],
    -1 * mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, params_gl, jax.random.PRNGKey(2))[0],
])

# or only selected connections
# xval_ll_vals = np.array([
#     -1 *  mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, masked_params_unreg, jax.random.PRNGKey(2))[0],
#     -1 *  mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, masked_params_ridge, jax.random.PRNGKey(2))[0],
#     -1 *  mc_ll._negative_log_likelihood(X_spikes_test, y_spikes_test, masked_params_gl, jax.random.PRNGKey(2))[0],
# ])

# test ll plot
x_labels = ["UnReg", "Ridge", "GL"]
x = np.arange(len(x_labels))

colors = ["#ff4dc4", "#ffa64d", "#4d94ff"]

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.bar(x, (xval_ll_vals - mean_ll), color=colors, edgecolor='black')
ax.set_xticks(x, x_labels)
ax.tick_params(axis='x', labelrotation=45)
ax.set_ylabel("test ll - mean ll")
# can be negative
ax.set_yscale("symlog")
plt.show()

print(xval_ll_vals - mean_ll)