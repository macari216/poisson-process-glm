# Point Process GLM

A memory-efficient implementation of a Poisson point process GLM for massive datasets.

## Installation

To install the packege (along with all dependencies):

```bash
pip install git+https://github.com/macari216/poisson-process-glm.git
```
To install in editable mode:
```bash
git clone https://github.com/macari216/poisson-process-glm.git
cd poisson-process-glm
pip install -e .
```

## Example scripts
There are two example scripts that introduce fitting Monte Carlo (MC), polynomial approximation (PA), 
and hybrid PA-MC PPGLMs on simulated data. `run_pp_glm.py` fits a single postsyaptic neuron and `run_population_pp_glm.py` 
fits a small recurrently connected neuronal population.

To run:

```bash
cd poisson-process-glm/_scripts
python run_pp_glm.py # or run_population_pp_glm.py
```

The output will be saved to `poisson-process-glm/_results/pp_glm_results.npz`.

To load the results in python:
```python
import numpy as np

loaded = np.load("_results/pp_glm_results.npz", allow_pickle=True)
results_pp_glm = {key: loaded[key].item() for key in loaded}
```

This will load a nested dictionary where each key corresponds to a model (PA-c, MC, Hybrid)
and each value contains the model results (run time, estimated filters, and MSE). The results also contain simulation parameters and true filters.