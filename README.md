# Point Process GLM

A memory-efficient implementation of a Poisson point process GLM for massive datasets.

## Installation

To install the packege in editable mode (along with all dependencies):

```bash
cd poisson-process-glm
pip install -e .
```

## Example script
This script shows how to run and evaluate the models on data simulated from all-to-one coupled GLM. It is for demonstration only and does not reproduce specific figures. 
The script is designed to run on GPU; while it can fall back to CPU, run times may be significantly longer.

To run:

```bash
cd poisson-process-glm/_scripts
python run_pp_glm.py
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