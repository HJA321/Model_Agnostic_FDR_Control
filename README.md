# Model-Agnostic FDR Control

This repository is the code repository for NeurIPS 2026 paper: Model-Agnostic FDR Control via Group Gaussian Mirror and Permutation SHAP.
If using this repository, please cite:
Han, J., Chen, J., & Fu, Y. (2026). Model-Agnostic FDR Control via Group Gaussian Mirror and Permutation SHAP. ArXiv, abs/2608.00989.

This repository includes code for Group Gaussian Mirror (GGM) and Permutation SHAP Gaussian
Mirror (PSGM), including synthetic simulations.


## Environment

Use Python 3.10. The pinned direct dependencies match a previously tested
environment; a clean installation and full-scale GPU runs remain to be verified.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running from the repository root

The scripts have substantial default training budgets. Use a small configuration
to check an installation before launching full experiments.

```bash
python -m simulation_codes.simulation
python simulation_codes/linear_extra_experiments.py --help
python simulation_codes/run_gj_lr_optimization_curves.py --help
```

PSGM is run by `simulation_codes/simulation.py`.

For example, to configure a high-dimensional GRU PSGM experiment:

```python
from simulation_codes import simulation as sim
sim.MODEL_TYPE = "gru"
sim.N_RELEVANT = 15
sim.N_IRRELEVANT = 685
sim.LOOKBACK = 5
sim.GROUP_KW_PCA_COMPONENTS = 2 * sim.LOOKBACK
sim.IMPORTANCE_MODES = ("shap",)
sim.main()
```

`simulation_codes/linear_extra_experiments.py` uses column counts, not group
counts: with group size 5, use 150 relevant columns and 1350/3350 null columns
for 300/700 groups respectively. Use `--data-mode lag_group` for the AR grouped
design, `--data-mode iid_group` for the static grouped design, and
`--experiment ar_error` for AR response errors and OLS/GLS comparisons.



