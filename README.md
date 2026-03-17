# Phase Transitions in Active Inference

This repository contains the code and dev-notebooks for modelling developmental phase transitions in active inference, with a minimal bifurcation model and a richer balance-scale task implementation.

## Repository structure

- `sim_bifurcation.py` - minimal two-state model and bifurcation verification
- `sim_balance_scale_pymdp.py` - main balance-scale implementation using `pymdp`
- `sim_balance_scale_appendix.py` - reduced model for the appendix (simpler, bridge-type)
- `notebooks/` - mathematical derivation and simulation notebooks in Markdown
- `third_party/pymdp/` - vendored local copy of `pymdp` used by the main implementation

## Main model

The main paper-facing implementation is:

```bash
python sim_balance_scale_pymdp.py
```

The reduced appendix model is:

```bash
python sim_balance_scale_appendix.py
```

The minimal bifurcation model is:

```bash
python sim_bifurcation.py
```

## Environment setup

Create a clean environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Outputs
Generated figures are written to Figs/ at runtime and are uploaded in Git only for use in the notebooks.

## Notebooks
In the relevant subfolder. This is where the analytical results and numerical sim mappings live, with step-by-step derivation of the main results.

## Lean 4
Those main results (notebook 01) have been converted into Lean 4 (https://github.com/leanprover) to double-check validity of derivations.

## Note
* The third_party/pymdp directory is a vendored dependency used for reproducibility; for details: https://github.com/infer-actively/pymdp
* The notebooks are stored separately in notebooks/.

## Last review
* 17 March 2026


