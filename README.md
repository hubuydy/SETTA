# SETTA

SETTA is a training-free, learnable-parameter-free, prediction-level graph test-time adaptation framework for frozen GNN predictions. It refines frozen node-classification outputs by combining semantic route construction, spectral-energy-guided prediction diffusion, and entropy-gated selective acceptance.

This repository contains the experimental code for:

**SETTA: Parameter-Free Test-Time Adaptation for Graph Neural Networks via Spectral-Energy-Guided Semantic Refinement**

## Core Protocol

SETTA follows a validation-based configuration-selection protocol.

- `K`, distance `metric`, and propagation depth `num_props` are fixed dataset-level structural settings in `config.SETTA_FIXED_PARAMS`.
- `alpha` and `beta` are selected from the validation split only.
- The test split is used only once after a dataset-level `alpha`/`beta` pair has been selected.
- For multi-step datasets, validation search uses `alpha in {0.1, ..., 0.9}` and `beta in {0.0, ..., 1.0}`.
- For one-step datasets (`num_props == 1`), the constrained shallow-refinement search fixes `beta = 1.0` and searches `alpha in {0.4, ..., 0.9}`.

The standard validation-selected configuration file is:

```text
results/validation_ab_selection/val_selected_ab_summary.csv
```

Main experiment scripts call `config.get_setta_config()`. If the validation-selected CSV is missing or lacks a dataset, the scripts fail with a clear error instead of silently falling back to hard-coded `alpha`/`beta` values.

## Repository Layout

```text
.
├── config.py
├── exp_val_ab_risk_selection_constrained_v2.py
├── exp_validation-based_selection.py
├── exp1_main_results.py
├── exp2_scalability_robustness.py
├── exp3_postprocessing.py
├── exp4_ablation.py
├── exp5_sensitivity.py
├── exp6_transition_analysis.py
├── exp7.py
├── run_validation_then_main.py
└── plot_all_figures.py
```

Generated result files and manuscript figures are written to `results/` and `figures/`, respectively. These output directories are ignored by Git.

## Environment

The experiments were developed with Python and PyTorch/PyTorch Geometric. A typical environment includes:

```bash
pip install -r requirements.txt
```

`torch` and `torch-geometric` installation can depend on CUDA, operating system, and Python version. If needed, install them from the official PyTorch and PyTorch Geometric instructions for your platform.

## Datasets

The code uses public benchmark datasets:

- Cora
- CiteSeer
- PubMed
- Coauthor-CS
- Amazon-Computers
- Amazon-Photo

The datasets are loaded through PyTorch Geometric. Local dataset caches are not included in this repository; they will be downloaded or processed by the dataset loaders when the scripts are run.

## Standard Run Order

Step 1: generate validation-selected `alpha`/`beta`.

```bash
python exp_val_ab_risk_selection_constrained_v2.py
```

This writes:

```text
results/validation_ab_selection/val_selected_ab_summary.csv
```

Step 2: run the main experiments.

```bash
python exp1_main_results.py
python exp2_scalability_robustness.py
python exp3_postprocessing.py
python exp4_ablation.py
python exp6_transition_analysis.py
python exp7.py
```

Alternatively, run validation selection followed by selected main scripts:

```bash
python run_validation_then_main.py
```

To reuse an existing validation-selected configuration:

```bash
python run_validation_then_main.py --skip-selection
```

## Figures

After result files are available, regenerate manuscript figures with:

```bash
python plot_all_figures.py
```

## Experiment Summary

- `exp1_main_results.py`: primary frozen-GCN SETTA results.
- `exp2_scalability_robustness.py`: larger benchmark graphs and runtime-related evaluation.
- `exp3_postprocessing.py`: prediction-level refinement baseline comparison.
- `exp4_ablation.py`: semantic injection, denoising, and risk-control ablations.
- `exp5_sensitivity.py`: validation-sensitivity visualization data.
- `exp6_transition_analysis.py`: node-level Wrong-to-Correct and Correct-to-Wrong transition analysis.
- `exp7.py`: backbone-agnostic frozen prediction refinement across multiple GNN predictors.

## Legacy Names

Some implementation symbols retain the old `DSSR`/`dssr_refine` name for backward compatibility. In the manuscript and result presentation, the method is SETTA. In code, `dssr_refine` is the legacy implementation name of SETTA refinement.

## Notes

- SETTA is not configuration-free: structural settings are fixed by dataset-level protocol, while `alpha` and `beta` are selected from validation data.
- SETTA does not use test labels, gradients, or parameter updates during adaptation.
- Current implementation focuses on benchmark-scale graph mining and uses dense semantic graph construction.
