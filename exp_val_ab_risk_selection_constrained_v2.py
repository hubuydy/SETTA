"""
exp_val_ab_risk_selection_constrained_v2.py

Validation-only constrained alpha/beta selection for SETTA.

Purpose
-------
This script fixes K and propagation steps from config.SETTA_FIXED_PARAMS, and searches
only alpha and beta using validation data. It is designed for the manuscript
protocol where K and propagation depth are fixed, while alpha/beta are selected
without using test labels.

Compared with an unrestricted alpha/beta search, this version adds a predefined
shallow-refinement constraint:

1. If a dataset uses one-step refinement, i.e. num_props == 1, alpha is searched
   only in the active propagation range alpha > 0.3. With only one propagation
   step, very small alpha values are near-identity and do not meaningfully test
   semantic refinement.

2. If a dataset uses one-step refinement, beta is also fixed to 1.0 during
   validation selection. This fully activates complementary semantic routes in
   the shallow setting.

3. For all non-one-step datasets, alpha and beta are selected from the full grids.

Selection criterion
-------------------
The script uses validation labels only. Test labels are not used until after a
single dataset-level alpha/beta pair has been selected.

For every alpha/beta pair, it records:
    - validation accuracy
    - validation NLL
    - validation prediction-change rate vs frozen GCN
    - validation probability perturbation L1 vs frozen GCN

Then it ranks configurations by a validation-only risk-controlled criterion:
    1. Mean validation accuracy lower confidence bound:
       LCB = Mean_Val_Acc - lcb_z * SE_Val_Acc
    2. Candidates within near_best_margin percentage points of the best LCB are
       treated as validation-comparable.
    3. Among those candidates, prefer lower validation NLL, lower prediction
       change rate, and lower probability perturbation.

This keeps the protocol validation-based, while avoiding over-interpreting tiny
validation-accuracy fluctuations.

Expected project layout
-----------------------
Put this file in the same directory as your config.py.
Your config.py should provide:
    DEVICE, SEEDS, SETTA_FIXED_PARAMS, NUM_PROPS,
    load_data, train_gcn, build_semantic_adj, get_cit_adj,
    dssr_refine, eval_acc, _get_mask

Example usage
-------------
Run all six datasets with default grids and constraints:
    python exp_val_ab_risk_selection_constrained_v2.py

Run only Computers:
    python exp_val_ab_risk_selection_constrained_v2.py --datasets Computers

Quick test:
    python exp_val_ab_risk_selection_constrained_v2.py --datasets Cora --seeds 42

Disable the shallow constraints for comparison:
    python exp_val_ab_risk_selection_constrained_v2.py --disable-shallow-constraints
"""

import argparse
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Directly reuse all reusable components from config.py.
from config import (  # noqa: E402
    DEVICE,
    SEEDS,
    SETTA_FIXED_PARAMS,
    NUM_PROPS,
    load_data,
    train_gcn,
    build_semantic_adj,
    get_cit_adj,
    dssr_refine,
    eval_acc,
    _get_mask,
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def parse_float_list(values: Iterable[str]) -> List[float]:
    """Parse command-line float values and round to avoid 0.30000000004 keys."""
    return [round(float(v), 6) for v in values]


def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(v) for v in values]


def mask_accuracy(output: torch.Tensor, data, mask: torch.Tensor) -> float:
    """
    Accuracy on an arbitrary mask.

    output can be logits, log-probabilities, or probabilities because argmax is
    unchanged by these representations.
    """
    return (output.argmax(dim=1)[mask] == data.y[mask]).float().mean().item() * 100.0


def eval_val_metrics(
    refined_probs: torch.Tensor,
    original_logits: torch.Tensor,
    data,
    split_idx: int = 0,
) -> Dict[str, float]:
    """
    Compute validation-only metrics for alpha/beta selection.

    Parameters
    ----------
    refined_probs:
        SETTA output probabilities with shape [N, C].
    original_logits:
        Frozen GCN log-probabilities from config.train_gcn(), shape [N, C].
    data:
        PyG data object.
    split_idx:
        Split index for datasets whose masks are two-dimensional.

    Returns
    -------
    dict with:
        Val_Acc: validation accuracy in percentage points.
        Val_NLL: validation negative log-likelihood.
        Val_Change_Rate: percentage of validation nodes whose predicted label
                         changed after refinement.
        Val_Perturb_L1: mean L1 probability perturbation on validation nodes.
    """
    val_mask = _get_mask(data.val_mask, split_idx)
    y_val = data.y[val_mask]

    original_probs = torch.exp(original_logits)

    refined_val = refined_probs[val_mask]
    original_val = original_probs[val_mask]

    pred_refined = refined_val.argmax(dim=1)
    pred_original = original_val.argmax(dim=1)

    val_acc = (pred_refined == y_val).float().mean().item() * 100.0

    # refined_probs are probabilities, so use log(prob) for NLL.
    val_nll = F.nll_loss(torch.log(refined_val + 1e-12), y_val).item()

    val_change_rate = (pred_refined != pred_original).float().mean().item() * 100.0

    # L1 perturbation of probability vectors. Range is roughly [0, 2].
    val_perturb_l1 = (refined_val - original_val).abs().sum(dim=1).mean().item()

    return {
        "Val_Acc": val_acc,
        "Val_NLL": val_nll,
        "Val_Change_Rate": val_change_rate,
        "Val_Perturb_L1": val_perturb_l1,
    }


def summarize_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def constrained_search_space(
    ds_name: str,
    base_alphas: List[float],
    base_betas: List[float],
    fixed_steps: int,
    one_step_min_alpha: float,
    force_beta_one_for_one_step: bool,
    enable_shallow_constraints: bool,
) -> Tuple[List[float], List[float], List[str]]:
    """
    Build dataset-specific validation search space.

    Correct constrained protocol:
        - If fixed_steps == 1, restrict BOTH alpha and beta:
            alpha > one_step_min_alpha
            beta = 1.0  (unless disabled from CLI)
        - If fixed_steps != 1, use the full base alpha/beta grids.

    This rule is based on the propagation depth and not on any test-set result.
    """
    alphas = list(base_alphas)
    betas = list(base_betas)
    notes: List[str] = []

    if enable_shallow_constraints and fixed_steps == 1:
        old_alphas = alphas
        alphas = [a for a in old_alphas if a > one_step_min_alpha]
        notes.append(
            f"one-step refinement: alpha constrained to > {one_step_min_alpha}"
        )

        if force_beta_one_for_one_step:
            if 1.0 not in [round(float(b), 6) for b in betas]:
                raise ValueError(
                    f"beta=1.0 is required for one-step dataset {ds_name}, "
                    f"but 1.0 is not in the provided beta grid: {base_betas}"
                )
            betas = [1.0]
            notes.append("one-step refinement: beta fixed to 1.0")

    if not alphas:
        raise ValueError(
            f"No alpha candidates left for {ds_name}. "
            f"Check --alphas and --one-step-min-alpha."
        )
    if not betas:
        raise ValueError(f"No beta candidates left for {ds_name}. Check --betas.")

    return alphas, betas, notes


# -----------------------------------------------------------------------------
# Validation-only risk-controlled selection
# -----------------------------------------------------------------------------

def rank_alpha_beta_by_validation_risk(
    grid_df: pd.DataFrame,
    lcb_z: float = 1.0,
    near_best_margin: float = 0.20,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Select alpha/beta using validation-only risk-controlled ranking.

    Parameters
    ----------
    grid_df:
        Rows from validation grid search. Must contain:
        Seed, Alpha, Beta, Val_Acc, Val_NLL, Val_Change_Rate, Val_Perturb_L1.
    lcb_z:
        Strength of the lower confidence bound penalty.
        LCB = Mean_Val_Acc - lcb_z * SE_Val_Acc.
    near_best_margin:
        Percentage-point margin from the best LCB. Candidates with
        Val_Acc_LCB >= Best_LCB - near_best_margin are considered practically
        comparable on validation accuracy. Set to 0.0 for strict LCB selection.

    Returns
    -------
    best:
        Selected alpha/beta row as a dictionary.
    ranked:
        Full grouped ranking table.
    candidates:
        Near-best candidates considered by the risk-control tie-breakers.
    """
    required = {
        "Seed",
        "Alpha",
        "Beta",
        "Val_Acc",
        "Val_NLL",
        "Val_Change_Rate",
        "Val_Perturb_L1",
    }
    missing = required.difference(grid_df.columns)
    if missing:
        raise ValueError(f"grid_df is missing required columns: {sorted(missing)}")

    grouped = (
        grid_df.groupby(["Alpha", "Beta"], as_index=False)
        .agg(
            Mean_Val_Acc=("Val_Acc", "mean"),
            Std_Val_Acc=("Val_Acc", "std"),
            Mean_Val_NLL=("Val_NLL", "mean"),
            Std_Val_NLL=("Val_NLL", "std"),
            Mean_Change_Rate=("Val_Change_Rate", "mean"),
            Mean_Perturb_L1=("Val_Perturb_L1", "mean"),
            Num_Seeds=("Seed", "nunique"),
        )
        .fillna({"Std_Val_Acc": 0.0, "Std_Val_NLL": 0.0})
    )

    grouped["SE_Val_Acc"] = grouped["Std_Val_Acc"] / np.sqrt(grouped["Num_Seeds"].clip(lower=1))
    grouped["Val_Acc_LCB"] = grouped["Mean_Val_Acc"] - lcb_z * grouped["SE_Val_Acc"]

    best_lcb = float(grouped["Val_Acc_LCB"].max())
    grouped["LCB_Gap_To_Best"] = best_lcb - grouped["Val_Acc_LCB"]

    # Candidate set: validation-near-best by LCB only.
    candidates = grouped[grouped["Val_Acc_LCB"] >= best_lcb - near_best_margin].copy()

    # Risk-controlled ranking inside the validation-near-best set.
    # Lower NLL, lower change rate, and lower perturbation mean less aggressive
    # and usually more stable refinement.
    candidates = candidates.sort_values(
        by=[
            "Mean_Val_NLL",
            "Mean_Change_Rate",
            "Mean_Perturb_L1",
            "Val_Acc_LCB",
            "Mean_Val_Acc",
            "Alpha",
            "Beta",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    if candidates.empty:
        raise RuntimeError("No validation candidates found. Check near_best_margin and input grid.")

    best = candidates.iloc[0].to_dict()
    best["Best_LCB_Overall"] = best_lcb
    best["Near_Best_Margin"] = float(near_best_margin)
    best["LCB_Z"] = float(lcb_z)
    best["Selection_Protocol"] = "constrained_validation_lcb_near_best_then_low_risk"

    # Also save a full ranking view for inspection.
    ranked = grouped.sort_values(
        by=[
            "Val_Acc_LCB",
            "Mean_Val_Acc",
            "Mean_Val_NLL",
            "Mean_Change_Rate",
            "Mean_Perturb_L1",
            "Alpha",
            "Beta",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return best, ranked, candidates


# -----------------------------------------------------------------------------
# Core experiment
# -----------------------------------------------------------------------------

def run_one_dataset(
    ds_name: str,
    base_alphas: List[float],
    base_betas: List[float],
    seeds: List[int],
    split_idx: int,
    lcb_z: float,
    near_best_margin: float,
    output_dir: str,
    one_step_min_alpha: float,
    force_beta_one_for_one_step: bool,
    enable_shallow_constraints: bool,
) -> Dict[str, object]:
    """Run validation-only alpha/beta selection and final test evaluation."""
    if ds_name not in SETTA_FIXED_PARAMS:
        raise ValueError(f"{ds_name} not found in config.SETTA_FIXED_PARAMS")

    cfg = SETTA_FIXED_PARAMS[ds_name]
    fixed_k = int(cfg["k"])
    metric = cfg["metric"]
    fixed_steps = int(cfg.get("num_props", NUM_PROPS))

    search_alphas, search_betas, constraint_notes = constrained_search_space(
        ds_name=ds_name,
        base_alphas=base_alphas,
        base_betas=base_betas,
        fixed_steps=fixed_steps,
        one_step_min_alpha=one_step_min_alpha,
        force_beta_one_for_one_step=force_beta_one_for_one_step,
        enable_shallow_constraints=enable_shallow_constraints,
    )
    search_space_note = "; ".join(constraint_notes) if constraint_notes else "full alpha/beta validation grid"

    print("\n" + "=" * 80)
    print(f"Dataset: {ds_name}")
    print(
        f"Fixed from config.py: K={fixed_k}, steps={fixed_steps}, metric={metric}, "
        "alpha/beta selected from validation grid"
    )
    print(f"Effective alpha search: {search_alphas}")
    print(f"Effective beta search : {search_betas}")
    print(f"Search-space note     : {search_space_note}")
    print("=" * 80)

    ds, data = load_data(ds_name)

    # Build semantic/topological adjacencies once per dataset using fixed K.
    t0 = time.time()
    sem = build_semantic_adj(data, fixed_k, metric)
    cit = get_cit_adj(data)
    print(f"Built semantic/topological adjacencies in {time.time() - t0:.2f}s")

    grid_rows: List[Dict[str, object]] = []
    gcn_cache: Dict[int, torch.Tensor] = {}
    base_val_rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Stage 1: validation grid search only. Test labels are not touched.
    # ------------------------------------------------------------------
    for seed in seeds:
        print(f"  Seed {seed}: training frozen GCN via config.train_gcn()")
        gcn_out = train_gcn(ds, data, seed=seed, split_idx=split_idx)
        gcn_cache[seed] = gcn_out.detach()

        val_mask = _get_mask(data.val_mask, split_idx)
        base_val_acc = mask_accuracy(gcn_out, data, val_mask)

        base_val_rows.append(
            {
                "Dataset": ds_name,
                "Seed": seed,
                "GCN_Val_Acc": base_val_acc,
            }
        )

        for alpha in search_alphas:
            for beta in search_betas:
                with torch.no_grad():
                    refined = dssr_refine(
                        gcn_out,
                        data,
                        sem,
                        cit,
                        alpha=float(alpha),
                        beta=float(beta),
                        metric=metric,
                        num_props=fixed_steps,
                    )

                    val_metrics = eval_val_metrics(
                        refined_probs=refined,
                        original_logits=gcn_out,
                        data=data,
                        split_idx=split_idx,
                    )

                grid_rows.append(
                    {
                        "Dataset": ds_name,
                        "Seed": seed,
                        "Alpha": float(alpha),
                        "Beta": float(beta),
                        "Fixed_K": fixed_k,
                        "Fixed_Steps": fixed_steps,
                        "Metric": metric,
                        "Search_Space_Note": search_space_note,
                        **val_metrics,
                        "GCN_Val_Acc": base_val_acc,
                    }
                )

    grid_df = pd.DataFrame(grid_rows)
    base_val_df = pd.DataFrame(base_val_rows)

    # Select one dataset-level alpha/beta pair from validation metrics only.
    best, ranked, candidates = rank_alpha_beta_by_validation_risk(
        grid_df=grid_df,
        lcb_z=lcb_z,
        near_best_margin=near_best_margin,
    )

    selected_alpha = float(best["Alpha"])
    selected_beta = float(best["Beta"])

    print(
        f"\nSelected by constrained validation-only risk criterion: "
        f"alpha={selected_alpha}, beta={selected_beta}"
    )
    print(
        f"Mean Val Acc={best['Mean_Val_Acc']:.4f}, "
        f"LCB={best['Val_Acc_LCB']:.4f}, "
        f"Val NLL={best['Mean_Val_NLL']:.6f}, "
        f"Change={best['Mean_Change_Rate']:.4f}%, "
        f"Perturb L1={best['Mean_Perturb_L1']:.6f}"
    )

    # ------------------------------------------------------------------
    # Stage 2: final test evaluation once the parameters are fixed.
    # Test labels are used here only for final reporting.
    # ------------------------------------------------------------------
    final_rows: List[Dict[str, object]] = []
    for seed in seeds:
        gcn_out = gcn_cache[seed]

        with torch.no_grad():
            refined = dssr_refine(
                gcn_out,
                data,
                sem,
                cit,
                alpha=selected_alpha,
                beta=selected_beta,
                metric=metric,
                num_props=fixed_steps,
            )

            val_metrics = eval_val_metrics(
                refined_probs=refined,
                original_logits=gcn_out,
                data=data,
                split_idx=split_idx,
            )
            gcn_test_acc = eval_acc(gcn_out, data, split_idx=split_idx) * 100.0
            setta_test_acc = eval_acc(refined, data, split_idx=split_idx) * 100.0

        base_info = base_val_df[base_val_df["Seed"] == seed].iloc[0].to_dict()

        final_rows.append(
            {
                "Dataset": ds_name,
                "Seed": seed,
                "Selected_Alpha": selected_alpha,
                "Selected_Beta": selected_beta,
                "Fixed_K": fixed_k,
                "Fixed_Steps": fixed_steps,
                "Metric": metric,
                "Search_Space_Note": search_space_note,
                "Selection_Protocol": best["Selection_Protocol"],
                "LCB_Z": lcb_z,
                "Near_Best_Margin": near_best_margin,
                "GCN_Val_Acc": float(base_info["GCN_Val_Acc"]),
                "GCN_Test_Acc": gcn_test_acc,
                "SETTA_Val_Acc": val_metrics["Val_Acc"],
                "SETTA_Val_NLL": val_metrics["Val_NLL"],
                "SETTA_Val_Change_Rate": val_metrics["Val_Change_Rate"],
                "SETTA_Val_Perturb_L1": val_metrics["Val_Perturb_L1"],
                "SETTA_Test_Acc": setta_test_acc,
                "Delta_Test_Acc": setta_test_acc - gcn_test_acc,
            }
        )

    final_df = pd.DataFrame(final_rows)

    # Dataset summary.
    gcn_test_mean, gcn_test_std = summarize_mean_std(final_df["GCN_Test_Acc"].tolist())
    setta_test_mean, setta_test_std = summarize_mean_std(final_df["SETTA_Test_Acc"].tolist())
    delta_mean, delta_std = summarize_mean_std(final_df["Delta_Test_Acc"].tolist())
    gcn_val_mean, gcn_val_std = summarize_mean_std(final_df["GCN_Val_Acc"].tolist())
    setta_val_mean, setta_val_std = summarize_mean_std(final_df["SETTA_Val_Acc"].tolist())

    summary_row = {
        "Dataset": ds_name,
        "Selected_Alpha": selected_alpha,
        "Selected_Beta": selected_beta,
        "Fixed_K": fixed_k,
        "Fixed_Steps": fixed_steps,
        "Metric": metric,
        "Search_Space_Note": search_space_note,
        "Selection_Protocol": best["Selection_Protocol"],
        "LCB_Z": lcb_z,
        "Near_Best_Margin": near_best_margin,
        "Selected_Val_Acc": float(best["Mean_Val_Acc"]),
        "Mean_Val_Acc": float(best["Mean_Val_Acc"]),
        "Selected_Mean_Val_Acc_From_Search": float(best["Mean_Val_Acc"]),
        "Selected_Val_Acc_LCB_From_Search": float(best["Val_Acc_LCB"]),
        "Selected_Mean_Val_NLL_From_Search": float(best["Mean_Val_NLL"]),
        "Selected_Mean_Change_Rate_From_Search": float(best["Mean_Change_Rate"]),
        "Selected_Mean_Perturb_L1_From_Search": float(best["Mean_Perturb_L1"]),
        "GCN_Val_Mean": gcn_val_mean,
        "GCN_Val_Std": gcn_val_std,
        "SETTA_Val_Mean": setta_val_mean,
        "SETTA_Val_Std": setta_val_std,
        "GCN_Test_Mean": gcn_test_mean,
        "GCN_Test_Std": gcn_test_std,
        "SETTA_Test_Mean": setta_test_mean,
        "SETTA_Test_Std": setta_test_std,
        "Delta_Test_Mean": delta_mean,
        "Delta_Test_Std": delta_std,
        "Num_Seeds": len(seeds),
    }

    print(
        f"Final test: GCN {gcn_test_mean:.2f}±{gcn_test_std:.2f}, "
        f"SETTA {setta_test_mean:.2f}±{setta_test_std:.2f}, "
        f"Delta {delta_mean:+.2f}±{delta_std:.2f}"
    )

    # Save per-dataset debugging files.
    safe_name = ds_name.replace("/", "_")
    grid_df.to_csv(os.path.join(output_dir, f"{safe_name}_grid_validation_only.csv"), index=False)
    ranked.to_csv(os.path.join(output_dir, f"{safe_name}_ranked_alpha_beta.csv"), index=False)
    candidates.to_csv(os.path.join(output_dir, f"{safe_name}_near_best_risk_candidates.csv"), index=False)
    final_df.to_csv(os.path.join(output_dir, f"{safe_name}_final_test_per_seed.csv"), index=False)

    return {
        "grid_df": grid_df,
        "ranked_df": ranked,
        "candidates_df": candidates,
        "final_df": final_df,
        "summary_row": summary_row,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constrained validation-only risk-controlled alpha/beta selection for SETTA."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora", "CiteSeer", "PubMed", "CS", "Computers", "Photo"],
        help="Datasets to run. Must exist in config.SETTA_FIXED_PARAMS.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=SEEDS,
        type=int,
        help="Random seeds. Default uses config.SEEDS.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
        help="Base alpha grid. Default: 0.1, ..., 0.9. One-step datasets use alpha > one-step-min-alpha.",
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        default=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
        help="Base beta grid. Default: 0.0, ..., 1.0. One-step datasets are forced to beta=1.0.",
    )
    parser.add_argument(
        "--split-idx",
        type=int,
        default=0,
        help="Split index for datasets with 2D masks. Default: 0.",
    )
    parser.add_argument(
        "--lcb-z",
        type=float,
        default=1.0,
        help="LCB penalty strength: LCB = mean val acc - lcb_z * SE. Default: 1.0.",
    )
    parser.add_argument(
        "--near-best-margin",
        type=float,
        default=0.20,
        help=(
            "Percentage-point margin from the best validation LCB. "
            "Candidates within this margin are ranked by validation risk. "
            "Set 0.0 for strict LCB selection. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--one-step-min-alpha",
        type=float,
        default=0.3,
        help="For one-step datasets, search only alpha > this value. Default: 0.3.",
    )
    parser.add_argument(
        "--disable-one-step-beta-constraint",
        action="store_true",
        help="Disable the rule that one-step datasets are forced to beta=1.0.",
    )
    parser.add_argument(
        "--disable-shallow-constraints",
        action="store_true",
        help="Disable both shallow-refinement constraints and use the full base alpha/beta grids.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation_ab_selection",
        help="Output directory for CSV files.",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    base_alphas = parse_float_list(args.alphas)
    base_betas = parse_float_list(args.betas)
    seeds = parse_int_list(args.seeds)

    os.makedirs(args.output_dir, exist_ok=True)

    enable_shallow_constraints = not args.disable_shallow_constraints
    force_beta_one_for_one_step = not args.disable_one_step_beta_constraint

    print("=" * 80)
    print("Constrained validation-only alpha/beta selection for SETTA")
    print("Device:", DEVICE)
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {seeds}")
    print(f"Base alpha grid: {base_alphas}")
    print(f"Base beta grid : {base_betas}")
    print(f"Shallow constraints enabled: {enable_shallow_constraints}")
    print(f"One-step alpha rule: alpha > {args.one_step_min_alpha}")
    print(f"Force beta=1.0 when steps=1: {force_beta_one_for_one_step}")
    print(f"LCB z: {args.lcb_z}")
    print(f"Near-best margin: {args.near_best_margin} percentage points")
    print(f"Output directory: {args.output_dir}")
    print("IMPORTANT: test labels are not used during alpha/beta selection.")
    print("=" * 80)

    all_grid = []
    all_ranked = []
    all_candidates = []
    all_final = []
    summary_rows = []

    start_all = time.time()

    for ds_name in args.datasets:
        result = run_one_dataset(
            ds_name=ds_name,
            base_alphas=base_alphas,
            base_betas=base_betas,
            seeds=seeds,
            split_idx=args.split_idx,
            lcb_z=args.lcb_z,
            near_best_margin=args.near_best_margin,
            output_dir=args.output_dir,
            one_step_min_alpha=args.one_step_min_alpha,
            force_beta_one_for_one_step=force_beta_one_for_one_step,
            enable_shallow_constraints=enable_shallow_constraints,
        )

        all_grid.append(result["grid_df"])
        ranked_df = result["ranked_df"].copy()
        ranked_df.insert(0, "Dataset", ds_name)
        all_ranked.append(ranked_df)

        cand_df = result["candidates_df"].copy()
        cand_df.insert(0, "Dataset", ds_name)
        all_candidates.append(cand_df)

        all_final.append(result["final_df"])
        summary_rows.append(result["summary_row"])

    # Save combined CSVs.
    combined_grid = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame()
    combined_ranked = pd.concat(all_ranked, ignore_index=True) if all_ranked else pd.DataFrame()
    combined_candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    combined_final = pd.concat(all_final, ignore_index=True) if all_final else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    combined_grid.to_csv(
        os.path.join(args.output_dir, "val_ab_grid_history_validation_only.csv"),
        index=False,
    )
    combined_ranked.to_csv(
        os.path.join(args.output_dir, "ranked_alpha_beta_by_dataset.csv"),
        index=False,
    )
    combined_candidates.to_csv(
        os.path.join(args.output_dir, "near_best_risk_candidates_by_dataset.csv"),
        index=False,
    )
    combined_final.to_csv(
        os.path.join(args.output_dir, "val_selected_ab_final_test_per_seed.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(args.output_dir, "val_selected_ab_summary.csv"),
        index=False,
    )

    print("\n" + "=" * 80)
    print("Finished all datasets")
    print(f"Elapsed: {time.time() - start_all:.2f}s")
    print("Saved:")
    print(f"  {os.path.join(args.output_dir, 'val_ab_grid_history_validation_only.csv')}")
    print(f"  {os.path.join(args.output_dir, 'ranked_alpha_beta_by_dataset.csv')}")
    print(f"  {os.path.join(args.output_dir, 'near_best_risk_candidates_by_dataset.csv')}")
    print(f"  {os.path.join(args.output_dir, 'val_selected_ab_final_test_per_seed.csv')}")
    print(f"  {os.path.join(args.output_dir, 'val_selected_ab_summary.csv')}")
    print("=" * 80)

    if not summary_df.empty:
        cols = [
            "Dataset",
            "Selected_Alpha",
            "Selected_Beta",
            "Fixed_K",
            "Fixed_Steps",
            "Search_Space_Note",
            "GCN_Test_Mean",
            "GCN_Test_Std",
            "SETTA_Test_Mean",
            "SETTA_Test_Std",
            "Delta_Test_Mean",
            "Delta_Test_Std",
        ]
        print("\nSummary preview:")
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
