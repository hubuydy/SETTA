"""
exp6_transition_analysis.py — SETTA node-level transition analysis

Purpose:
    Quantify whether SETTA mainly fixes wrong GCN predictions or also damages
    originally correct predictions.

Outputs:
    results/exp6_transition_raw.csv
    results/exp6_transition_summary.csv
    results/exp6_transition_paper_table.csv

Protocol:
    For each dataset and seed:
      1. Train the same two-layer GCN backbone used by the existing scripts.
      2. Apply SETTA to the frozen GCN log-probabilities.
      3. Compare GCN vs. SETTA predictions on the test mask.
      4. Count Wrong->Correct, Correct->Wrong, Wrong->Wrong, Correct->Correct.

Notes:
    - This script does not tune parameters and does not use test labels during SETTA.
      Test labels are used only after adaptation for diagnostic evaluation.
    - The SETTA refinement logic mirrors config.dssr_refine, but returns the gating
      mask and diagnostic quantities needed for transition analysis.
"""

import argparse
import gc
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from config import (
    DEVICE,
    DEFAULT_VALIDATION_SELECTED_AB_PATH,
    NUM_PROPS,
    SEEDS,
    build_semantic_adj,
    calc_dirichlet_energy,
    eval_acc,
    get_cit_adj,
    get_setta_config,
    load_data,
    train_gcn,
)
from config import _get_mask


ALL_DATASETS = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]


@torch.no_grad()
def setta_refine_with_diagnostics(
    logits: torch.Tensor,
    data,
    sem_rewire: torch.Tensor,
    cit_adj: torch.Tensor,
    alpha: float,
    beta: float,
    num_props: int = NUM_PROPS,
) -> Dict[str, torch.Tensor]:
    """Run SETTA refinement and return final predictions plus diagnostics.

    This intentionally mirrors the current config.dssr_refine implementation:
    - Z0 = exp(logits)
    - inject semantic edges absent from the observed topology
    - row-normalize the hybrid adjacency
    - run controlled prediction diffusion
    - if the energy early-exit condition triggers, the low-energy Z_new is NOT
      accepted, matching the original code path that breaks before `Z = Z_new`
    - apply entropy/confidence gating using the same scale-aware thresholds
    """
    Z0 = torch.exp(logits)
    num_classes = Z0.size(1)

    m_rewire = (sem_rewire == 1) & (cit_adj == 0)
    hybrid = cit_adj + beta * m_rewire.float() + torch.eye(cit_adj.shape[0], device=DEVICE)
    norm = hybrid / (hybrid.sum(1, keepdim=True) + 1e-10)

    Z = Z0.clone()
    e0 = calc_dirichlet_energy(Z0, norm)
    used_steps = 0
    stopped_early = False
    last_energy = e0

    for step in range(num_props):
        Z_new = (1.0 - alpha) * Z + alpha * torch.mm(norm, Z)
        e_current = calc_dirichlet_energy(Z_new, norm)
        last_energy = e_current

        if step >= 3 and e_current < 0.08 * e0:
            stopped_early = True
            # Match config.dssr_refine: break before assigning Z = Z_new.
            break

        Z = Z_new
        used_steps = step + 1

    # Scale-aware zero-shot entropy/confidence gating.
    n_nodes = data.num_nodes
    scale_factor = torch.log10(torch.tensor(n_nodes, dtype=torch.float32))
    tau_max = max(0.85, 0.99 - 0.128 * max(0.0, scale_factor.item() - 3.43))
    tau_min = max(0.70, tau_max - 0.20)

    entropy = -(Z0 * torch.log(Z0 + 1e-9)).sum(dim=1)
    entropy_norm = entropy / torch.log(torch.tensor(num_classes, dtype=Z0.dtype, device=Z0.device))
    tau_v = tau_max - (tau_max - tau_min) * entropy_norm

    base_confidence = Z0.max(dim=1)[0]
    accepted_mask = base_confidence < tau_v
    final_Z = torch.where(accepted_mask.unsqueeze(1), Z, Z0)

    return {
        "final": final_Z,
        "propagated": Z,
        "accepted_mask": accepted_mask,
        "tau_v": tau_v,
        "base_confidence": base_confidence,
        "entropy_norm": entropy_norm,
        "used_steps": torch.tensor(used_steps),
        "stopped_early": torch.tensor(stopped_early),
        "e0": torch.tensor(e0),
        "last_energy": torch.tensor(last_energy),
    }


def _safe_mean_bool(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return x.float().mean().item()


def transition_record(
    dataset_name: str,
    seed: int,
    gcn_logits: torch.Tensor,
    setta_logits: torch.Tensor,
    accepted_mask: torch.Tensor,
    data,
    used_steps: int,
    stopped_early: bool,
    split_idx: int = 0,
) -> Dict[str, float]:
    """Compute Wrong->Correct / Correct->Wrong transitions on the test split."""
    test_mask = _get_mask(data.test_mask, split_idx).bool()
    y = data.y[test_mask]

    gcn_pred = gcn_logits.argmax(dim=1)[test_mask]
    setta_pred = setta_logits.argmax(dim=1)[test_mask]
    accepted = accepted_mask[test_mask]

    gcn_correct = gcn_pred.eq(y)
    setta_correct = setta_pred.eq(y)
    changed = gcn_pred.ne(setta_pred)

    wrong_to_correct = (~gcn_correct & setta_correct).sum().item()
    correct_to_wrong = (gcn_correct & ~setta_correct).sum().item()
    wrong_to_wrong = (~gcn_correct & ~setta_correct).sum().item()
    correct_to_correct = (gcn_correct & setta_correct).sum().item()

    accepted_n = accepted.sum().item()
    kept_n = (~accepted).sum().item()
    n_test = test_mask.sum().item()

    accepted_gcn_acc = _safe_mean_bool(gcn_correct[accepted])
    accepted_setta_acc = _safe_mean_bool(setta_correct[accepted])
    kept_gcn_acc = _safe_mean_bool(gcn_correct[~accepted])
    kept_setta_acc = _safe_mean_bool(setta_correct[~accepted])

    # For sanity checking: most changes should occur inside accepted nodes.
    changed_accepted = (changed & accepted).sum().item()
    changed_kept = (changed & ~accepted).sum().item()

    return {
        "Dataset": dataset_name,
        "Seed": seed,
        "N_test": n_test,
        "GCN_Acc": gcn_correct.float().mean().item() * 100.0,
        "SETTA_Acc": setta_correct.float().mean().item() * 100.0,
        "Delta_Acc": (setta_correct.float().mean().item() - gcn_correct.float().mean().item()) * 100.0,
        "Accepted_Test": accepted_n,
        "Accepted_Test_Ratio": accepted_n / max(n_test, 1) * 100.0,
        "Kept_Test": kept_n,
        "Changed": changed.sum().item(),
        "Changed_Ratio": changed.float().mean().item() * 100.0,
        "Changed_Accepted": changed_accepted,
        "Changed_Kept": changed_kept,
        "Wrong_to_Correct": wrong_to_correct,
        "Correct_to_Wrong": correct_to_wrong,
        "Wrong_to_Wrong": wrong_to_wrong,
        "Correct_to_Correct": correct_to_correct,
        "Net_Corrected": wrong_to_correct - correct_to_wrong,
        "Wrong_to_Correct_Rate": wrong_to_correct / max(n_test, 1) * 100.0,
        "Correct_to_Wrong_Rate": correct_to_wrong / max(n_test, 1) * 100.0,
        "Net_Corrected_Rate": (wrong_to_correct - correct_to_wrong) / max(n_test, 1) * 100.0,
        "Accepted_GCN_Acc": accepted_gcn_acc * 100.0,
        "Accepted_SETTA_Acc": accepted_setta_acc * 100.0,
        "Accepted_Delta_Acc": (accepted_setta_acc - accepted_gcn_acc) * 100.0,
        "Kept_GCN_Acc": kept_gcn_acc * 100.0,
        "Kept_SETTA_Acc": kept_setta_acc * 100.0,
        "Kept_Delta_Acc": (kept_setta_acc - kept_gcn_acc) * 100.0,
        "Used_Steps": used_steps,
        "Stopped_Early": int(stopped_early),
    }


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std summary for all numeric diagnostics by dataset."""
    numeric_cols = [c for c in raw_df.columns if c not in ["Dataset", "Seed"]]
    summary = raw_df.groupby("Dataset")[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary.round(4)


def format_mean_std(summary_df: pd.DataFrame, base_col: str) -> List[str]:
    mean_col = f"{base_col}_mean"
    std_col = f"{base_col}_std"
    return [
        f"{row[mean_col]:.2f} ± {row[std_col]:.2f}"
        for _, row in summary_df.iterrows()
    ]


def build_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compact table suitable for the manuscript."""
    table = pd.DataFrame({"Dataset": summary_df["Dataset"]})
    for col, name in [
        ("GCN_Acc", "GCN Acc (%)"),
        ("SETTA_Acc", "SETTA Acc (%)"),
        ("Delta_Acc", "Delta Acc"),
        ("Accepted_Test_Ratio", "Accepted (%)"),
        ("Wrong_to_Correct_Rate", "Wrong→Correct (%)"),
        ("Correct_to_Wrong_Rate", "Correct→Wrong (%)"),
        ("Net_Corrected_Rate", "Net Corrected (%)"),
        ("Accepted_Delta_Acc", "Accepted ΔAcc"),
    ]:
        table[name] = format_mean_std(summary_df, col)
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="SETTA Wrong->Correct / Correct->Wrong transition analysis")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate. Default: all six datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds. Default: config.SEEDS.",
    )
    parser.add_argument("--out-dir", default="results", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 88)
    print("Exp 6 — SETTA transition analysis: Wrong→Correct / Correct→Wrong")
    print(f"Device  : {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds   : {args.seeds}")
    print(f"Selected alpha/beta: {DEFAULT_VALIDATION_SELECTED_AB_PATH}")
    print("=" * 88)

    records = []

    for dataset_name in args.datasets:
        print(f"\n[{dataset_name}] Loading data and building SETTA structures...")
        ds, data = load_data(dataset_name)
        cfg = get_setta_config(dataset_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)

        sem = build_semantic_adj(data, cfg["k"], cfg["metric"])
        cit = get_cit_adj(data)
        num_props = cfg["num_props"]
        print(
            f"  SETTA config: alpha={cfg['alpha']}, beta={cfg['beta']}, "
            f"k={cfg['k']}, num_props={num_props}"
        )

        for seed in args.seeds:
            print(f"  Seed {seed}")
            gcn_logits = train_gcn(ds, data, seed)

            diag = setta_refine_with_diagnostics(
                gcn_logits,
                data,
                sem,
                cit,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                num_props=num_props,
            )

            rec = transition_record(
                dataset_name=dataset_name,
                seed=seed,
                gcn_logits=gcn_logits,
                setta_logits=diag["final"],
                accepted_mask=diag["accepted_mask"],
                data=data,
                used_steps=int(diag["used_steps"].item()),
                stopped_early=bool(diag["stopped_early"].item()),
            )
            records.append(rec)

            print(
                f"    GCN {rec['GCN_Acc']:.2f} | SETTA {rec['SETTA_Acc']:.2f} | "
                f"Δ {rec['Delta_Acc']:+.2f} | "
                f"W→C {rec['Wrong_to_Correct']} | C→W {rec['Correct_to_Wrong']} | "
                f"Accepted {rec['Accepted_Test_Ratio']:.1f}%"
            )

        # Release dense graph memory before the next dataset.
        del sem, cit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(records)
    summary_df = summarize(raw_df)
    paper_df = build_paper_table(summary_df)

    raw_path = os.path.join(args.out_dir, "exp6_transition_raw.csv")
    summary_path = os.path.join(args.out_dir, "exp6_transition_summary.csv")
    paper_path = os.path.join(args.out_dir, "exp6_transition_paper_table.csv")

    raw_df.round(4).to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {paper_path}")
    print("\nCompact paper table:")
    print(paper_df.to_string(index=False))


if __name__ == "__main__":
    main()
