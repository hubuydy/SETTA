"""
exp7_backbone_setta_transfer.py — SETTA on multiple frozen GNN backbones

Purpose
-------
Evaluate whether SETTA is tied to the GCN backbone or can also improve frozen
predictions from other message-passing GNNs.

Protocol
--------
For each dataset, backbone, and seed:
  1. Train a backbone GNN using the same train/validation/test split protocol.
  2. Freeze its output log-probabilities.
  3. Apply SETTA refinement to the frozen predictions only.
  4. Compare frozen vs. SETTA-refined accuracy on the test mask.

Important notes
---------------
- No test labels are used during SETTA refinement. Test labels are used only for
  final reporting.
- This script uses the fixed dataset-level SETTA structural protocol and
  validation-selected alpha/beta from the configured selection CSV.
  It does NOT re-tune alpha/beta separately for each backbone by default.
  This makes the experiment a stricter backbone-transfer/generalization check.
- SETTA refinement expects log-probabilities as input, matching the existing
  config.dssr_refine implementation, which internally computes exp(logits).

Outputs
-------
  results/exp7_backbone_setta_raw.csv
  results/exp7_backbone_setta_summary.csv
  results/exp7_backbone_setta_paper_table.csv

Example usage
-------------
Run all six datasets, three backbones, all seeds:
  python exp7_backbone_setta_transfer.py

Quick smoke test:
  python exp7_backbone_setta_transfer.py --datasets Cora --backbones GCN GraphSAGE GAT --seeds 42

Run only citation datasets:
  python exp7_backbone_setta_transfer.py --datasets Cora CiteSeer PubMed

Disable cached frozen predictions:
  python exp7_backbone_setta_transfer.py --no-cache
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

from config import (  # noqa: E402
    DEVICE,
    DEFAULT_VALIDATION_SELECTED_AB_PATH,
    NUM_PROPS,
    SEEDS,
    build_semantic_adj,
    dssr_refine,
    eval_acc,
    get_cit_adj,
    get_setta_config,
    load_data,
    set_seed,
    train_gcn,
)
from config import _get_mask  # noqa: E402


ALL_DATASETS = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]
ALL_BACKBONES = ["GCN", "GraphSAGE", "GAT"]


# -----------------------------------------------------------------------------
# Backbone definitions: keep architecture aligned with exp1/exp2.
# -----------------------------------------------------------------------------

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = SAGEConv(in_channels, 64)
        self.c2 = SAGEConv(64, out_channels)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.c2(x, data.edge_index), dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.c2 = GATConv(64, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.c1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(self.c2(x, data.edge_index), dim=1)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_int_list(values: Iterable[str]) -> List[int]:
    return [int(v) for v in values]


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def measure(fn: Callable):
    sync_device()
    t0 = time.perf_counter()
    result = fn()
    sync_device()
    return result, time.perf_counter() - t0


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def mask_accuracy(output: torch.Tensor, data, mask: torch.Tensor) -> float:
    return (output.argmax(dim=1)[mask] == data.y[mask]).float().mean().item() * 100.0


def prediction_change_rate(before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> float:
    changed = before.argmax(dim=1)[mask] != after.argmax(dim=1)[mask]
    return changed.float().mean().item() * 100.0


def probability_l1_shift(before_log_probs: torch.Tensor, after_probs: torch.Tensor, mask: torch.Tensor) -> float:
    before_probs = torch.exp(before_log_probs)
    return (before_probs[mask] - after_probs[mask]).abs().sum(dim=1).mean().item()


# -----------------------------------------------------------------------------
# Training routines
# -----------------------------------------------------------------------------

def train_generic_model(
    model_factory: Callable[[], torch.nn.Module],
    data,
    seed: int,
    split_idx: int = 0,
    epochs: int = 300,
    patience: int = 50,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
) -> torch.Tensor:
    """Train GraphSAGE-style models and return best validation output."""
    set_seed(seed)
    model = model_factory().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)

    best_out = None
    best_score = -float("inf")
    wait = 0
    is_large_val = val_mask.sum().item() > 100

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = mask_accuracy(out_eval, data, val_mask)

        if score > best_score:
            best_score = score
            best_out = out_eval.detach().clone()
            wait = 0
        else:
            wait += 1

        if wait > patience:
            break

    if best_out is None:
        raise RuntimeError("Training failed to produce a validation-best output.")

    return best_out


def train_gat_model(
    dataset,
    data,
    seed: int,
    split_idx: int = 0,
    epochs: int = 300,
    patience: int = 50,
) -> torch.Tensor:
    """Train GAT with the same parameter-group weight decay used in exp1/exp2."""
    set_seed(seed)
    model = GAT(dataset.num_features, dataset.num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(
        [
            {"params": model.c1.parameters(), "weight_decay": 5e-4},
            {"params": model.c2.parameters(), "weight_decay": 0.0},
        ],
        lr=0.005,
    )

    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)

    best_out = None
    best_score = -float("inf")
    wait = 0
    is_large_val = val_mask.sum().item() > 100

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = mask_accuracy(out_eval, data, val_mask)

        if score > best_score:
            best_score = score
            best_out = out_eval.detach().clone()
            wait = 0
        else:
            wait += 1

        if wait > patience:
            break

    if best_out is None:
        raise RuntimeError("GAT training failed to produce a validation-best output.")

    return best_out


def train_backbone(
    backbone: str,
    dataset,
    data,
    seed: int,
    split_idx: int,
    epochs: int,
    patience: int,
) -> torch.Tensor:
    if backbone == "GCN":
        return train_gcn(dataset, data, seed=seed, epochs=epochs, patience=patience, split_idx=split_idx)

    if backbone == "GraphSAGE":
        return train_generic_model(
            lambda: GraphSAGE(dataset.num_features, dataset.num_classes),
            data=data,
            seed=seed,
            split_idx=split_idx,
            epochs=epochs,
            patience=patience,
            lr=0.005,
            weight_decay=5e-4,
        )

    if backbone == "GAT":
        return train_gat_model(
            dataset=dataset,
            data=data,
            seed=seed,
            split_idx=split_idx,
            epochs=epochs,
            patience=patience,
        )

    raise ValueError(f"Unknown backbone: {backbone}")


# -----------------------------------------------------------------------------
# Cache helpers for frozen predictions
# -----------------------------------------------------------------------------

def cache_path(cache_dir: Path, dataset_name: str, backbone: str, seed: int) -> Path:
    safe_name = f"{dataset_name}_{backbone}_seed{seed}.pt"
    return cache_dir / safe_name


def get_or_train_frozen_output(
    args,
    dataset_name: str,
    backbone: str,
    dataset,
    data,
    seed: int,
) -> Tuple[torch.Tensor, float, bool]:
    """Return frozen log-probabilities, training time, and whether cache was used."""
    cp = cache_path(Path(args.cache_dir), dataset_name, backbone, seed)
    if args.use_cache and cp.exists():
        out = torch.load(cp, map_location=DEVICE)
        return out.to(DEVICE), 0.0, True

    out, train_time = measure(
        lambda: train_backbone(
            backbone=backbone,
            dataset=dataset,
            data=data,
            seed=seed,
            split_idx=args.split_idx,
            epochs=args.epochs,
            patience=args.patience,
        )
    )

    if args.use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out.detach().cpu(), cp)

    return out, train_time, False


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def std0(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.std(ddof=0))


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["Dataset", "Backbone"], as_index=False)
    summary = grouped.agg(
        Frozen_Mean=("Frozen_Acc", "mean"),
        Frozen_Std=("Frozen_Acc", std0),
        SETTA_Mean=("SETTA_Acc", "mean"),
        SETTA_Std=("SETTA_Acc", std0),
        Delta_Mean=("Delta", "mean"),
        Delta_Std=("Delta", std0),
        Change_Rate_Mean=("Test_Change_Rate", "mean"),
        Prob_L1_Shift_Mean=("Test_Prob_L1_Shift", "mean"),
        Train_Time_Mean=("Train_Time", "mean"),
        SETTA_Time_Mean=("SETTA_Time", "mean"),
        Num_Seeds=("Seed", "nunique"),
        Improved_Seeds=("Improved", "sum"),
        Tied_Seeds=("Tied", "sum"),
        Worse_Seeds=("Worse", "sum"),
    )

    summary["Win_Rate"] = summary["Improved_Seeds"] / summary["Num_Seeds"]
    summary["Delta_SE"] = summary["Delta_Std"] / np.sqrt(summary["Num_Seeds"].clip(lower=1))
    summary["Delta_CI95"] = 1.96 * summary["Delta_SE"]
    summary["Improves_On_Mean"] = summary["Delta_Mean"] > 0

    numeric_cols = [
        "Frozen_Mean", "Frozen_Std", "SETTA_Mean", "SETTA_Std",
        "Delta_Mean", "Delta_Std", "Change_Rate_Mean", "Prob_L1_Shift_Mean",
        "Train_Time_Mean", "SETTA_Time_Mean", "Win_Rate", "Delta_SE", "Delta_CI95",
    ]
    for col in numeric_cols:
        summary[col] = summary[col].astype(float).round(4)

    return summary.sort_values(["Dataset", "Backbone"]).reset_index(drop=True)


def build_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Backbone": row["Backbone"],
                "Frozen": f"{row['Frozen_Mean']:.2f} ± {row['Frozen_Std']:.2f}",
                "SETTA": f"{row['SETTA_Mean']:.2f} ± {row['SETTA_Std']:.2f}",
                "Δ": f"{row['Delta_Mean']:+.2f}",
                "Wins/Seeds": f"{int(row['Improved_Seeds'])}/{int(row['Num_Seeds'])}",
                "Conclusion": "Improves" if bool(row["Improves_On_Mean"]) else "No mean gain",
            }
        )
    return pd.DataFrame(rows)


def print_compact_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("Exp 7 summary: frozen backbone vs. backbone+SETTA")
    print("=" * 96)
    for dataset_name in summary_df["Dataset"].unique():
        print(f"\n[{dataset_name}]")
        sub = summary_df[summary_df["Dataset"] == dataset_name]
        for _, r in sub.iterrows():
            print(
                f"  {r['Backbone']:10s}  "
                f"Frozen {r['Frozen_Mean']:.2f}±{r['Frozen_Std']:.2f}  |  "
                f"SETTA {r['SETTA_Mean']:.2f}±{r['SETTA_Std']:.2f}  |  "
                f"Δ {r['Delta_Mean']:+.2f}  |  "
                f"wins {int(r['Improved_Seeds'])}/{int(r['Num_Seeds'])}  |  "
                f"changed {r['Change_Rate_Mean']:.2f}%"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, choices=ALL_BACKBONES)
    parser.add_argument("--seeds", nargs="+", default=SEEDS, type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="results/exp7_cache")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 96)
    print("  Exp 7 — Backbone-transfer test: SETTA on GCN / GraphSAGE / GAT frozen predictions")
    print("=" * 96)
    print(f"  Device   : {DEVICE}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Backbones: {args.backbones}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Cache    : {'on' if args.use_cache else 'off'}")
    print(f"  Selected alpha/beta: {DEFAULT_VALIDATION_SELECTED_AB_PATH}")

    raw_records: List[Dict] = []

    for dataset_name in args.datasets:
        print("\n" + "-" * 96)
        print(f"Dataset: {dataset_name}")
        print("-" * 96)

        dataset, data = load_data(dataset_name)
        cfg = get_setta_config(dataset_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
        num_props = cfg["num_props"]

        print(
            f"Building SETTA structures once: "
            f"k={cfg['k']}, metric={cfg['metric']}, alpha={cfg['alpha']}, "
            f"beta={cfg['beta']}, num_props={num_props}"
        )
        (sem, cit), preprocess_time = measure(
            lambda: (
                build_semantic_adj(data, cfg["k"], cfg["metric"]),
                get_cit_adj(data),
            )
        )
        print(f"SETTA one-time structure build: {preprocess_time:.4f}s")

        test_mask = _get_mask(data.test_mask, args.split_idx).bool()

        for backbone in args.backbones:
            print(f"\n  Backbone: {backbone}")
            for seed in args.seeds:
                frozen_out, train_time, used_cache = get_or_train_frozen_output(
                    args=args,
                    dataset_name=dataset_name,
                    backbone=backbone,
                    dataset=dataset,
                    data=data,
                    seed=seed,
                )

                frozen_acc = eval_acc(frozen_out, data, split_idx=args.split_idx) * 100.0

                with torch.no_grad():
                    setta_out, setta_time = measure(
                        lambda: dssr_refine(
                            frozen_out,
                            data,
                            sem,
                            cit,
                            alpha=cfg["alpha"],
                            beta=cfg["beta"],
                            metric=cfg["metric"],
                            num_props=num_props,
                        )
                    )

                setta_acc = eval_acc(setta_out, data, split_idx=args.split_idx) * 100.0
                delta = setta_acc - frozen_acc
                change_rate = prediction_change_rate(frozen_out, setta_out, test_mask)
                prob_shift = probability_l1_shift(frozen_out, setta_out, test_mask)

                raw_records.append(
                    {
                        "Dataset": dataset_name,
                        "Backbone": backbone,
                        "Seed": seed,
                        "Frozen_Acc": frozen_acc,
                        "SETTA_Acc": setta_acc,
                        "Delta": delta,
                        "Improved": int(delta > 1e-12),
                        "Tied": int(abs(delta) <= 1e-12),
                        "Worse": int(delta < -1e-12),
                        "Test_Change_Rate": change_rate,
                        "Test_Prob_L1_Shift": prob_shift,
                        "Alpha": cfg["alpha"],
                        "Beta": cfg["beta"],
                        "K": cfg["k"],
                        "Metric": cfg["metric"],
                        "Num_Props": num_props,
                        "Train_Time": train_time,
                        "SETTA_Time": setta_time,
                        "Preprocess_Time": preprocess_time,
                        "Used_Cache": int(used_cache),
                    }
                )

                cache_tag = " cache" if used_cache else ""
                print(
                    f"    seed {seed}: "
                    f"Frozen={frozen_acc:.2f}, SETTA={setta_acc:.2f}, "
                    f"Δ={delta:+.2f}, changed={change_rate:.2f}%{cache_tag}"
                )

                cleanup()

        # Release dense semantic/citation matrices before next dataset.
        del sem, cit, data, dataset
        cleanup()

    raw_df = pd.DataFrame(raw_records)
    raw_path = Path(args.out_dir) / "exp7_backbone_setta_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize(raw_df)
    summary_path = Path(args.out_dir) / "exp7_backbone_setta_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    paper_table = build_paper_table(summary_df)
    paper_table_path = Path(args.out_dir) / "exp7_backbone_setta_paper_table.csv"
    paper_table.to_csv(paper_table_path, index=False)

    print_compact_summary(summary_df)
    print("\nSaved outputs:")
    print(f"  raw        -> {raw_path}")
    print(f"  summary    -> {summary_path}")
    print(f"  paper table-> {paper_table_path}")


if __name__ == "__main__":
    main()
