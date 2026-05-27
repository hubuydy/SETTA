"""
exp3_postprocessing_onetime_preprocess.py
Section 4.4: Post-processing comparison with SETTA semantic graph construction
reported as one-time dataset-level preprocessing.

Timing protocol used in this script:

1. GCN:
   runtime = GCN training time

2. APPNP:
   runtime = APPNP model training time
   This keeps the same APPNP training protocol as the original experiment.

3. Correct & Smooth:
   runtime = shared GCN training time + C&S correction/smoothing time
   C&S is evaluated on the same frozen GCN output as SETTA for each seed.

4. SETTA:
   one_time_preprocess = semantic graph construction + citation adjacency construction
   runtime = shared GCN training time + SETTA refinement time
   amortized_runtime = shared GCN training time + SETTA refinement time
                       + one_time_preprocess / number_of_seeds

Rationale:
   The semantic graph depends only on the dataset features/topology and SETTA
   hyperparameters, not on random seed or the trained GCN weights. Therefore,
   it is measured once per dataset and reported separately as a one-time
   preprocessing cost instead of being repeatedly charged to every seed.

Important fixes over the original script:
   - Uses one shared frozen GCN output for GCN, C&S, and SETTA under each seed.
   - Reports SETTA semantic graph construction separately as one-time preprocessing.
   - Uses device synchronization for reliable CUDA/MPS timing.
   - Uses time.perf_counter() instead of time.time().
   - Saves raw per-seed results and summary results.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.nn import APPNP
from torch_geometric.nn.models import CorrectAndSmooth

from config import *
from config import _get_mask


# ============================================================
# Timing utilities
# ============================================================

def sync_device():
    """Synchronize CUDA/MPS device before and after timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def warmup_device():
    """Warm up kernels so initialization overhead is not charged to the first method."""
    try:
        sync_device()
        x = torch.randn(256, 256, device=DEVICE)
        for _ in range(5):
            _ = x @ x
        sync_device()
    except Exception:
        pass


def measure(fn):
    """Measure wall-clock time of fn() with device synchronization."""
    sync_device()
    t0 = time.perf_counter()
    result = fn()
    sync_device()
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ============================================================
# APPNP baseline
# ============================================================

class APPNPModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = torch.nn.Linear(in_channels, 64)
        self.l2 = torch.nn.Linear(64, out_channels)
        self.appnp = APPNP(K=10, alpha=0.1)

    def forward(self, data):
        x = F.relu(self.l1(data.x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x)
        x = self.appnp(x, data.edge_index)
        return F.log_softmax(x, dim=1)


def train_appnp(dataset, data, seed=42, epochs=300, patience=50, split_idx=0):
    set_seed(seed)

    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)

    model = APPNPModel(dataset.num_features, dataset.num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_out = None
    best_score = -float("inf")
    wait = 0
    is_large_val = val_mask.sum().item() > 100

    for _ in range(epochs):
        model.train()
        opt.zero_grad()

        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = (
                    out_eval[val_mask].argmax(1) == data.y[val_mask]
                ).float().mean().item()

        if score > best_score:
            best_score = score
            best_out = out_eval.detach().clone()
            wait = 0
        else:
            wait += 1

        if wait > patience:
            break

    return best_out


# ============================================================
# C&S and SETTA applied to the same frozen GCN output
# ============================================================

def run_cs_from_gcn(dataset, data, gcn_log_probs, split_idx=0):
    """Apply Correct & Smooth to a frozen GCN output and return accuracy in %."""
    y_soft = torch.exp(gcn_log_probs).detach().cpu()

    cs = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=1.0,
        num_smoothing_layers=50,
        smoothing_alpha=0.8,
        autoscale=True,
    )

    train_mask = _get_mask(data.train_mask, split_idx).detach().cpu().bool()
    test_mask = _get_mask(data.test_mask, split_idx).detach().cpu().bool()

    y_cpu = data.y.detach().cpu()
    edge_index_cpu = data.edge_index.detach().cpu()
    y_true = F.one_hot(y_cpu[train_mask], dataset.num_classes).float()

    y_corr = cs.correct(y_soft, y_true, train_mask, edge_index_cpu)
    y_sm = cs.smooth(y_corr, y_true, train_mask, edge_index_cpu)

    acc = (y_sm.argmax(1)[test_mask] == y_cpu[test_mask]).float().mean().item()
    return acc * 100.0


def build_setta_preprocess(dataset_name, data):
    """
    Build SETTA dataset-level structures once.

    These structures depend on node features/topology and fixed SETTA hyperparameters,
    not on the trained GCN weights or the random seed.
    """
    cfg = get_setta_config(dataset_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
    sem = build_semantic_adj(data, cfg["k"], cfg["metric"])
    cit = get_cit_adj(data)
    return sem, cit


def run_setta_refine_from_gcn(dataset_name, data, gcn_log_probs, sem, cit):
    """Apply only SETTA refinement to a frozen GCN output."""
    cfg = get_setta_config(dataset_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
    with torch.no_grad():
        out_setta = dssr_refine(
            gcn_log_probs,
            data,
            sem,
            cit,
            cfg["alpha"],
            cfg["beta"],
            metric=cfg["metric"],
            num_props=cfg["num_props"],
        )
    return out_setta


# ============================================================
# Summarization
# ============================================================

def summarize_records(records, num_seeds):
    df = pd.DataFrame(records)

    summary = (
        df.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean=("acc", "mean"),
            std=("acc", "std"),
            train_time_mean=("train_time", "mean"),
            train_time_std=("train_time", "std"),
            post_time_mean=("post_time", "mean"),
            post_time_std=("post_time", "std"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
            one_time_preprocess=("one_time_preprocess", "mean"),
            amortized_runtime_mean=("amortized_runtime", "mean"),
            amortized_runtime_std=("amortized_runtime", "std"),
        )
    )

    summary = summary.fillna(0.0)

    # Backward-compatible names for old plotting scripts.
    # time_mean excludes one-time preprocessing, because that is reported separately.
    summary["time_mean"] = summary["runtime_mean"]
    summary["time_std"] = summary["runtime_std"]

    # The one-time preprocessing cost per seed, useful if a reviewer asks for amortized cost.
    summary["preprocess_per_seed"] = summary["one_time_preprocess"] / float(num_seeds)

    numeric_cols = [
        "mean", "std",
        "train_time_mean", "train_time_std",
        "post_time_mean", "post_time_std",
        "runtime_mean", "runtime_std",
        "one_time_preprocess",
        "amortized_runtime_mean", "amortized_runtime_std",
        "time_mean", "time_std",
        "preprocess_per_seed",
    ]
    for col in numeric_cols:
        summary[col] = summary[col].round(4)

    return df, summary


# ============================================================
# Main experiment
# ============================================================

def main():
    os.makedirs("results", exist_ok=True)

    datasets = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]
    methods_print_order = ["GCN", "APPNP", "C&S", "SETTA"]
    num_seeds = len(SEEDS)

    print("=" * 92)
    print("  Exp 3 — Section 4.4: Post-processing Comparison")
    print("  SETTA semantic graph construction is reported as one-time preprocessing")
    print("=" * 92)
    print(f"  Device: {DEVICE}")
    print(f"  Seeds : {SEEDS}")
    print(f"  Validation-selected alpha/beta: {DEFAULT_VALIDATION_SELECTED_AB_PATH}")

    all_records = []
    preprocess_summary = {}

    for ds_name in datasets:
        print(f"\n[{ds_name}] Loading data...")
        ds, data = load_data(ds_name)
        cfg = get_setta_config(ds_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
        print(
            f"  SETTA config: alpha={cfg['alpha']}, beta={cfg['beta']}, "
            f"k={cfg['k']}, num_props={cfg['num_props']}"
        )
        warmup_device()

        # ------------------------------------------------------------
        # One-time SETTA preprocessing per dataset
        # ------------------------------------------------------------
        print(f"  Building SETTA semantic/topology structures once for {ds_name}...")
        (sem, cit), setta_preprocess_time = measure(
            lambda: build_setta_preprocess(ds_name, data)
        )
        preprocess_summary[ds_name] = setta_preprocess_time
        print(f"  SETTA one-time preprocessing: {setta_preprocess_time:.4f}s")

        for seed in SEEDS:
            print(f"  Seed {seed}")

            # --------------------------------------------------------
            # Shared GCN output for GCN / C&S / SETTA
            # --------------------------------------------------------
            gcn_out, gcn_train_time = measure(
                lambda: train_gcn(ds, data, seed)
            )
            gcn_acc = eval_acc(gcn_out, data) * 100.0

            all_records.append({
                "dataset": ds_name,
                "seed": seed,
                "method": "GCN",
                "acc": gcn_acc,
                "train_time": gcn_train_time,
                "post_time": 0.0,
                "runtime": gcn_train_time,
                "one_time_preprocess": 0.0,
                "amortized_runtime": gcn_train_time,
            })

            # --------------------------------------------------------
            # Correct & Smooth on the same frozen GCN output
            # --------------------------------------------------------
            cs_acc, cs_post_time = measure(
                lambda: run_cs_from_gcn(ds, data, gcn_out)
            )

            all_records.append({
                "dataset": ds_name,
                "seed": seed,
                "method": "C&S",
                "acc": cs_acc,
                "train_time": gcn_train_time,
                "post_time": cs_post_time,
                "runtime": gcn_train_time + cs_post_time,
                "one_time_preprocess": 0.0,
                "amortized_runtime": gcn_train_time + cs_post_time,
            })

            # --------------------------------------------------------
            # SETTA refinement on the same frozen GCN output
            # One-time semantic graph construction is NOT charged here.
            # It is stored separately in one_time_preprocess.
            # --------------------------------------------------------
            setta_out, setta_refine_time = measure(
                lambda: run_setta_refine_from_gcn(ds_name, data, gcn_out, sem, cit)
            )
            setta_acc = eval_acc(setta_out, data) * 100.0
            setta_runtime = gcn_train_time + setta_refine_time
            setta_amortized_runtime = setta_runtime + setta_preprocess_time / float(num_seeds)

            all_records.append({
                "dataset": ds_name,
                "seed": seed,
                "method": "SETTA",
                "acc": setta_acc,
                "train_time": gcn_train_time,
                "post_time": setta_refine_time,
                "runtime": setta_runtime,
                "one_time_preprocess": setta_preprocess_time,
                "amortized_runtime": setta_amortized_runtime,
            })

            # --------------------------------------------------------
            # APPNP trained as a separate model
            # --------------------------------------------------------
            appnp_out, appnp_train_time = measure(
                lambda: train_appnp(ds, data, seed)
            )
            appnp_acc = eval_acc(appnp_out, data) * 100.0

            all_records.append({
                "dataset": ds_name,
                "seed": seed,
                "method": "APPNP",
                "acc": appnp_acc,
                "train_time": appnp_train_time,
                "post_time": 0.0,
                "runtime": appnp_train_time,
                "one_time_preprocess": 0.0,
                "amortized_runtime": appnp_train_time,
            })

        # Dataset-level temporary summary
        _, temp_summary = summarize_records(
            [r for r in all_records if r["dataset"] == ds_name],
            num_seeds=num_seeds,
        )

        print(f"\n  Summary for {ds_name}")
        print(f"  SETTA one-time preprocessing: {setta_preprocess_time:.4f}s")
        for method in methods_print_order:
            row = temp_summary[temp_summary["method"] == method].iloc[0]
            if method == "SETTA":
                print(
                    f"    {method:6s} "
                    f"Acc {row['mean']:.2f} ± {row['std']:.2f} | "
                    f"Runtime {row['runtime_mean']:.4f}s | "
                    f"Train {row['train_time_mean']:.4f}s | "
                    f"Refine {row['post_time_mean']:.4f}s | "
                    f"One-time prep {row['one_time_preprocess']:.4f}s | "
                    f"Amortized {row['amortized_runtime_mean']:.4f}s"
                )
            else:
                print(
                    f"    {method:6s} "
                    f"Acc {row['mean']:.2f} ± {row['std']:.2f} | "
                    f"Runtime {row['runtime_mean']:.4f}s | "
                    f"Train {row['train_time_mean']:.4f}s | "
                    f"Post {row['post_time_mean']:.4f}s"
                )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    raw_df, summary_df = summarize_records(all_records, num_seeds=num_seeds)

    raw_path = "results/table4_postprocessing_onetime_preprocess_raw.csv"
    summary_path = "results/table4_postprocessing_onetime_preprocess.csv"
    legacy_path = "results/table3_postprocessing.csv"
    preprocess_path = "results/setta_onetime_preprocess.csv"
    json_path = "results/efficiency_time_onetime_preprocess.json"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Optional compatibility for existing plotting scripts.
    summary_df.to_csv(legacy_path, index=False)

    preprocess_df = pd.DataFrame([
        {"dataset": k, "setta_one_time_preprocess": round(v, 4)}
        for k, v in preprocess_summary.items()
    ])
    preprocess_df.to_csv(preprocess_path, index=False)

    timing = {}
    for _, row in summary_df.iterrows():
        key = f"{row['dataset']}_{row['method']}"
        timing[key] = {
            "acc_mean": float(row["mean"]),
            "acc_std": float(row["std"]),
            "train_time_mean": float(row["train_time_mean"]),
            "train_time_std": float(row["train_time_std"]),
            "post_time_mean": float(row["post_time_mean"]),
            "post_time_std": float(row["post_time_std"]),
            "runtime_mean_excluding_one_time_preprocess": float(row["runtime_mean"]),
            "runtime_std_excluding_one_time_preprocess": float(row["runtime_std"]),
            "one_time_preprocess": float(row["one_time_preprocess"]),
            "preprocess_per_seed": float(row["preprocess_per_seed"]),
            "amortized_runtime_mean": float(row["amortized_runtime_mean"]),
            "amortized_runtime_std": float(row["amortized_runtime_std"]),
        }

    with open(json_path, "w") as f:
        json.dump(timing, f, indent=2)

    print("\n" + "=" * 92)
    print("Saved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {legacy_path}")
    print(f"  {preprocess_path}")
    print(f"  {json_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
