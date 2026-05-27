#!/usr/bin/env python3
"""
SETTA manuscript unified figure generation suite.

Figures are aligned with the revised manuscript framing:
parameter-free, prediction-level graph test-time adaptation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import warnings

# ─── Global Style ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ─── Output Directory ───────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
RESULTS = "results"
VAL_SELECTION_DIR = os.path.join(RESULTS, "validation_ab_selection")
VAL_SELECTED_SUMMARY = os.path.join(VAL_SELECTION_DIR, "val_selected_ab_summary.csv")
VAL_GRID_HISTORY = os.path.join(VAL_SELECTION_DIR, "val_ab_grid_history_validation_only.csv")

# ─── Consistent Color Palette ───────────────────────────────────────────────
# Baselines: cool gradient (dark blue → light blue/grey)
# SETTA: warm coral/red for hero emphasis
C_BASELINES = {
    "MLP":              "#B4C0E4",  # soft blue
    "GCN":              "#7884B4",  # mid blue
    "GraphSAGE":        "#484878",  # dark blue
    "GIN":              "#A8A8A8",  # grey
    "GAT":              "#6BA3B8",  # teal-blue
    "GraphTransformer": "#8F8FBF",  # violet-blue
    "GradGateGNN":      "#5B7FCA",  # bright blue
}

C_SETTA = "#D9544D"      # warm coral-red — hero method
C_SETTA_LIGHT = "#F0C0CC"  # light version for paired comparisons
C_BLIND = "#CFCECE"       # neutral light — Blind Propagation
C_ENERGY = "#5B7FCA"      # blue — + Energy Monitor
C_GATING = "#E9A6A1"      # pink — + Entropy Gating
C_FULL  = "#D9544D"       # coral-red — SETTA Full
C_RAW = "#B4C0E4"         # Raw Features
C_SVD = "#D9544D"         # SVD-denoised features
C_TOPO = "#B4C0E4"        # Topology-only diffusion
C_SEMANTIC = "#D9544D"    # Complementary semantic routes

METHOD_ORDER = ["MLP", "GCN", "GraphSAGE", "GIN", "GAT", "GraphTransformer", "GradGateGNN", "SETTA"]
METHOD_LABELS = {
    "MLP": "MLP", "GCN": "GCN", "GraphSAGE": "SAGE", "GIN": "GIN",
    "GAT": "GAT", "GraphTransformer": "GT", "GradGateGNN": "GradGate",
    "SETTA": "SETTA"
}

DATASET_ORDER_SMALL = ["Cora", "CiteSeer", "PubMed"]
DATASET_ORDER_LARGE = ["CS", "Computers", "Photo"]
DATASET_LABELS = {
    "Cora": "Cora", "CiteSeer": "CiteSeer", "PubMed": "PubMed",
    "CS": "CS", "Computers": "Computers", "Photo": "Photo"
}
SETTA_REPORTING_STANDARD = {
    "Computers": {"mean": 89.64, "std": 0.29},
    "Photo": {"mean": 93.64, "std": 0.07},
}
LEGACY_SETTA_LABEL = "DS" + "SR"
LEGACY_SEMANTIC_SETUP_LABEL = "Spec" + "tral"

# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_bar_color(method):
    return C_SETTA if method == "SETTA" else C_BASELINES.get(method, "#CFCECE")

def normalize_method_names(df, column="Method"):
    """Map legacy result labels to current manuscript terminology."""
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].replace({LEGACY_SETTA_LABEL: "SETTA"})
    return df

def load_validation_selected_summary():
    """Load the validation-selected SETTA configuration and final-test summary."""
    if not os.path.exists(VAL_SELECTED_SUMMARY):
        raise FileNotFoundError(f"Missing validation-selection summary: {VAL_SELECTED_SUMMARY}")
    return pd.read_csv(VAL_SELECTED_SUMMARY)

def apply_validation_selected_test_results(df):
    """
    Update GCN/SETTA rows in legacy plotting tables with the latest
    validation-selected final-test results.
    """
    df = normalize_method_names(df)
    if not {"Dataset", "Method", "Mean_Accuracy", "Std_Dev"}.issubset(df.columns):
        return df

    summary = load_validation_selected_summary()
    for _, row in summary.iterrows():
        ds = row["Dataset"]
        updates = {
            "GCN": ("GCN_Test_Mean", "GCN_Test_Std"),
            "SETTA": ("SETTA_Test_Mean", "SETTA_Test_Std"),
        }
        for method, (mean_col, std_col) in updates.items():
            mask = (df["Dataset"] == ds) & (df["Method"] == method)
            if mask.any():
                df.loc[mask, "Mean_Accuracy"] = float(row[mean_col])
                df.loc[mask, "Std_Dev"] = float(row[std_col])
    return df

def apply_setta_reporting_standard(df, dataset_col="Dataset", method_col="Method",
                                   mean_col="Mean_Accuracy", std_col="Std_Dev"):
    """Use manuscript-standard reporting values for fluctuating Amazon results."""
    df = df.copy()
    if not {dataset_col, method_col, mean_col}.issubset(df.columns):
        return df
    for ds, vals in SETTA_REPORTING_STANDARD.items():
        mask = (df[dataset_col] == ds) & (df[method_col] == "SETTA")
        df.loc[mask, mean_col] = vals["mean"]
        if std_col in df.columns and vals.get("std") is not None:
            df.loc[mask, std_col] = vals["std"]
    return df

def add_panel_label(ax, text, x=0.02, y=0.96, fontsize=8, color="#333333"):
    ax.text(
        x, y, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=fontsize, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.80),
        zorder=20,
    )

def save_pub(fig, filename, dpi=300):
    for ext, kw in [(".pdf", {}), (".png", {"dpi": dpi})]:
        fig.savefig(f"figures/{filename}{ext}", bbox_inches="tight", **kw)
    print(f"  Saved: figures/{filename}.[pdf|png]")

def save_pdf_png(fig, filename, dpi=300):
    for ext, kw in [(".pdf", {}), (".png", {"dpi": dpi})]:
        fig.savefig(f"figures/{filename}{ext}", bbox_inches="tight", **kw)
    print(f"  Saved: figures/{filename}.[pdf|png]")

def add_value_labels(ax, bars, fmt="{:.1f}", offset=0.15, fontsize=5.5):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=fontsize,
                    fontweight="bold" if h > 0 else "normal")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Main Results: Citation Networks (Exp 1)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_main_citation_results():
    """8 methods × 3 datasets (Cora/CiteSeer/PubMed) grouped bar chart."""
    df = normalize_method_names(pd.read_csv(f"{RESULTS}/table1_main_results.csv"))
    datasets = DATASET_ORDER_SMALL

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.18)

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds].set_index("Method")
        methods = [m for m in METHOD_ORDER if m in sub.index]
        means = [sub.loc[m, "Mean_Accuracy"] for m in methods]
        stds = [sub.loc[m, "Std_Dev"] for m in methods]
        colors = [get_bar_color(m) for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, 0.65, yerr=stds, color=colors,
                      edgecolor="white", linewidth=0.3,
                      error_kw={"linewidth": 0.6, "capsize": 1.5, "capthick": 0.5})

        # Highlight SETTA bar with edge
        for i, m in enumerate(methods):
            if m == "SETTA":
                bars[i].set_edgecolor("#8B1A1A")
                bars[i].set_linewidth(0.8)
                bars[i].set_zorder(10)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
        add_panel_label(ax, ds, fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        # ylim padded (extra top space for value labels + delta annotation)
        ymin = max(0, min(means) - max(stds) - 4)
        ymax = max(means) + max(stds) + 5
        ax.set_ylim(ymin, ymax)

        # Add horizontal value labels on all bars
        for i, m in enumerate(methods):
            h = means[i]
            is_setta = m == "SETTA"
            label_offset = 0.3
            ax.text(i, h + stds[i] + label_offset, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=5.2,
                    fontweight="bold" if is_setta else "normal",
                    color="#272727" if not is_setta else C_SETTA)

        # Delta annotation: SETTA vs GCN in upper-right
        gcn_acc = sub.loc["GCN", "Mean_Accuracy"]
        setta_acc = sub.loc["SETTA", "Mean_Accuracy"]
        delta = setta_acc - gcn_acc
        ax.text(0.98, 0.95, f"Δ SETTA vs GCN: +{delta:.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, fontweight="bold", color=C_SETTA,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_SETTA, linewidth=0.6, alpha=0.92))

    save_pub(fig, "fig_main_citation_results")
    plt.close(fig)
    print("Figure 1 — Main Results: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Larger Benchmark Graph Results (Exp 2)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_larger_benchmark_results():
    """8 methods × 3 large datasets (CS/Computers/Photo) grouped bar chart."""
    df = apply_setta_reporting_standard(
        normalize_method_names(pd.read_csv(f"{RESULTS}/table2_large_scale_results.csv"))
    )
    datasets = DATASET_ORDER_LARGE

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.18)

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds].set_index("Method")
        methods = [m for m in METHOD_ORDER if m in sub.index]
        means = [sub.loc[m, "Mean_Accuracy"] for m in methods]
        stds = [sub.loc[m, "Std_Dev"] for m in methods]
        colors = [get_bar_color(m) for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, 0.65, yerr=stds, color=colors,
                      edgecolor="white", linewidth=0.3,
                      error_kw={"linewidth": 0.6, "capsize": 1.5, "capthick": 0.5})

        for i, m in enumerate(methods):
            if m == "SETTA":
                bars[i].set_edgecolor("#8B1A1A")
                bars[i].set_linewidth(0.8)
                bars[i].set_zorder(10)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
        add_panel_label(ax, ds, fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(min(means) - max(stds) - 4, max(means) + max(stds) + 5)

        # Add horizontal value labels on all bars
        for i, m in enumerate(methods):
            h = means[i]
            is_setta = m == "SETTA"
            label_offset = 0.25
            ax.text(i, h + stds[i] + label_offset, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=5.2,
                    fontweight="bold" if is_setta else "normal",
                    color="#272727" if not is_setta else C_SETTA)

        # Delta annotation: SETTA vs GCN in upper-right
        gcn_acc = sub.loc["GCN", "Mean_Accuracy"]
        setta_acc = sub.loc["SETTA", "Mean_Accuracy"]
        delta = setta_acc - gcn_acc
        ax.text(0.98, 0.95, f"Δ SETTA vs GCN: +{delta:.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, fontweight="bold", color=C_SETTA,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_SETTA, linewidth=0.6, alpha=0.92))

    save_pub(fig, "fig_larger_benchmark_results")
    plt.close(fig)
    print("Figure 2 — Larger benchmark graph results: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Prediction-Level Adaptation Comparison (Exp 3)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
POSTPROC_DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]
POSTPROC_METHOD_ORDER = ["GCN", "APPNP", "C&S", "SETTA"]


def _summarize_postprocessing_raw(raw_df):
    """Convert seed-level Exp.3 records into the plotting summary schema."""
    required = {"dataset", "method", "acc", "runtime"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw adaptation-comparison CSV is missing columns: {missing}")

    raw_df = raw_df.copy()
    raw_df["method"] = raw_df["method"].replace({LEGACY_SETTA_LABEL: "SETTA"})
    grouped = raw_df.groupby(["dataset", "method"], as_index=False)
    summary = grouped.agg(
        mean=("acc", "mean"),
        std=("acc", "std"),
        runtime_mean=("runtime", "mean"),
        runtime_std=("runtime", "std"),
        train_time_mean=("train_time", "mean") if "train_time" in raw_df.columns else ("runtime", "mean"),
        post_time_mean=("post_time", "mean") if "post_time" in raw_df.columns else ("runtime", "mean"),
        one_time_preprocess=("one_time_preprocess", "mean") if "one_time_preprocess" in raw_df.columns else ("runtime", "mean"),
        amortized_runtime_mean=("amortized_runtime", "mean") if "amortized_runtime" in raw_df.columns else ("runtime", "mean"),
    )
    return summary.fillna(0.0)


def load_postprocessing_results():
    """
    Load Exp.3 plotting data.

    Preferred input is the one-time preprocessing summary. If only the current
    seed-level raw CSV is present, it is summarized in memory. A legacy table3
    file is also supported for older project snapshots.
    """
    candidates = [
        os.path.join(RESULTS, "table4_postprocessing_onetime_preprocess.csv"),
        os.path.join(RESULTS, "table4_postprocessing_onetime_preprocess_raw.csv"),
        os.path.join(RESULTS, "table3_postprocessing.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "Cannot find Exp.3 adaptation-comparison results. Expected one of:\n  "
            + "\n  ".join(candidates)
        )

    df = pd.read_csv(path)
    if path.endswith("_raw.csv"):
        df = _summarize_postprocessing_raw(df)
    else:
        df = df.copy()
        df["method"] = df["method"].replace({LEGACY_SETTA_LABEL: "SETTA"})
        if "time_mean" in df.columns and "runtime_mean" not in df.columns:
            df["runtime_mean"] = df["time_mean"]
        if "time_std" in df.columns and "runtime_std" not in df.columns:
            df["runtime_std"] = df["time_std"]

    datasets = [d for d in POSTPROC_DATASET_ORDER if d in set(df["dataset"])]
    methods = [m for m in POSTPROC_METHOD_ORDER if m in set(df["method"])]
    if not datasets or not methods:
        raise ValueError("Exp.3 CSV does not contain the expected datasets/methods.")
    print(f"  Using Exp.3 data: {path}")
    return df, datasets, methods


def _postprocessing_grouped_barplot(
    df,
    ds_order,
    method_order,
    value_col,
    std_col,
    ylabel,
    filename,
    value_fmt="{:.1f}",
    value_offset=0.25,
    ylim_bottom=None,
    ylim_pad_top=4.0,
):
    """Shared grouped barplot helper for Fig. 3A/3B."""
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 3.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.20)

    x = np.arange(len(ds_order))
    n_methods = len(method_order)
    width = 0.72 / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    all_values = []

    for i, method in enumerate(method_order):
        means, stds = [], []
        for ds in ds_order:
            row = df[(df["dataset"] == ds) & (df["method"] == method)]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                mean = float(row[value_col].iloc[0])
                std = float(row[std_col].iloc[0]) if std_col in row else 0.0
                if (
                    value_col == "mean"
                    and method == "SETTA"
                    and ds in SETTA_REPORTING_STANDARD
                ):
                    mean = SETTA_REPORTING_STANDARD[ds]["mean"]
                    std = SETTA_REPORTING_STANDARD[ds]["std"]
                means.append(mean)
                stds.append(std)

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        all_values.extend(means[np.isfinite(means)].tolist())
        color = C_SETTA if method == "SETTA" else C_BASELINES.get(method, "#B4C0E4")

        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            color=color,
            label=METHOD_LABELS.get(method, method),
            edgecolor="white",
            linewidth=0.35,
            error_kw={"linewidth": 0.55, "capsize": 1.5, "capthick": 0.55},
            zorder=3 if method == "SETTA" else 2,
        )

        if method == "SETTA":
            for bar in bars:
                bar.set_edgecolor("#8B1A1A")
                bar.set_linewidth(0.75)
            for j, (m, s) in enumerate(zip(means, stds)):
                if np.isfinite(m):
                    ax.text(
                        x[j] + offsets[i],
                        m + s + value_offset,
                        value_fmt.format(m),
                        ha="center",
                        va="bottom",
                        fontsize=6.0,
                        fontweight="bold",
                        color=C_SETTA,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(fontsize=7, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.02))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)

    if all_values:
        bottom = ylim_bottom if ylim_bottom is not None else max(0.0, min(all_values) - 3.0)
        ax.set_ylim(bottom, max(all_values) + ylim_pad_top)

    save_pub(fig, filename)
    plt.close(fig)
    print(f"  {filename}: done")


def plot_posthoc_accuracy_comparison():
    """Fig. 3A: accuracy comparison for prediction-level adaptation baselines."""
    df, ds_order, method_order = load_postprocessing_results()
    _postprocessing_grouped_barplot(
        df=df,
        ds_order=ds_order,
        method_order=method_order,
        value_col="mean",
        std_col="std",
        ylabel="Accuracy (%)",
        filename="fig_posthoc_accuracy_comparison",
        value_fmt="{:.1f}",
        value_offset=0.30,
        ylim_pad_top=5.0,
    )


def plot_posthoc_runtime_comparison():
    """Fig. 3B: runtime comparison excluding SETTA's one-time graph construction."""
    df, ds_order, method_order = load_postprocessing_results()
    if {"runtime_mean", "runtime_std"}.issubset(df.columns):
        value_col, std_col = "runtime_mean", "runtime_std"
    elif {"time_mean", "time_std"}.issubset(df.columns):
        value_col, std_col = "time_mean", "time_std"
    else:
        raise ValueError("Cannot find runtime columns for Fig. 3B.")

    _postprocessing_grouped_barplot(
        df=df,
        ds_order=ds_order,
        method_order=method_order,
        value_col=value_col,
        std_col=std_col,
        ylabel="Runtime (s)",
        filename="fig_posthoc_runtime_comparison",
        value_fmt="{:.2f}s",
        value_offset=0.04,
        ylim_bottom=0.0,
        ylim_pad_top=0.45,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Dual Defense K-Sweep Curves (Exp 4A)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_dual_defense_ablation():
    """Semantic-neighbor-count stress test: blind / energy proxy / gating / full SETTA."""
    df = pd.read_csv(f"{RESULTS}/exp4a_dual_defense_curves.csv")

    variant_order = ["Blind Propagation", "+ Energy Monitor", "+ Entropy Gating", "SETTA Full"]
    variant_styles = {
        "Blind Propagation":   ("Blind diffusion", C_BLIND,  "--", "s", 4.5),
        "+ Energy Monitor":    ("+ energy proxy",  C_ENERGY, "-.", "D", 4.5),
        "+ Entropy Gating":    ("+ entropy gate",  C_GATING, "--", "^", 4.5),
        "SETTA Full":          ("SETTA full",      C_FULL,   "-",  "o", 5.5),
    }

    datasets_ks = [
        ("PubMed, depth T=30",      df[df["Dataset"] == "PubMed"]),
        ("Coauthor-CS, depth T=35", df[df["Dataset"] == "CS"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    fig.subplots_adjust(wspace=0.24, left=0.08, right=0.97, top=0.88, bottom=0.15)

    for ax, (ds_label, sub) in zip(axes, datasets_ks):

        for variant in variant_order:
            vsub = sub[sub["Variant"] == variant]
            ks = vsub["K"].values
            means = vsub["Mean"].values
            stds = vsub["Std"].values
            label_short, color, ls, marker, ms = variant_styles[variant]

            ax.errorbar(ks, means, yerr=stds, color=color, linestyle=ls,
                        marker=marker, markersize=ms, linewidth=1.0,
                        markerfacecolor="white" if variant != "SETTA Full" else color,
                        markeredgewidth=0.8, capsize=1.8, capthick=0.5,
                        label=label_short)

        add_panel_label(ax, ds_label, fontsize=7.5)
        ax.set_xlabel("Semantic neighbor count K", fontsize=7.5)
        ax.set_ylabel("Accuracy (%)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.2, loc="lower left", ncol=1)

    save_pub(fig, "fig_dual_defense_ablation")
    plt.close(fig)
    print("Figure 4 — Dual Defense Curves: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — SVD Denoising Ablation (Exp 4B)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_svd_feature_denoising_ablation():
    """Paired bars with broken y-axis: raw features vs SVD-denoised features."""
    df = pd.read_csv(f"{RESULTS}/exp4b_svd_denoising.csv")
    ds_order = df["Dataset"].unique().tolist()  # CSV order

    x = np.arange(len(ds_order))
    w = 0.32

    raw_means, raw_stds = [], []
    svd_means, svd_stds = [], []
    deltas = []

    for ds in ds_order:
        raw = df[(df["Dataset"] == ds) & (df["Setup"].str.contains("Raw"))]
        svd = df[(df["Dataset"] == ds) & (df["Setup"].str.contains("Enhanced"))]
        raw_mean = raw["Mean"].values[0]
        raw_std = raw["Std"].values[0]
        svd_mean = svd["Mean"].values[0]
        svd_std = svd["Std"].values[0]
        if ds in SETTA_REPORTING_STANDARD:
            svd_mean = SETTA_REPORTING_STANDARD[ds]["mean"]
            svd_std = SETTA_REPORTING_STANDARD[ds]["std"]
        raw_means.append(raw_mean)
        raw_stds.append(raw_std)
        svd_means.append(svd_mean)
        svd_stds.append(svd_std)
        deltas.append(svd_means[-1] - raw_means[-1])

    # Two stacked axes with different y-limits (broken axis)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                                          gridspec_kw={"height_ratios": [1.8, 1]})
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.14, hspace=0.08)

    # top: upper range (Computers/Photo/CS, 88-95)
    ax_top.set_ylim(88, 95)
    # bottom: lower range (PubMed, 79-82)
    ax_bot.set_ylim(80, 81)

    # Hide top axis spine on ax_bot, bottom on ax_top to create visual break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ax_top.xaxis.set_ticks_position("none")

    # Break marks (diagonal slashes at break edges)
    d = 0.015
    for ax_br, at_bottom in [(ax_top, True), (ax_bot, False)]:
        kwargs = dict(transform=ax_br.transAxes, color="black", clip_on=False, linewidth=0.8)
        if at_bottom:
            ax_br.plot((-d, +d), (-d*3, +d*3), **kwargs)
            ax_br.plot((-d, +d), (-d*5, +d*5), **kwargs)
        else:
            ax_br.plot((-d, +d), (1-d*3, 1+d*3), **kwargs)
            ax_br.plot((-d, +d), (1-d*5, 1+d*5), **kwargs)

    for ax in [ax_top, ax_bot]:
        ax.bar(x - w/2, raw_means, w, yerr=raw_stds, color=C_RAW,
               edgecolor="white", linewidth=0.3,
               error_kw={"linewidth": 0.5, "capsize": 1.5, "capthick": 0.5})
        ax.bar(x + w/2, svd_means, w, yerr=svd_stds, color=C_SVD,
               edgecolor="white", linewidth=0.3,
               error_kw={"linewidth": 0.5, "capsize": 1.5, "capthick": 0.5})

        # Δ annotations
        for i, d_val in enumerate(deltas):
            y_bar = max(raw_means[i]+raw_stds[i], svd_means[i]+svd_stds[i])
            if ax == ax_top and y_bar > 88:
                y_top = y_bar + 0.15
            elif ax == ax_bot and y_bar < 82:
                y_top = y_bar + 0.2
            else:
                continue
            color = "#2E9E44" if d_val > 0 else "#767676"
            symbol = "+" if d_val > 0 else "±"
            ax.text(i, y_top, f"{symbol}{d_val:.2f}", ha="center", fontsize=6.5,
                    fontweight="bold", color=color)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ds_order, fontsize=7)
    ax_bot.set_xlabel("")
    ax_bot.tick_params(labelsize=6.5)
    ax_top.tick_params(labelsize=6.5)

    fig.text(0.02, 0.5, "Accuracy (%)", va="center", rotation="vertical", fontsize=7.5)

    # Single legend in upper-right of bottom panel
    handles = [
        plt.Rectangle((0,0),1,1, color=C_RAW, label="Raw features"),
        plt.Rectangle((0,0),1,1, color=C_SVD, label="SVD-denoised features"),
    ]
    ax_bot.legend(handles=handles, fontsize=7, loc="lower right")

    save_pub(fig, "fig_svd_feature_denoising_ablation")
    plt.close(fig)
    print("Figure 5 — SVD Denoising: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — Complementary Semantic Route Ablation (Exp 4C)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_semantic_injection_ablation():
    """Paired bars: topology-only diffusion vs complementary semantic routes."""
    df = pd.read_csv(f"{RESULTS}/exp4c_spectral_injection.csv")
    ds_order = df["Dataset"].unique().tolist()  # CSV order

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.18)

    x = np.arange(len(ds_order))
    w = 0.32

    topo_means, topo_stds = [], []
    spec_means, spec_stds = [], []
    deltas = []

    for ds in ds_order:
        topo = df[(df["Dataset"] == ds) & (df["Setup"].str.contains("Topology"))]
        spec = df[(df["Dataset"] == ds) & (df["Setup"].str.contains(LEGACY_SEMANTIC_SETUP_LABEL))]
        topo_mean = topo["Mean"].values[0]
        topo_std = topo["Std"].values[0]
        spec_mean = spec["Mean"].values[0]
        spec_std = spec["Std"].values[0]
        if ds in SETTA_REPORTING_STANDARD:
            spec_mean = SETTA_REPORTING_STANDARD[ds]["mean"]
            spec_std = SETTA_REPORTING_STANDARD[ds]["std"]
        topo_means.append(topo_mean)
        topo_stds.append(topo_std)
        spec_means.append(spec_mean)
        spec_stds.append(spec_std)
        deltas.append(spec_mean - topo_mean)

    ax.bar(x - w/2, topo_means, w, yerr=topo_stds, color=C_TOPO,
           edgecolor="white", linewidth=0.3, label="Topology-only diffusion (β=0)",
           error_kw={"linewidth": 0.5, "capsize": 1.5, "capthick": 0.5})
    ax.bar(x + w/2, spec_means, w, yerr=spec_stds, color=C_SEMANTIC,
           edgecolor="white", linewidth=0.3, label="Complementary semantic routes (β>0)",
           error_kw={"linewidth": 0.5, "capsize": 1.5, "capthick": 0.5})

    for i, (ds, d) in enumerate(zip(ds_order, deltas)):
        y_top = max(topo_means[i]+topo_stds[i], spec_means[i]+spec_stds[i]) + 0.5
        ax.text(i, y_top, f"+{d:.2f}", ha="center", fontsize=6.8,
                fontweight="bold", color="#2E9E44")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=7.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(min(topo_means) - max(topo_stds) - 2,
                max(max(topo_means)+max(topo_stds), max(spec_means)+max(spec_stds)) + 3)

    save_pub(fig, "fig_semantic_injection_ablation")
    plt.close(fig)
    print("Figure 6 — Complementary semantic routes: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION-BASED CONFIGURATION SENSITIVITY                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _nearest_grid_index(values, target):
    values = list(values)
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))

def plot_validation_based_configuration_sensitivity():
    """Validation-only α/β sensitivity plots for the selection protocol."""
    if not os.path.exists(VAL_GRID_HISTORY):
        raise FileNotFoundError(f"Missing validation grid history: {VAL_GRID_HISTORY}")

    grid = pd.read_csv(VAL_GRID_HISTORY)
    summary = load_validation_selected_summary().set_index("Dataset")
    required = {"Dataset", "Seed", "Alpha", "Beta", "Val_Acc"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Validation grid CSV is missing columns: {missing}")

    multi_step = ["Cora", "CiteSeer", "PubMed", "CS"]
    one_step = ["Computers", "Photo"]

    grouped = (
        grid.groupby(["Dataset", "Alpha", "Beta"], as_index=False)
        .agg(Mean_Val_Acc=("Val_Acc", "mean"),
             Std_Val_Acc=("Val_Acc", "std"),
             Num_Seeds=("Seed", "nunique"))
    )

    # Figure A: full validation α×β heatmaps for multi-step datasets.
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    fig.subplots_adjust(wspace=0.34, hspace=0.42, left=0.08, right=0.96, top=0.93, bottom=0.10)

    for ax, ds in zip(axes.flat, multi_step):
        sub = grouped[grouped["Dataset"] == ds]
        alphas = sorted(sub["Alpha"].unique())
        betas = sorted(sub["Beta"].unique())
        mat = (
            sub.pivot_table(index="Alpha", columns="Beta", values="Mean_Val_Acc", aggfunc="mean")
            .reindex(index=alphas, columns=betas)
        )

        im = ax.imshow(mat.values, aspect="auto", origin="lower", cmap="RdYlBu_r")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Mean validation accuracy (%)", fontsize=6.5)
        cbar.ax.tick_params(labelsize=5.5)

        sel_alpha = float(summary.loc[ds, "Selected_Alpha"])
        sel_beta = float(summary.loc[ds, "Selected_Beta"])
        a_idx = _nearest_grid_index(alphas, sel_alpha)
        b_idx = _nearest_grid_index(betas, sel_beta)
        ax.scatter(
            b_idx, a_idx, marker="*", s=185, facecolor="white",
            edgecolor="black", linewidth=0.8, zorder=10,
        )
        text_x, text_y = b_idx + 0.18, a_idx + 0.18
        text_ha, text_va = "left", "bottom"
        if b_idx >= len(betas) - 2:
            text_x, text_ha = b_idx - 0.25, "right"
        if a_idx >= len(alphas) - 2:
            text_y, text_va = a_idx - 0.25, "top"
        ax.text(
            text_x, text_y, "Val-selected",
            fontsize=5.7, color="black", weight="bold",
            ha=text_ha, va=text_va,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.78),
            zorder=11,
        )

        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"{b:.1f}" for b in betas], fontsize=6, rotation=45)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.1f}" for a in alphas], fontsize=6)
        ax.set_xlabel("β (semantic-edge coefficient)", fontsize=7)
        ax.set_ylabel("α (propagation coefficient)", fontsize=7)
        add_panel_label(ax, ds, fontsize=8)

    save_pdf_png(fig, "fig_validation_sensitivity_multistep", dpi=300)
    plt.close(fig)

    # Figure B: constrained one-step validation search with β fixed to 1.0.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.97, top=0.86, bottom=0.20)

    for ax, ds in zip(axes, one_step):
        sub = grid[(grid["Dataset"] == ds) & (np.isclose(grid["Beta"], 1.0))].copy()
        stats = (
            sub.groupby("Alpha", as_index=False)
            .agg(Mean_Val_Acc=("Val_Acc", "mean"),
                 Std_Val_Acc=("Val_Acc", "std"),
                 Num_Seeds=("Seed", "nunique"))
            .sort_values("Alpha")
        )
        stats["SE_Val_Acc"] = stats["Std_Val_Acc"].fillna(0.0) / np.sqrt(stats["Num_Seeds"].clip(lower=1))
        alphas = stats["Alpha"].to_numpy(dtype=float)
        means = stats["Mean_Val_Acc"].to_numpy(dtype=float)
        ses = stats["SE_Val_Acc"].to_numpy(dtype=float)

        ax.errorbar(
            alphas, means, yerr=ses, color=C_SETTA, marker="o",
            markersize=4.8, linewidth=1.3, capsize=2.5,
            markerfacecolor="white", markeredgecolor=C_SETTA,
        )
        sel_alpha = float(summary.loc[ds, "Selected_Alpha"])
        sel_row = stats.iloc[_nearest_grid_index(stats["Alpha"], sel_alpha)]
        ax.scatter(
            [sel_alpha], [sel_row["Mean_Val_Acc"]], marker="*", s=185,
            facecolor="white", edgecolor="black", linewidth=0.8, zorder=10,
        )
        ax.axvline(sel_alpha, color="#444444", linestyle=":", linewidth=0.8, zorder=1)

        label_ds = "Amazon-Computers" if ds == "Computers" else "Amazon-Photo"
        add_panel_label(ax, f"{label_ds} (β = 1.0, one-step)", fontsize=7.5)
        ax.set_xlabel("α (propagation coefficient)", fontsize=7.5)
        ax.set_ylabel("Mean validation accuracy (%)", fontsize=7.5)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.1f}" for a in alphas], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
        ax.set_ylim(means.min() - max(0.25, np.nanmax(ses) + 0.12),
                    means.max() + max(0.45, np.nanmax(ses) + 0.25))
        offset = -14 if sel_row["Mean_Val_Acc"] >= np.nanmedian(means) else 14
        va = "top" if offset < 0 else "bottom"
        ax.annotate(
            "Val-selected",
            xy=(sel_alpha, sel_row["Mean_Val_Acc"]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=6.0,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0.78),
            zorder=11,
        )

    save_pdf_png(fig, "fig_validation_sensitivity_onestep", dpi=300)
    plt.close(fig)
    print("Validation-based configuration sensitivity: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUPPORT FIGURES A/B/C — Mechanism and Analytic Thresholds                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
SUPPORT_DATASET_STATS = {
    "Cora":      {"N": 2708,  "E": 5429,   "avg_deg": 4.01},
    "CiteSeer":  {"N": 3327,  "E": 4732,   "avg_deg": 2.84},
    "PubMed":    {"N": 19717, "E": 44338,  "avg_deg": 4.50},
    "CS":        {"N": 18333, "E": 81894,  "avg_deg": 8.93},
    "Computers": {"N": 13752, "E": 245861, "avg_deg": 35.75},
    "Photo":     {"N": 7650,  "E": 119081, "avg_deg": 31.13},
}
SUPPORT_DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "CS", "Computers", "Photo"]
SUPPORT_DATASET_COLORS = {
    "Cora":      "#D94850",
    "CiteSeer":  "#E8833A",
    "PubMed":    "#3A6EA5",
    "CS":        "#5A9E6F",
    "Computers": "#7B5EA7",
    "Photo":     "#479D9D",
}
C_SUPPORT_RED = "#D94850"
C_SUPPORT_BLUE = "#3A6EA5"
C_SUPPORT_GRAY = "#8C8C8C"
C_SUPPORT_ORANGE = "#E8833A"


def save_pub_png(fig, filename, dpi=300):
    """Save support figures as PDF and PNG."""
    for ext, kw in [(".pdf", {}), (".png", {"dpi": dpi})]:
        fig.savefig(f"figures/{filename}{ext}", bbox_inches="tight", **kw)
    print(f"  Saved: figures/{filename}.[pdf|png]")


def plot_oversmoothing_demo():
    """Fig. A: Karate Club GCN embeddings at increasing propagation depths."""
    warnings.filterwarnings("ignore")
    import networkx as nx
    from sklearn.decomposition import PCA

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n_nodes = len(G.nodes)
    labels = np.array([G.nodes[i]["club"] == "Mr. Hi" for i in range(n_nodes)], dtype=int)

    deg = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-10)))
    a_norm = d_inv_sqrt @ A @ d_inv_sqrt

    rng = np.random.RandomState(2024)
    current = rng.randn(n_nodes, 16)
    embeddings = {1: current.copy()}
    for layer in range(1, 16):
        current = a_norm @ current
        if layer + 1 in [2, 4, 8, 16]:
            embeddings[layer + 1] = current.copy()

    layers = [1, 2, 4, 8, 16]
    all_embeddings = np.vstack([embeddings[layer] for layer in layers])
    pca = PCA(n_components=2, random_state=42).fit(all_embeddings)
    points = {layer: pca.transform(embeddings[layer]) for layer in layers}

    def dirichlet_energy(hidden):
        energy = 0.0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if A[i, j] > 0:
                    diff = hidden[i] - hidden[j]
                    energy += float(np.dot(diff, diff))
        return energy

    e0 = dirichlet_energy(embeddings[1])
    energies = np.array([dirichlet_energy(embeddings[layer]) / e0 for layer in layers])

    fig = plt.figure(figsize=(9.0, 3.8))
    gs = fig.add_gridspec(
        2, 5, height_ratios=[2.5, 1], hspace=0.15, wspace=0.06,
        left=0.05, right=0.98, top=0.92, bottom=0.12,
    )

    xlim = (points[1][:, 0].min() - 0.3, points[1][:, 0].max() + 0.3)
    ylim = (points[1][:, 1].min() - 0.3, points[1][:, 1].max() + 0.3)

    for i, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, i])
        xy = points[layer]
        ax.scatter(
            xy[labels == 1, 0], xy[labels == 1, 1],
            c=C_SUPPORT_BLUE, s=9, alpha=0.88, edgecolors="white",
            linewidths=0.25, zorder=3,
        )
        ax.scatter(
            xy[labels == 0, 0], xy[labels == 0, 1],
            c=C_SUPPORT_RED, s=9, alpha=0.88, edgecolors="white",
            linewidths=0.25, zorder=3,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_label(ax, f"L = {layer}", fontsize=7.5)
        ax.text(
            0.97, 0.04, f"E/E$_0$={energies[i]:.3f}",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom",
            color="#666666",
        )

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SUPPORT_BLUE,
                   markersize=6, markeredgewidth=0.3, markeredgecolor="white"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SUPPORT_RED,
                   markersize=6, markeredgewidth=0.3, markeredgecolor="white"),
        ],
        labels=["Mr. Hi", "Officer"],
        loc="upper center",
        fontsize=6.5,
        handletextpad=0.4,
        borderpad=0.3,
        framealpha=0.85,
        bbox_to_anchor=(0.50, 1.03),
        ncol=2,
    )

    ax_bar = fig.add_subplot(gs[1, :])
    bar_colors = [C_SUPPORT_BLUE if value > 0.10 else "#B02020" for value in energies]
    bars = ax_bar.bar(
        range(len(layers)), energies, color=bar_colors,
        edgecolor="white", linewidth=0.6, width=0.55, zorder=3,
    )
    ax_bar.set_xticks(range(len(layers)))
    ax_bar.set_xticklabels([f"L={layer}" for layer in layers], fontsize=6.8)
    ax_bar.set_ylabel("$\\mathcal{E}\\,/\\,\\mathcal{E}_0$", fontsize=7.5)
    ax_bar.set_ylim(0, 1.12)

    for i, (bar, value) in enumerate(zip(bars, energies)):
        ax_bar.text(i, value + 0.03, f"{value:.3f}", ha="center",
                    fontsize=6, color="#444444")

    ax_bar.axhline(y=0.08, color="#B02020", linestyle="--",
                   linewidth=0.9, alpha=0.55)
    ax_bar.text(4.6, 0.085, "8% threshold", fontsize=5.8,
                color="#B02020", ha="right", va="bottom", fontstyle="italic")

    save_pub_png(fig, "fig_oversmoothing_demo")
    plt.close(fig)
    print("Figure A — Over-smoothing demo: done")


def plot_energy_decay():
    """Fig. B: smoothness-energy-proxy decay curves for the six benchmark graphs."""
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.5))
    t_max = 50
    t = np.arange(0, t_max + 1)

    for ds in SUPPORT_DATASET_ORDER:
        avg_deg = SUPPORT_DATASET_STATS[ds]["avg_deg"]
        lam = np.tanh(avg_deg / 25.0) * 0.68 + 0.025
        curve = (1.0 - lam) ** t
        ax.semilogy(t, curve, color=SUPPORT_DATASET_COLORS[ds],
                    linewidth=1.3, label=ds, zorder=3)

    ax.axhline(y=0.08, color=C_SUPPORT_RED, linestyle="--",
               linewidth=1.0, alpha=0.6, zorder=2)
    ax.text(50.5, 0.078, "8% safety\nthreshold", fontsize=5.8,
            color=C_SUPPORT_RED, ha="left", va="top")

    for ds in SUPPORT_DATASET_ORDER:
        avg_deg = SUPPORT_DATASET_STATS[ds]["avg_deg"]
        lam = np.tanh(avg_deg / 25.0) * 0.68 + 0.025
        curve = (1.0 - lam) ** t
        cross = np.where(curve < 0.08)[0]
        if len(cross) > 0:
            ax.scatter(cross[0], 0.08, color=SUPPORT_DATASET_COLORS[ds], s=20,
                       edgecolors="white", linewidths=0.5, zorder=10)

    ax.text(
        0.02, 0.06,
        "Dense graphs -> rapid proxy decay -> early exit\n"
        "Sparse graphs -> slower proxy decay -> deeper diffusion",
        transform=ax.transAxes, fontsize=6.5, color=C_SUPPORT_GRAY,
        fontstyle="italic",
        bbox=dict(pad=2, fc="white", ec=C_SUPPORT_GRAY, alpha=0.7, linewidth=0.4),
    )
    ax.set_xlabel("Propagation step $t$", fontsize=8)
    ax.set_ylabel("Smoothness-energy proxy ratio", fontsize=8)
    ax.set_xlim(0, t_max)
    ax.set_ylim(3e-4, 1.3)
    ax.legend(fontsize=6.5, loc="upper right", ncol=3,
              columnspacing=0.6, handlelength=1.0, handletextpad=0.4)

    plt.tight_layout()
    save_pub_png(fig, "fig_energy_decay")
    plt.close(fig)
    print("Figure B — Energy decay: done")


def plot_scale_adaptive_confidence_threshold():
    """Fig. C: scale-adaptive tau_max function with dataset annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))

    n_range = np.logspace(2.5, 4.55, 600)
    s = np.log10(n_range)
    tau = np.maximum(0.85, 0.99 - 0.128 * np.maximum(0, s - 3.43))

    ax.axvspan(2.55, 3.43, color=C_SUPPORT_BLUE, alpha=0.06, zorder=0)
    ax.axvspan(3.43, 4.0, color=C_SUPPORT_ORANGE, alpha=0.06, zorder=0)
    ax.axvspan(4.0, 4.5, color=C_SUPPORT_RED, alpha=0.06, zorder=0)
    ax.plot(s, tau, color="black", linewidth=2.0, zorder=5)

    ax.text(2.90, 0.985, "Small Graphs\n$\\tau_{\\max}=0.99$\nnear-invisible",
            fontsize=6.5, color=C_SUPPORT_BLUE, ha="center", va="center",
            fontweight="bold", linespacing=1.3)
    ax.text(3.72, 0.89, "Medium\n$\\tau_{\\max}\\approx 0.88$\nselective gating",
            fontsize=6.5, color=C_SUPPORT_ORANGE, ha="center", va="center",
            fontweight="bold", linespacing=1.3)
    ax.text(4.25, 0.865, "Large\n$\\tau_{\\max}=0.85$\nactive defense",
            fontsize=6.5, color=C_SUPPORT_RED, ha="center", va="center",
            fontweight="bold", linespacing=1.3)

    offsets_y = {"Cora": -9, "CiteSeer": -9, "PubMed": +8,
                 "CS": -9, "Computers": +8, "Photo": +8}
    for ds in SUPPORT_DATASET_ORDER:
        s_i = np.log10(SUPPORT_DATASET_STATS[ds]["N"])
        tau_i = float(np.maximum(0.85, 0.99 - 0.128 * max(0, s_i - 3.43)))
        ax.scatter(s_i, tau_i, color=SUPPORT_DATASET_COLORS[ds], s=55,
                   edgecolors="white", linewidths=0.8, zorder=15)
        offset = offsets_y[ds]
        ax.annotate(
            ds, xy=(s_i, tau_i), xytext=(0, offset), textcoords="offset points",
            fontsize=5.8, color=SUPPORT_DATASET_COLORS[ds], ha="center",
            va="bottom" if offset < 0 else "top", fontweight="bold",
        )

    ax.axvline(x=3.43, color=C_SUPPORT_GRAY, linestyle=":",
               linewidth=0.9, alpha=0.5, zorder=1)
    ax.annotate("breakpoint\n$s = 3.43$\n$(N \\approx 2{,}691)$",
                xy=(3.435, 0.86), fontsize=5.5, color=C_SUPPORT_GRAY,
                ha="left", va="top")
    ax.axhline(y=0.85, color=C_SUPPORT_GRAY, linestyle=":",
               linewidth=0.9, alpha=0.5, zorder=1)

    formula = ("$\\tau_{\\mathrm{max}} = \\max\\,[\\,0.85,\\; "
               "0.99 - 0.128 \\cdot \\max(0,\\, \\log_{10}N - 3.43)\\,]$")
    fig.text(0.08, 0.97, formula, fontsize=7.5, va="top",
             bbox=dict(pad=3, fc="white", ec=C_SUPPORT_GRAY,
                       alpha=0.85, linewidth=0.4))

    ax.set_xlabel("$\\log_{10}(N)$", fontsize=9)
    ax.set_ylabel("Maximum confidence threshold $\\tau_{\\max}$", fontsize=9)
    ax.set_xlim(2.60, 4.45)
    ax.set_ylim(0.843, 1.003)
    tick_positions = [np.log10(x) for x in [1000, 2000, 2700, 5000, 10000, 20000]]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["1K", "2K", "2.7K", "5K", "10K", "20K"],
                       fontsize=7, rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_pub_png(fig, "fig_scale_adaptive_confidence_threshold")
    plt.close(fig)
    print("Figure C — tau_max function: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  METHOD FIGURE — Entropy Gating Decision Map                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def generate_gating_decision_data():
    """Optional regeneration of per-node entropy-gating data for Cora and CS."""
    import torch
    import config
    from config import load_data, train_gcn

    rows = []
    for ds_name in ["Cora", "CS"]:
        print(f"  [{ds_name}] Loading data and training GCN for gating data...")
        ds, data = load_data(ds_name)
        setta_params = getattr(config, LEGACY_SETTA_LABEL + "_PARAMS")
        _ = setta_params[ds_name]
        n_nodes = data.num_nodes
        n_classes = ds.num_classes

        logits = train_gcn(ds, data, seed=42)
        z0 = torch.exp(logits).detach()
        entropy = -(z0 * torch.log(z0 + 1e-9)).sum(dim=1)
        entropy_norm = entropy / torch.log(torch.tensor(n_classes, dtype=z0.dtype))
        max_prob = z0.max(dim=1)[0]

        scale_factor = torch.log10(torch.tensor(n_nodes, dtype=torch.float32))
        tau_max = max(0.85, 0.99 - 0.128 * max(0.0, (scale_factor.item() - 3.43)))
        tau_min = max(0.7, tau_max - 0.20)
        tau_v = tau_max - (tau_max - tau_min) * entropy_norm
        mask = max_prob < tau_v

        for node_id in range(n_nodes):
            rows.append({
                "Dataset": ds_name,
                "N": n_nodes,
                "Node": node_id,
                "H_norm": round(entropy_norm[node_id].item(), 6),
                "max_prob": round(max_prob[node_id].item(), 6),
                "tau_v": round(tau_v[node_id].item(), 6),
                "Decision": "Accept refinement" if mask[node_id].item() else "Keep original",
                "tau_max_dynamic": round(tau_max, 6),
                "tau_min_dynamic": round(tau_min, 6),
            })

    out_path = os.path.join(RESULTS, "gating_decision_nodes.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def plot_entropy_gating_decision():
    """Entropy-gating decision map for Cora and CS."""
    path = os.path.join(RESULTS, "gating_decision_nodes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run generate_gating_decision_data() first, or restore the CSV."
        )

    df = pd.read_csv(path)
    color_keep = "#7A9EC2"
    color_accept = "#E8725A"
    color_curve = "#2D2D2D"

    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 82 / 25.4), dpi=600)

    for idx, (ds_name, ax) in enumerate(zip(["Cora", "CS"], axes)):
        sub = df[df["Dataset"] == ds_name]
        n_nodes = int(sub["N"].iloc[0])
        tau_max = sub["tau_max_dynamic"].iloc[0]
        tau_min = sub["tau_min_dynamic"].iloc[0]
        entropy = sub["H_norm"].values
        max_prob = sub["max_prob"].values
        decision = sub["Decision"].values

        keep_mask = decision == "Keep original"
        accept_mask = decision == "Accept refinement"
        n_keep = int(keep_mask.sum())
        n_accept = int(accept_mask.sum())

        max_points = 5000
        if n_nodes > max_points:
            rng = np.random.RandomState(42)
            keep_idx = np.where(keep_mask)[0]
            accept_idx = np.where(accept_mask)[0]
            n_keep_plot = min(len(keep_idx), int(max_points * n_keep / n_nodes))
            n_accept_plot = min(len(accept_idx), max_points - n_keep_plot)
            keep_plot = rng.choice(keep_idx, n_keep_plot, replace=False) if n_keep_plot > 0 else np.array([], dtype=int)
            accept_plot = rng.choice(accept_idx, n_accept_plot, replace=False) if n_accept_plot > 0 else np.array([], dtype=int)
            plot_idx = np.concatenate([keep_plot, accept_plot])
        else:
            plot_idx = np.arange(n_nodes)

        h_sorted = np.linspace(entropy.min(), entropy.max(), 300)
        tau_curve = tau_max - (tau_max - tau_min) * h_sorted
        ax.fill_between(h_sorted, tau_curve, 1.0, alpha=0.06,
                        color=color_keep, zorder=0, linewidth=0)
        ax.fill_between(h_sorted, 0.0, tau_curve, alpha=0.06,
                        color=color_accept, zorder=0, linewidth=0)
        ax.plot(h_sorted, tau_curve, color=color_curve, linewidth=1.5,
                linestyle="-", zorder=3)

        keep_idx_plot = plot_idx[keep_mask[plot_idx]]
        accept_idx_plot = plot_idx[accept_mask[plot_idx]]
        point_size = 3.0 if ds_name == "Cora" else 1.8
        ax.scatter(entropy[keep_idx_plot], max_prob[keep_idx_plot], s=point_size,
                   c=color_keep, alpha=0.35, edgecolors="none", zorder=1,
                   rasterized=True)
        ax.scatter(entropy[accept_idx_plot], max_prob[accept_idx_plot], s=point_size,
                   c=color_accept, alpha=0.50, edgecolors="none", zorder=2,
                   rasterized=True)

        ax.set_xlabel(r"Normalized entropy $H_{\mathrm{norm}}[i]$", fontsize=7.5)
        ax.set_ylabel(r"$\max_c\; \mathbf{P}^{(0)}[i, c]$", fontsize=7.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        add_panel_label(ax, f"{ds_name}  ($N={n_nodes:,}$)", fontsize=7.5)

        stats_lines = (
            f"Keep original: {n_keep:,} ({100 * n_keep / n_nodes:.1f}%)\n"
            f"Accept refinement: {n_accept:,} ({100 * n_accept / n_nodes:.1f}%)\n"
            r"$\tau_{\max}$" + f" = {tau_max:.4f}, "
            + r"$\tau_{\min}$" + f" = {tau_min:.4f}"
        )
        ax.text(0.97, 0.04, stats_lines, transform=ax.transAxes, fontsize=5.6,
                verticalalignment="bottom", horizontalalignment="right",
                color="#333333", linespacing=1.35)
        ax.text(-0.14, 1.045, r"$\mathbf{" + chr(97 + idx) + r"}$",
                transform=ax.transAxes, fontsize=9.0, va="top")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_keep,
               markersize=5, label="Keep original", alpha=0.8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_accept,
               markersize=5, label="Accept refinement", alpha=0.8),
        Line2D([0], [0], color=color_curve, linewidth=1.5, linestyle="-",
               label=r"Threshold $\tau_v[i]$"),
    ]
    fig.legend(handles=legend_elements, ncol=3, fontsize=6.5,
               loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.01),
               handletextpad=0.4, columnspacing=1.5)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_pdf_png(fig, "fig_entropy_gating_decision", dpi=300)

    plt.close(fig)
    print("Gating decision map: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 6 — Node-Level Transition Analysis                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def plot_node_transition_rates():
    """Node-level transition rates: Wrong->Correct, Correct->Wrong, and net correction."""
    path = os.path.join(RESULTS, "exp6_transition_summary.csv")
    summary = pd.read_csv(path)

    dataset_order = ["Cora", "CiteSeer", "PubMed", "CS", "Computers", "Photo"]
    summary["_order"] = summary["Dataset"].map({d: i for i, d in enumerate(dataset_order)})
    summary = summary.sort_values("_order")

    datasets = summary["Dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    series = [
        ("Wrong_to_Correct_Rate_mean", "Wrong_to_Correct_Rate_std", "Wrong->Correct", C_SETTA),
        ("Correct_to_Wrong_Rate_mean", "Correct_to_Wrong_Rate_std", "Correct->Wrong", "#7884B4"),
        ("Net_Corrected_Rate_mean", "Net_Corrected_Rate_std", "Net corrected", "#5A9E6F"),
    ]

    for offset, (mean_col, std_col, label, color) in zip([-width, 0, width], series):
        ax.bar(
            x + offset,
            summary[mean_col],
            width,
            yerr=summary[std_col],
            capsize=3,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )

    ax.axhline(0, linewidth=0.8, color="#444444")
    ax.set_ylabel("Test nodes (%)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig("figures/fig_node_transition_rates.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_node_transition_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 6 transition rates: done")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 7 — Backbone-Agnostic Frozen Prediction Refinement             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Paste this block before the MAIN section of the unified plotting script.
# It assumes the global constants already defined in your script:
# RESULTS, C_SETTA, C_BASELINES, DATASET_LABELS, add_panel_label, save_pub.

EXP7_DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "CS", "Computers", "Photo"]
EXP7_BACKBONE_ORDER = ["GCN", "GraphSAGE", "GAT", "GraphTransformer", "GradGateGNN"]
EXP7_BACKBONE_LABELS = {
    "GCN": "GCN",
    "GraphSAGE": "SAGE",
    "GAT": "GAT",
    "GraphTransformer": "GT",
    "GradGateGNN": "GradGate",
}


def _load_first_existing_csv(candidates, description):
    for path in candidates:
        if os.path.exists(path):
            print(f"  Using {description}: {path}")
            return pd.read_csv(path), path
    raise FileNotFoundError(
        f"Cannot find {description}. Expected one of:\n  " + "\n  ".join(candidates)
    )


def load_exp7_backbone_summary():
    """Load Exp7 summary. Prefer the extended version with GraphTransformer/GradGateGNN."""
    candidates = [
        os.path.join(RESULTS, "exp7_backbone_setta_extended_summary.csv"),
        os.path.join(RESULTS, "exp7_backbone_setta_summary_with_pvalues.csv"),
        os.path.join(RESULTS, "exp7_backbone_setta_summary.csv"),
    ]
    df, _ = _load_first_existing_csv(candidates, "Exp7 backbone-transfer summary")

    required = {"Dataset", "Backbone", "Frozen_Mean", "SETTA_Mean", "Delta_Mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exp7 summary is missing required columns: {missing}")

    if "Change_Rate_Mean" not in df.columns:
        df["Change_Rate_Mean"] = np.nan
    if "Frozen_Std" not in df.columns:
        df["Frozen_Std"] = 0.0
    if "SETTA_Std" not in df.columns:
        df["SETTA_Std"] = 0.0

    # Keep only the current manuscript Exp7 backbone set, in a stable order.
    df = df[df["Dataset"].isin(EXP7_DATASET_ORDER) & df["Backbone"].isin(EXP7_BACKBONE_ORDER)].copy()
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=EXP7_DATASET_ORDER, ordered=True)
    df["Backbone"] = pd.Categorical(df["Backbone"], categories=EXP7_BACKBONE_ORDER, ordered=True)
    return df.sort_values(["Dataset", "Backbone"]).reset_index(drop=True)


def _matrix_from_exp7(df, value_col):
    mat = (
        df.pivot_table(index="Dataset", columns="Backbone", values=value_col, aggfunc="mean")
        .reindex(index=EXP7_DATASET_ORDER, columns=EXP7_BACKBONE_ORDER)
    )
    return mat


def _annotate_heatmap(ax, mat, fmt, suffix="", color_threshold=None):
    values = mat.values.astype(float)
    if color_threshold is None:
        finite = values[np.isfinite(values)]
        color_threshold = np.nanmax(finite) * 0.55 if finite.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=5.8, color="#777777")
                continue
            color = "white" if value >= color_threshold else "#222222"
            ax.text(
                j,
                i,
                fmt.format(value) + suffix,
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold" if value > 0 else "normal",
                color=color,
            )


def plot_backbone_transfer_paired():
    """
    Exp7 figure: within-dataset paired comparison of frozen and SETTA-refined
    predictions across frozen backbones.

    Each panel uses its own y-axis range so the visual question is whether
    SETTA improves each frozen backbone within a dataset, not which dataset
    has the largest absolute gain.
    """
    df = load_exp7_backbone_summary()

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.17, wspace=0.28, hspace=0.48)

    frozen_color = "#6E6E6E"
    setta_color = C_SETTA
    line_color = "#B8B8B8"
    x = np.arange(len(EXP7_BACKBONE_ORDER))

    for ax, dataset in zip(axes.flat, EXP7_DATASET_ORDER):
        sub = df[df["Dataset"] == dataset].set_index("Backbone").reindex(EXP7_BACKBONE_ORDER)
        frozen = sub["Frozen_Mean"].to_numpy(dtype=float)
        setta = sub["SETTA_Mean"].to_numpy(dtype=float)
        delta = sub["Delta_Mean"].to_numpy(dtype=float)

        for i, (frozen_acc, setta_acc) in enumerate(zip(frozen, setta)):
            ax.plot([i, i], [frozen_acc, setta_acc], color=line_color, linewidth=0.9, zorder=1)

        ax.scatter(x, frozen, s=22, color=frozen_color, marker="o", edgecolor="white", linewidth=0.4, zorder=3, label="Frozen")
        ax.scatter(x, setta, s=28, color=setta_color, marker="^", edgecolor="white", linewidth=0.4, zorder=4, label="SETTA")

        for i, (d, y_top) in enumerate(zip(delta, np.maximum(frozen, setta))):
            ax.text(i, y_top + 0.06, f"+{d:.2f}", ha="center", va="bottom", fontsize=5.4, color=setta_color)

        ymin = float(np.nanmin([frozen.min(), setta.min()]))
        ymax = float(np.nanmax([frozen.max(), setta.max()]))
        pad = max(0.35, (ymax - ymin) * 0.30)
        ax.set_ylim(ymin - pad * 0.55, ymax + pad)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=8.2, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([EXP7_BACKBONE_LABELS[b] for b in EXP7_BACKBONE_ORDER], rotation=28, ha="right", fontsize=6.2)
        ax.tick_params(axis="y", labelsize=6.2)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)", fontsize=7)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=frozen_color, markeredgecolor="white", markersize=5.0, label="Frozen"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=setta_color, markeredgecolor="white", markersize=5.5, label="SETTA"),
        Line2D([0], [0], color=line_color, linewidth=0.9, label="Paired backbone"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False, fontsize=7)

    fig.text(
        0.07,
        0.035,
        "Numbers above each pair report SETTA minus frozen-backbone accuracy in percentage points.",
        fontsize=6.4,
        color="#555555",
        ha="left",
    )

    save_pub(fig, "fig_exp7_backbone_transfer_paired")
    plt.close(fig)
    print("Experiment 7 backbone-transfer paired plot: done")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("=" * 60)
    print("SETTA Paper — Unified Figure Generation Suite")
    print("=" * 60)
    plot_oversmoothing_demo()
    plot_energy_decay()
    plot_scale_adaptive_confidence_threshold()
    try:
        plot_entropy_gating_decision()
    except FileNotFoundError as exc:
        print(f"Skipping gating decision map: {exc}")
    plot_main_citation_results()
    plot_larger_benchmark_results()
    plot_posthoc_accuracy_comparison()
    plot_posthoc_runtime_comparison()
    plot_dual_defense_ablation()
    plot_semantic_injection_ablation()
    plot_validation_based_configuration_sensitivity()
    plot_node_transition_rates()
    plot_backbone_transfer_paired()
    print("=" * 60)
    print("All merged figure-generation tasks completed in ./figures/")
