"""
exp5_sensitivity.py — Section 4.6: Hyperparameter Sensitivity
Target: Sensitivity grid over alpha and beta across all 6 datasets.
Uses fixed structural settings from config.SETTA_FIXED_PARAMS.
"""
import numpy as np, pandas as pd
# 🌟 直接导入 config.py 中的所有原始逻辑，杜绝任何分歧！
from config import *

print("=" * 60)
print("  Exp 5 — Universal Hyperparameter Grid Search")
print("  Datasets: Cora, CiteSeer, PubMed, Computers, Photo, CS")
print("  Grid: α ∈ [0.1, 0.9], β ∈ [0.0, 1.0]")
print("=" * 60)

ALL_DATASETS = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]
alpha_vals = np.arange(0.1, 1.0, 0.1).round(1)
beta_vals = np.arange(0.0, 1.1, 0.1).round(1)

grid_rows = []

for ds_name in ALL_DATASETS:
    print(f"\n" + "═"*40)
    print(f"  🚀 Grid Search for [{ds_name}]")
    print("═"*40)
    
    ds, data = load_data(ds_name)
    cfg = SETTA_FIXED_PARAMS[ds_name]
    cit = get_cit_adj(data)
    
    sem = build_semantic_adj(data, k=cfg["k"], metric=cfg["metric"])
    
    # 提前计算所有 10 个 seed 的 GCN logits (极大节省时间)
    print(f"    Pre-computing GCN logits for 10 seeds...")
    gcn_logits = {seed: train_gcn(ds, data, seed) for seed in SEEDS}
    
    best_acc = 0
    best_params = (None, None)
    
    for a in alpha_vals:
        for b in beta_vals:
            accs = []
            for seed in SEEDS:
                out = dssr_refine(gcn_logits[seed], data, sem, cit, 
                                  alpha=a, beta=b, 
                                  metric=cfg["metric"], 
                                  num_props=cfg.get("num_props", NUM_PROPS))
                accs.append(eval_acc(out, data) * 100)
            
            m, s = np.mean(accs), np.std(accs)
            grid_rows.append({
                "Dataset": ds_name, "Alpha": a, "Beta": b, 
                "Mean": round(m, 2), "Std": round(s, 2)
            })
            
            if m > best_acc:
                best_acc = m
                best_params = (a, b)
                
    print(f"\n  🌟 [Optimal Found for {ds_name}]: α={best_params[0]:.1f}, β={best_params[1]:.1f} -> Acc: {best_acc:.2f}%")

df = pd.DataFrame(grid_rows)
df.to_csv("results/exp5_grid_search_all.csv", index=False)
print("\n🎉 All grid searches finished! Saved to results/exp5_grid_search_all.csv")
