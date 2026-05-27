"""
exp4_ablation.py — Section 4.5: SETTA Ablation Studies
Framework: Spectral-Energy Test-Time Adaptation (SETTA)
Target: Evaluate Dual Defense (Energy + Gating), Adaptive SVD, and Spectral Injection.
"""
import numpy as np, pandas as pd, torch
from config import *
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# ── 模块 1：狄利克雷能量监控 (Scale-Aware Macro Defense) ──
def calc_dirichlet_energy(Z, norm_adj):
    """计算基于归一化邻接矩阵的狄利克雷能量"""
    Z_prop = torch.mm(norm_adj, Z)
    return (Z * (Z - Z_prop)).sum().item()

# ── 模块 2：尺度自适应流形去噪 (Scale-Adaptive Manifold Denoising) ──
def build_semantic_adj_ablation(data, k, metric="cosine", use_svd=True):
    X = data.x.cpu().numpy()
    N, D = X.shape

    if metric == "jaccard":
        sim = 1 - pairwise_distances(X.astype(bool), metric="jaccard", n_jobs=8)
    else:
        if use_svd and D > 128:
            # 🌟 尺度自适应 SVD：保持 128 的保底维度
            adaptive_dim = max(128, D // 8)
            svd = TruncatedSVD(n_components=adaptive_dim, random_state=42)
            X_reduced = svd.fit_transform(X)
        else:
            X_reduced = X
        sim = cosine_similarity(X_reduced)
        
    np.fill_diagonal(sim, 0)
    adj_rewire = torch.zeros(N, N, device=DEVICE)
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.arange(N).repeat(k)
    adj_rewire[rows, idx.flatten()] = 1.0
    return adj_rewire

# ── 核心引擎：SETTA Test-Time Adaptation ──
def setta_ablation_engine(logits, data, sem_rewire, cit_adj, alpha, beta, num_props,
                          use_energy=True, use_gating=True):
    Z0 = torch.exp(logits)
    C = Z0.size(1)
    
    # 模块 3：谱域低通感知注入
    M_rewire = (sem_rewire == 1) & (cit_adj == 0)
    hybrid = cit_adj + beta * M_rewire.float() + torch.eye(cit_adj.shape[0], device=DEVICE)
    norm = hybrid / (hybrid.sum(1, keepdim=True) + 1e-10)
    
    # 执行混合传播
    Z = Z0.clone()
    E0 = calc_dirichlet_energy(Z0, norm)
    
    for step in range(num_props):
        Z_new = (1 - alpha) * Z + alpha * torch.mm(norm, Z)
        
        if use_energy:
            E_current = calc_dirichlet_energy(Z_new, norm)
            # 能量防火墙：跌破 5% 初始能量即视为流形坍缩，紧急刹车
            if step >= 3 and E_current < 0.1 * E0:
                break
        Z = Z_new
        
    # ── 模块 4：尺度感知零样本熵门控 (Scale-Aware Zero-Shot Gating) ──
    if use_gating:
        N = data.num_nodes
        scale_factor = torch.log10(torch.tensor(N, dtype=torch.float32))

        tau_max_dynamic = max(0.85, 0.99 - 0.128 * max(0.0, (scale_factor.item() - 3.43)))
        tau_min_dynamic = max(0.7, tau_max_dynamic - 0.20)
        
        H = -(Z0 * torch.log(Z0 + 1e-9)).sum(dim=1)
        H_norm = H / torch.log(torch.tensor(C, dtype=Z0.dtype, device=Z0.device))
        
        tau_v = tau_max_dynamic - (tau_max_dynamic - tau_min_dynamic) * H_norm
        
        # 纯净逻辑：如果不够自信 (小于 tau_v)，则使用传播平滑后的 Z；否则强制退回保护 Z0
        mask = Z0.max(dim=1)[0] < tau_v
        return torch.where(mask.unsqueeze(1), Z, Z0)
    else:
        return Z


# ══════════════════════════════════════════════════
#  Exp 4A: Macro/Micro Dual Defense Curves (K-Sweep)
# ══════════════════════════════════════════════════
print("=" * 60)
print("  Exp 4A — Continuous Curve Ablation (Dual Defense)")
print("  Datasets: Cora, PubMed, Coauthor-CS")
print("=" * 60)

SWEEP_CONFIGS = {
    "PubMed": {"num_props": 30, "K_values": [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]},
    "CS":     {"num_props": 30, "K_values": [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]},
}

VARIANTS = [
    {"name": "Blind Propagation", "energy": False, "gating": False},
    {"name": "+ Energy Monitor",  "energy": True,  "gating": False},
    {"name": "+ Entropy Gating",  "energy": False, "gating": True},
    {"name": "SETTA Full",        "energy": True,  "gating": True}
]

exp4a_rows = []

for ds_name in ["PubMed", "CS"]:
    ds, data = load_data(ds_name)
    cit = get_cit_adj(data)
    cfg = get_setta_config(ds_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
    sweep_cfg = SWEEP_CONFIGS[ds_name]
    metric = cfg["metric"]
    
    print(
        f"\n  [{ds_name}] Deep Propagation Test (T={sweep_cfg['num_props']}) | "
        f"alpha={cfg['alpha']}, beta={cfg['beta']}"
    )
    gcn_logits = {seed: train_gcn(ds, data, seed) for seed in SEEDS}
    
    for k_val in sweep_cfg["K_values"]:
        sem = build_semantic_adj_ablation(data, k=k_val, metric=metric, use_svd=True)
        
        for var in VARIANTS:
            accs = []
            for seed in SEEDS:
                out = setta_ablation_engine(
                    gcn_logits[seed], data, sem, cit, 
                    alpha=cfg["alpha"], beta=cfg["beta"], 
                    num_props=sweep_cfg["num_props"], 
                    use_energy=var["energy"], use_gating=var["gating"]
                )
                accs.append(eval_acc(out, data) * 100)
            
            m, s = np.mean(accs), np.std(accs)
            exp4a_rows.append({
                "Dataset": ds_name, "K": k_val, "Variant": var["name"], 
                "Mean": round(m, 2), "Std": round(s, 2)
            })
        print(f"    [K={k_val:3d}] Done.")

pd.DataFrame(exp4a_rows).to_csv("results/exp4a_dual_defense_curves.csv", index=False)


# ══════════════════════════════════════════════════
#  Global Datasets Settings for Exp 4B & 4C
# ══════════════════════════════════════════════════
ALL_DATASETS = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS"]

# ══════════════════════════════════════════════════
#  Exp 4B: High-Dimensional Manifold Denoising (SVD)
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Exp 4B — Universal Manifold Denoising (All 6 Datasets)")
print("=" * 60)

exp4b_rows = []
for ds_name in ALL_DATASETS:
    ds, data = load_data(ds_name)
    cfg = get_setta_config(ds_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
    cit = get_cit_adj(data)
    
    print(f"\n  [{ds_name}] Feature Denoising Evaluation...")
    gcn_logits_list = {seed: train_gcn(ds, data, seed) for seed in SEEDS}
    
    for setup_name, use_svd_flag in [("Raw Features (No SVD)", False), ("SVD-Enhanced (Adaptive)", True)]:
        sem = build_semantic_adj_ablation(data, k=cfg["k"], metric=cfg["metric"], use_svd=use_svd_flag)
        
        accs = []
        for seed in SEEDS:
            out = setta_ablation_engine(
                gcn_logits_list[seed], data, sem, cit, 
                cfg["alpha"], cfg["beta"], 
                num_props=cfg.get("num_props", NUM_PROPS), 
                use_energy=True, use_gating=True
            )
            accs.append(eval_acc(out, data) * 100)
            
        m, s = np.mean(accs), np.std(accs)
        exp4b_rows.append({"Dataset": ds_name, "Setup": setup_name, "Mean": round(m,2), "Std": round(s,2)})
        print(f"    {setup_name:35s}: {m:.2f} ± {s:.2f}")

pd.DataFrame(exp4b_rows).to_csv("results/exp4b_svd_denoising.csv", index=False)


# ══════════════════════════════════════════════════
#  Exp 4C: Spectral Low-pass Injection vs Pure Topology
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Exp 4C — Spectral Injection vs Pure Topology (All 6 Datasets)")
print("=" * 60)

exp4c_rows = []
for ds_name in ALL_DATASETS:
    ds, data = load_data(ds_name)
    cfg = get_setta_config(ds_name, DEFAULT_VALIDATION_SELECTED_AB_PATH)
    cit = get_cit_adj(data)
    
    sem = build_semantic_adj_ablation(data, k=cfg["k"], metric=cfg["metric"], use_svd=True)
    
    print(f"\n  [{ds_name}] Spectral Injection Evaluation...")
    gcn_logits_list = {seed: train_gcn(ds, data, seed) for seed in SEEDS}
    
    for setup_name, beta_val in [("Topology-Only (β=0)", 0.0), (f"Spectral-Aware (β={cfg['beta']})", cfg["beta"])]:
        accs = []
        for seed in SEEDS:
            out = setta_ablation_engine(
                gcn_logits_list[seed], data, sem, cit, 
                cfg["alpha"], beta_val, 
                num_props=cfg.get("num_props", NUM_PROPS), 
                use_energy=True, use_gating=True
            )
            accs.append(eval_acc(out, data) * 100)
            
        m, s = np.mean(accs), np.std(accs)
        exp4c_rows.append({"Dataset": ds_name, "Setup": setup_name, "Mean": round(m,2), "Std": round(s,2)})
        print(f"    {setup_name:35s}: {m:.2f} ± {s:.2f}")

pd.DataFrame(exp4c_rows).to_csv("results/exp4c_spectral_injection.csv", index=False)
print("\n🎉 All 4A, 4B, 4C ablation experiments generated successfully!")
