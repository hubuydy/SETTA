"""
config.py — SETTA (Spectral-Energy Test-Time Adaptation)

This file stores only dataset-level structural protocol settings. The SETTA
adaptation coefficients alpha and beta are loaded from a validation-selection
CSV generated before running the main experiments.

Some function and variable names keep the legacy DSSR label for compatibility.
"""

import csv
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Device ──
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            d = torch.device("mps")
            ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=d)
            x = torch.randn(2, 16, device=d)
            _ = GCNConv(16, 16).to(d)(x, ei)
            return d
        except NotImplementedError:
            return torch.device("cpu")
    return torch.device("cpu")

DEVICE = get_device()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ── Seeds ──
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

NUM_PROPS = 10

DEFAULT_VALIDATION_SELECTED_AB_PATH = os.environ.get(
    "SETTA_SELECTED_AB_PATH",
    os.path.join("results", "validation_ab_selection", "val_selected_ab_summary.csv"),
)

# Fixed dataset-level structural protocol. These values are not selected using
# validation or test accuracy.
SETTA_FIXED_PARAMS = {
    "Cora":      {"k": 18, "metric": "jaccard", "num_props": 10},
    "CiteSeer":  {"k": 12, "metric": "jaccard", "num_props": 10},
    "PubMed":    {"k": 5,  "metric": "cosine",  "num_props": 10},
    "CS":        {"k": 20, "metric": "cosine",  "num_props": 10},
    "Computers": {"k": 25, "metric": "cosine",  "num_props": 1},
    "Photo":     {"k": 30, "metric": "cosine",  "num_props": 1},
}

# Backward-compatible aliases. These intentionally contain only fixed
# structural settings; alpha/beta must come from validation selection.
DSSR_PARAMS = SETTA_FIXED_PARAMS
SETTA_PARAMS = SETTA_FIXED_PARAMS


def load_validation_selected_ab(path=DEFAULT_VALIDATION_SELECTED_AB_PATH):
    """
    Load validation-selected SETTA alpha/beta values.

    The CSV must contain Dataset, Selected_Alpha, and Selected_Beta columns.
    Main experiments should fail loudly when this file is missing so that test
    labels cannot be used, even accidentally, for configuration selection.
    """
    if path is None:
        path = DEFAULT_VALIDATION_SELECTED_AB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Validation-selected alpha/beta file not found. "
            "Please run exp_val_ab_risk_selection_constrained_v2.py first. "
            f"Expected path: {path}"
        )

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Dataset", "Selected_Alpha", "Selected_Beta"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Validation-selected alpha/beta CSV is missing required columns: "
                f"{sorted(missing)}. Expected at least Dataset, Selected_Alpha, "
                "and Selected_Beta."
            )

        selected = {}
        for row in reader:
            dataset = row["Dataset"].strip()
            if not dataset:
                continue
            selected[dataset] = {
                "alpha": float(row["Selected_Alpha"]),
                "beta": float(row["Selected_Beta"]),
            }

    return selected


def get_setta_config(dataset_name, selected_ab_path=None):
    """
    Return the complete SETTA configuration for one dataset.

    k, metric, and num_props come from SETTA_FIXED_PARAMS. alpha and beta are
    loaded from the validation-selected CSV.
    """
    if dataset_name not in SETTA_FIXED_PARAMS:
        raise ValueError(f"No fixed SETTA structural config found for dataset {dataset_name}.")

    selected = load_validation_selected_ab(
        selected_ab_path or DEFAULT_VALIDATION_SELECTED_AB_PATH
    )
    if dataset_name not in selected:
        raise ValueError(f"No validation-selected alpha/beta found for dataset {dataset_name}.")

    cfg = dict(SETTA_FIXED_PARAMS[dataset_name])
    cfg.update(selected[dataset_name])
    return cfg

# Adaptive gating bounds (Zero-shot)
TAU_MAX = 0.90
TAU_MIN = 0.70

# ── Data Loading ──
def load_data(name):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        ds = Planetoid(root="./data", name=name)
    elif name in ["CS", "Physics"]:
        ds = Coauthor(root="./data", name=name)
    elif name in ["Computers", "Photo"]:
        ds = Amazon(root="./data", name=name)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data = ds[0]
    if name in ["CS", "Physics", "Computers", "Photo"]:
        transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.8)
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        data = transform(data)
        torch.set_rng_state(rng_state)
    return ds, data.to(DEVICE)

# ── GCN (hidden=64) ──
class GCN(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, 64)
        self.conv2 = GCNConv(64, out_ch)
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)

def _get_mask(mask, split_idx=0):
    if mask.dim() == 2:
        return mask[:, split_idx]
    return mask

def train_gcn(dataset, data, seed=42, epochs=300, patience=50, split_idx=0):
    set_seed(seed)
    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)
    model = GCN(dataset.num_features, dataset.num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_out, best_score, wait = None, -float("inf"), 0
    is_large_val = val_mask.sum().item() > 100

    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data)
        F.nll_loss(out[train_mask], data.y[train_mask]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = (out_eval[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
        if score > best_score: best_score = score; best_out = out_eval.detach().clone(); wait = 0
        else: wait += 1
        if wait > patience: break
    return best_out

# ── Structural Operations ──
def build_semantic_adj(data, k, metric="jaccard"):
    """
    SVD-Enhanced Semantic Graph Construction (Adaptive SVD Denoising).
    Compresses to max(128, D//10) to preserve semantic variance.
    """
    X = data.x.cpu().numpy()
    N, D = X.shape

    if metric == "jaccard":
        sim = 1 - pairwise_distances(X.astype(bool), metric="jaccard", n_jobs=8)
    else:
        if D > 128:
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

def get_cit_adj(data):
    return to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(DEVICE)

# ── 新增：狄利克雷能量监控辅助函数 (零参数, 极速计算) ──
def calc_dirichlet_energy(Z, norm_adj):
    """
    计算基于随机游走拉普拉斯的狄利克雷能量 (Dirichlet Energy).
    公式: E = Tr(Z^T (I - \tilde{A}) Z) = sum(Z * (Z - \tilde{A}Z))
    """
    Z_prop = torch.mm(norm_adj, Z)
    energy = (Z * (Z - Z_prop)).sum().item()
    return energy

# ── SETTA refinement ──
# dssr_refine is the legacy implementation name of SETTA refinement.
def dssr_refine(logits, data, sem_rewire, cit_adj, alpha, beta, metric="cosine", num_props=NUM_PROPS):
    Z0 = torch.exp(logits)
    C = Z0.size(1)
    
    # ── 1. 谱域低通注入 (Spectral Low-Pass Injection) ──
    # 在论文中我们会这样包装：通过拉普拉斯矩阵 L 分离频率，
    # 提取其互补的低频平滑算子 (S = I - L = A_sem) 来增强同配性。
    M_rewire = (sem_rewire == 1) & (cit_adj == 0)
    
    # 修复：退回原始的绝对安全混合公式，确保不会出现负权重的斥力边。
    # 这里的 M_rewire.float() 在谱图意义上就是未经度归一化的低通滤波器 S_sem。
    hybrid = cit_adj + beta * M_rewire.float() + torch.eye(cit_adj.shape[0], device=DEVICE)
    norm = hybrid / (hybrid.sum(1, keepdim=True) + 1e-10)
    
    # ── 2. Test-Time Compute: 狄利克雷能量驱动 (Energy Early-Exit) ──
    Z = Z0.clone()
    E0 = calc_dirichlet_energy(Z0, norm)
    
    for step in range(num_props):
        Z_new = (1 - alpha) * Z + alpha * torch.mm(norm, Z)
        
        # 实时监控能量耗散 (Over-smoothing Monitor)
        E_current = calc_dirichlet_energy(Z_new, norm)
        
        # 修复：Cora 需要平滑，不能太早退出！
        # 设定至少传播 3 次，且能量跌破初始的 5% 才判定为极端过平滑危险并截断。
        # 这样在 Cora 上不会误触发，但在大深度的 Amazon 图上依然能起保护作用。
        if step >= 3 and E_current < 0.08 * E0:
            break
            
        Z = Z_new
        
    # ── 3. Zero-Shot Entropy-Driven Adaptive Gating (尺度感知门控兜底) ──
    # 获取图的节点规模
    N = data.num_nodes
    
    # 🌟 尺度感知 TAU_MAX 动态计算 (基于对数平滑过渡)
    scale_factor = torch.log10(torch.tensor(N, dtype=torch.float32))
    tau_max_dynamic = max(0.85,0.99 - 0.128 * max(0.0, (scale_factor.item() - 3.43)))
    tau_min_dynamic = max(0.7,tau_max_dynamic - 0.2) # 保持 0.25 的落差
    
    H = -(Z0 * torch.log(Z0 + 1e-9)).sum(dim=1)
    H_norm = H / torch.log(torch.tensor(C, dtype=Z0.dtype, device=Z0.device))
    
    # 使用动态阈值
    tau_v = tau_max_dynamic - (tau_max_dynamic - tau_min_dynamic) * H_norm
    mask = Z0.max(dim=1)[0] < tau_v
    
    return torch.where(mask.unsqueeze(1), Z, Z0)


setta_refine = dssr_refine


# ── Evaluation ──
def eval_acc(logits, data, split_idx=0):
    test_mask = _get_mask(data.test_mask, split_idx)
    return (logits.argmax(1)[test_mask] == data.y[test_mask]).float().mean().item()
