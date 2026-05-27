"""
exp2_scalability_robustness.py — Section 4.3: Scalability and Large-scale Results (Table 2)
Datasets: Coauthor-CS, Amazon-Computers, Amazon-Photo
8 methods × 3 large-scale datasets × 10 seeds
"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, time
from torch_geometric.nn import GATConv, SAGEConv, GINConv, TransformerConv
from config import *
from config import _get_mask

# ── Additional Models ──
class MLP(torch.nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.l1 = torch.nn.Linear(i, h); self.l2 = torch.nn.Linear(h, o)
    def forward(self, d):
        x = F.relu(self.l1(d.x)); x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.l2(x), dim=1)

class GAT(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = GATConv(i, 8, heads=8, dropout=0.6)
        self.c2 = GATConv(64, o, heads=1, concat=False, dropout=0.6)
    def forward(self, d):
        x = F.dropout(d.x, p=0.6, training=self.training)
        x = F.elu(self.c1(x, d.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(self.c2(x, d.edge_index), dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.c1 = SAGEConv(i, 64); self.c2 = SAGEConv(64, o)
    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.c2(x, d.edge_index), dim=1)

class GIN(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = GINConv(torch.nn.Sequential(torch.nn.Linear(i,64),torch.nn.BatchNorm1d(64),torch.nn.ReLU(),torch.nn.Linear(64,64)))
        self.c2 = GINConv(torch.nn.Sequential(torch.nn.Linear(64,64),torch.nn.BatchNorm1d(64),torch.nn.ReLU(),torch.nn.Linear(64,o)))
    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.c2(x, d.edge_index), dim=1)

class GraphTransformer(torch.nn.Module):
    def __init__(self, i, h, o, heads=4, cd=0.1, md=0.3):
        super().__init__()
        self.c1 = TransformerConv(i, h//heads, heads=heads, dropout=cd)
        self.c2 = TransformerConv(h, o, heads=1, concat=False, dropout=cd)
        self.n1 = torch.nn.LayerNorm(h); self.n2 = torch.nn.LayerNorm(o); self.md = md
    def forward(self, d):
        x = F.relu(self.n1(self.c1(d.x, d.edge_index)))
        x = F.dropout(x, p=self.md, training=self.training)
        return F.log_softmax(self.n2(self.c2(x, d.edge_index)), dim=1)

class GradGateGNN(torch.nn.Module):
    def __init__(self, i, h, o, drop=0.5):
        super().__init__(); self.drop = drop
        self.ls1 = torch.nn.Linear(i, h); self.ln1 = torch.nn.Linear(i, h)
        self.g1 = torch.nn.Linear(h*2, h); self.n1 = torch.nn.LayerNorm(h)
        self.ls2 = torch.nn.Linear(h, o); self.ln2 = torch.nn.Linear(h, o)
        self.g2 = torch.nn.Linear(o*2, o); self.n2 = torch.nn.LayerNorm(o)
    def _gate_conv(self, x, ei, ls, ln, gate, norm):
        h_s = ls(x); row, col = ei
        deg = torch.zeros(x.size(0), device=x.device).scatter_add_(0, row, torch.ones(row.size(0), device=x.device)).clamp(min=1)
        agg = torch.zeros_like(x).scatter_add_(0, col.view(-1,1).expand(-1,x.size(1)), x[row]) / deg.unsqueeze(1)
        h_n = ln(agg); g = torch.sigmoid(gate(torch.cat([h_s, h_n], -1)))
        return norm(g * h_n + (1-g) * h_s)
    def forward(self, d):
        x = F.relu(self._gate_conv(d.x, d.edge_index, self.ls1, self.ln1, self.g1, self.n1))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.log_softmax(self._gate_conv(x, d.edge_index, self.ls2, self.ln2, self.g2, self.n2), dim=1)

# ── 包含规模感知早停的统一训练函数 ──
def train_model(model, dataset, data, seed, split_idx=0, epochs=300, patience=50, lr=0.005, wd=5e-4):
    set_seed(seed); model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)
    
    best_out, best_score, wait = None, -float("inf"), 0
    is_large_val = val_mask.sum().item() > 100
    
    for _ in range(epochs):
        model.train(); opt.zero_grad(); out = model(data)
        F.nll_loss(out[train_mask], data.y[train_mask]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = (out_eval[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
                
        if score > best_score: 
            best_score = score; best_out = out_eval.detach().clone(); wait = 0
        else: 
            wait += 1
        if wait > patience: break
    return best_out

def train_gat(dataset, data, seed, split_idx=0, epochs=300, patience=50):
    set_seed(seed)
    model = GAT(dataset.num_features, dataset.num_classes).to(DEVICE)
    opt = torch.optim.Adam([dict(params=model.c1.parameters(), weight_decay=5e-4),
                            dict(params=model.c2.parameters(), weight_decay=0)], lr=0.005)
                            
    train_mask = _get_mask(data.train_mask, split_idx)
    val_mask = _get_mask(data.val_mask, split_idx)
    
    best_out, best_score, wait = None, -float("inf"), 0
    is_large_val = val_mask.sum().item() > 100
    
    for _ in range(epochs):
        model.train(); opt.zero_grad(); out = model(data)
        F.nll_loss(out[train_mask], data.y[train_mask]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            if is_large_val:
                score = -F.nll_loss(out_eval[val_mask], data.y[val_mask]).item()
            else:
                score = (out_eval[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
                
        if score > best_score: 
            best_score = score; best_out = out_eval.detach().clone(); wait = 0
        else: 
            wait += 1
        if wait > patience: break
    return best_out

# ── Main ──
print("=" * 60)
print("  Exp 2 — Table 2: Scalability and Large-scale Results")
print("  Datasets: Coauthor-CS, Amazon-Computers, Amazon-Photo")
print("  Methods: 8 Baselines (10 seeds each)")
print("=" * 60)

# 加载工业级大规模数据集
datasets = {
    "CS": load_data("CS"),
    "Computers": load_data("Computers"),
    "Photo": load_data("Photo")
}

methods = ["MLP", "GCN", "GraphSAGE", "GIN", "GAT", "GraphTransformer", "GradGateGNN", "SETTA"]
SELECTED_AB_PATH = DEFAULT_VALIDATION_SELECTED_AB_PATH
print(f"  Validation-selected alpha/beta: {SELECTED_AB_PATH}")
rows = []

for ds_name, (ds, data) in datasets.items():
    cfg = get_setta_config(ds_name, SELECTED_AB_PATH)
    # 使用带 SVD 去噪的构建函数
    sem = build_semantic_adj(data, cfg["k"], cfg["metric"])
    cit = get_cit_adj(data)
    print(
        f"\n  [{ds_name}]  Large-scale homophilic graph | "
        f"SETTA config: alpha={cfg['alpha']}, beta={cfg['beta']}, "
        f"k={cfg['k']}, num_props={cfg['num_props']}"
    )

    for method in methods:
        accs = []
        start_time = time.time()
        for seed in SEEDS:
            if method == "MLP":
                out = train_model(MLP(ds.num_features, 64, ds.num_classes), ds, data, seed)
            elif method == "GCN":
                out = train_gcn(ds, data, seed)
            elif method == "GraphSAGE":
                out = train_model(GraphSAGE(ds.num_features, ds.num_classes), ds, data, seed)
            elif method == "GIN":
                out = train_model(GIN(ds.num_features, ds.num_classes), ds, data, seed)
            elif method == "GAT":
                out = train_gat(ds, data, seed)
            elif method == "GraphTransformer":
                out = train_model(GraphTransformer(ds.num_features, 128, ds.num_classes), ds, data, seed)
            elif method == "GradGateGNN":
                out = train_model(GradGateGNN(ds.num_features, 256, ds.num_classes, 0.6), ds, data, seed)
            elif method == "SETTA":
                # SETTA receives frozen GCN predictions as its backbone output.
                out = train_gcn(ds, data, seed)
                props = cfg["num_props"]
                out = dssr_refine(out, data, sem, cit, cfg["alpha"], cfg["beta"], metric=cfg["metric"], num_props=props)
            
            accs.append(eval_acc(out, data) * 100)
            
        m, s = np.mean(accs), np.std(accs)
        elapsed = time.time() - start_time
        rows.append({"Dataset": ds_name, "Method": method, "Mean_Accuracy": round(m, 2), "Std_Dev": round(s, 2)})
        print(f"    {method:18s}  {m:.2f} ± {s:.2f}  ({elapsed:.1f}s)")

df = pd.DataFrame(rows)
df.to_csv("results/table2_large_scale_results.csv", index=False)
print(f"\n  💾 Saved to results/table2_large_scale_results.csv")
