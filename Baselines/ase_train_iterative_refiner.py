"""
Iterative Element Refiner
- Takes corrupted graph, predicts correct element at each node (23-class)
- K iterations of distance-aware multi-head attention (GAT + RBF distance bias)
  with soft element conditioning (P @ E_matrix avoids circular dependency)
- Loss: CrossEntropy only on supervised nodes (ignore_index=-1)
- Training: supervise ALL vocab-nodes with correct element
            (corrupted sites → z_old, clean sites → current z)
"""
import torch
import torch.nn as nn
import dgl
import dgl.function as fn 
from dgl.ops import edge_softmax
import os, random
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ase_model import TopoSSL
from ase_finetune import ensure_node_features

ELEM_VOCAB = [
    6, 7, 8, 17, 35,                          # C, N, O, Cl, Br
    22, 23, 24, 25, 26, 27, 28, 29, 30,       # Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
    42, 44, 45, 46, 47, 48,                   # Mo, Ru, Rh, Pd, Ag, Cd
    74, 78, 79,                               # W, Pt, Au
]
Z_TO_IDX = {z: i for i, z in enumerate(ELEM_VOCAB)}
N_ELEM   = len(ELEM_VOCAB)   # 23

METALLIC_CKPT = 'ase_checkpoints_metallic/checkpoint_best.pt'
ND_TRAIN      = 'data/processed_data_2/node_detection/train_nd.pt'
ND_VAL        = 'data/processed_data_2/node_detection/val_nd.pt'
OUT_DIR       = 'ase_checkpoints_iterative_refiner'

# Dataset
class RefinerDataset(Dataset):
    def __init__(self, path):
        self.records = torch.load(path)['records']
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        r = self.records[index]
        g = r['graph']
        N = g.num_nodes()
        
        # Defalut: map every node to its vocal class, -1 if not in vocab
        atomic_nums = g.ndata['atomic_num'].long()
        elem_labels = torch.tensor(
            [Z_TO_IDX.get(z.item(), -1) for z in atomic_nums],
            dtype=torch.long
        )
        
        # override corrupted sites with the correct elemt (z_old)
        for site, z_correct in zip(r['corrupted_sites'], r['z_old']):
            elem_labels[site] = Z_TO_IDX.get(z_correct, -1)
            
        return g, elem_labels

def refiner_collate_fn(batch):
    graphs, labels = zip(*batch)
    graphs = [ensure_node_features(g, 53) for g in graphs]
    batched = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0) #[N_total]
    return batched, labels

class IterativeRefiner(nn.Module):
    def __init__(self, metallic_ckpt, n_elem=N_ELEM, 
                 n_heads=8, d_head=64, d_elem=32, rbf_dim=50, K=3):
        super().__init__()
        
        hidden_dim, num_layers = 512, 5
        D       = hidden_dim * num_layers   # 2560
        d_model = n_heads * d_head          # 256

        # Frozen TopoSSL encoder
        self.ssl_model = TopoSSL(
            num_features=53, hidden_dim=512, num_layers=5,
            proj_dim=256, pool='mean'
        )
        ckpt = torch.load(metallic_ckpt, map_location='cpu')
        self.ssl_model.load_state_dict(ckpt['encoder_state_dict'])
        for p in self.ssl_model.parameters():
            p.requires_grad = False

        # Shared projections (reused across K iterations)
        self.W_q = nn.Linear(D, d_model, bias=False)
        self.W_k = nn.Linear(D, d_model, bias=False)
        self.W_v = nn.Linear(D, d_model, bias=False)

        # Element-conditioned distance bias [n_elem, n_heads, rbf_dim]
        self.W_b = nn.Parameter(torch.zeros(n_elem, n_heads, rbf_dim))
        nn.init.normal_(self.W_b, std=0.01)

        # Head aggregation
        self.W_o = nn.Linear(d_model, 128, bias=False)

        # Soft element embedding table [n_elem, d_elem]
        self.elem_embed = nn.Embedding(n_elem, d_elem)

        # Prediction head (shared across K iterations)
        self.head = nn.Sequential(
            nn.Linear(D + 128 + d_elem, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_elem),
        )

        self.n_heads = n_heads
        self.d_head  = d_head
        self.d_model = d_model
        self.K       = K
        self.scale   = d_head ** -0.5
        self.n_elem  = n_elem

    def _attend_all(self, g, h_node, W_b_soft):
        """vectorized distance-aware multi-head attention"""
        N = h_node.shape[0]
        H = self.n_heads
        Dh = self.d_head
        
        # 1. Pre-compute projections [N, H, Dh]
        Q = self.W_q(h_node).view(N, H, Dh)
        K = self.W_k(h_node).view(N, H, Dh)
        V = self.W_v(h_node).view(N, H, Dh)
        
        # 2. soft distance bias [N, H, rbf_dim]
        #W_b_soft = torch.einsum('nz,zhr->nhr', P, self.W_b)
        
        # 3. store on graph nodes
        g.ndata.update({'_q': Q, '_k': K, '_v': V, '_wb': W_b_soft})
        
        def compute_scores(edges):
            scores = (edges.dst['_q'] * edges.src['_k']).sum(-1) * self.scale  # [E, H]
            if 'rbf' in edges.data:
                phi    = edges.data['rbf']                                           # [E, rbf_dim]
                bias   = (edges.dst['_wb'] * phi.unsqueeze(1)).sum(-1)              # [E, H]
                scores = scores + bias
            return {'_score': scores}

        g.apply_edges(compute_scores)
        
        g.edata['_alpha'] = edge_softmax(g, g.edata['_score']) #_alpha[E,H]
        
        def weight_v(edges):
            return {'_av': edges.src['_v'] * edges.data['_alpha'].unsqueeze(-1)}  # [E, H, Dh]
        g.apply_edges(weight_v)
        
        g.update_all(fn.copy_e('_av', '_m'), fn.sum('_m', '_h'))
        
        h_ctx = g.ndata['_h'].reshape(N, -1) # [N, d_model]
        
        # clean iup
        for k in ['_q', '_k', '_v', '_wb', '_h']:
            g.ndata.pop(k, None)
        for k in ['_score', '_alpha', '_av', '_m']:
            g.edata.pop(k, None)
            
        return h_ctx # [N, d_model]
    
    def forward(self, g, use_elem_cond=True, use_dist_bias=True):
        N   = g.num_nodes()
        dev = next(self.parameters()).device

        # 1. Frozen encode
        _, h_node = self.ssl_model.encoder(g)            # [N, 2560]

        # 2. Uniform initialization
        P = torch.ones(N, self.n_elem, device=dev) / self.n_elem   # [N, 23]

        logits = None
        for _ in range(self.K):
            # Soft element embedding
            if use_elem_cond:
                e_soft = P @ self.elem_embed.weight              # [N, d_elem]
            else:
                e_soft = torch.zeros(N, self.elem_embed.embedding_dim, device=dev)

            # Soft distance bias
            if use_dist_bias:
                W_b_soft = torch.einsum('nz,zhr->nhr', P, self.W_b)   # [N, H, rbf_dim]
            else:
                W_b_soft = torch.zeros(N, self.n_heads, self.W_b.shape[-1], device=dev)

            h_ctx  = self._attend_all(g, h_node, W_b_soft)      # [N, d_model]
            h_ctx  = self.W_o(h_ctx)                             # [N, 128]
            x      = torch.cat([h_node, h_ctx, e_soft], dim=-1)
            logits = self.head(x)                                 # [N, 23]
            P      = torch.softmax(logits, dim=-1)

        return logits   # [N, 23]
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_elem_cond',  action='store_true', help='Disable soft element conditioning')
    parser.add_argument('--no_dist_bias',  action='store_true', help='Disable distance bias (W_b)')
    parser.add_argument('--K',             type=int, default=3)
    args = parser.parse_args()

    USE_ELEM_COND = not args.no_elem_cond
    USE_DIST_BIAS = not args.no_dist_bias

    # Dynamic output dir based on ablation flags
    suffix = ''
    if not USE_ELEM_COND: suffix += '_no_elem'
    if not USE_DIST_BIAS: suffix += '_no_dist'
    OUT_DIR = f'ase_checkpoints_iterative_refiner{suffix}'
    print(f"use_elem_cond={USE_ELEM_COND}  use_dist_bias={USE_DIST_BIAS}  OUT_DIR={OUT_DIR}")

    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS     = 300
    LR         = 1e-4
    BATCH_SIZE = 256
    PATIENCE   = 20
    K          = args.K

    train_dataset = RefinerDataset(ND_TRAIN)
    val_dataset   = RefinerDataset(ND_VAL)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  collate_fn=refiner_collate_fn, num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=refiner_collate_fn, num_workers=4)

    model     = IterativeRefiner(METALLIC_CKPT, K=K).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss, patience_counter = float('inf'), 0
    train_losses, val_losses = [], []
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss, n_batches = 0.0, 0
        for g, labels in tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False):
            g, labels = g.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(g, use_elem_cond=USE_ELEM_COND, use_dist_bias=USE_DIST_BIAS), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for g, labels in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False):
                g, labels = g.to(DEVICE), labels.to(DEVICE)
                logits    = model(g, use_elem_cond=USE_ELEM_COND, use_dist_bias=USE_DIST_BIAS)
                val_loss_total += criterion(logits, labels).item()
                mask    = labels >= 0
                preds   = logits[mask].argmax(dim=-1)
                val_correct += (preds == labels[mask]).sum().item()
                val_total   += mask.sum().item()

        scheduler.step()
        val_loss = val_loss_total / len(val_loader)
        val_acc  = val_correct / val_total if val_total > 0 else 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        is_best = val_loss < best_val_loss
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"Acc={val_acc:.4f}" + (" ← best" if is_best else ""))

        if is_best:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         val_loss,
                'val_acc':          val_acc,
                'config': {'n_elem': N_ELEM, 'K': K,
                           'vocab': ELEM_VOCAB, 'z_to_idx': Z_TO_IDX,
                           'use_elem_cond': USE_ELEM_COND,
                           'use_dist_bias': USE_DIST_BIAS},
            }, f'{OUT_DIR}/checkpoint_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('CrossEntropy')
    plt.legend()
    plt.savefig(f'{OUT_DIR}/loss_curve.png', dpi=150)
    plt.close()
    print(f"Best val loss: {best_val_loss:.4f}")