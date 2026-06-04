import os, random, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import dgl
from dgl.nn import edge_softmax 
import dgl.function as fn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from ase_model import TopoSSL
from ase_train_iterative_refiner import Z_TO_IDX, N_ELEM
from ase_prepare_tmqm_data import TRANSITION_METALS, N_TAB

TARGET_PROPS = ['Metal_q', 'HL_Gap', 'HOMO_Energy', 'LUMO_Energy']

N_DFT  = 2
D      = 2560                  # TopoSSL output (512 × 5 layers)
D_AUG  = D + N_TAB + N_DFT    # 2565

METALLIC_CKPT = 'ase_checkpoints_phase2/checkpoint_best.pt'
DATA_DIR      = 'data/processed_data_2/tmqm'
OUT_DIR       = 'ase_checkpoints_refiner_v2'

BATCH_SIZE        = 1024
LR                = 3e-4
WEIGHT_DECAY      = 1e-5
EPOCHS            = 100
PATIENCE          = 15
RECORDS_PER_EPOCH = 200_000
LAMBDA_REG   = 0.5   # weight for regression vs repair loss

random.seed(42)
torch.manual_seed(42)

# Model 
class IterativeRefinerV2(nn.Module):
    def __init__(self, metalic_ckpt, n_elem=N_ELEM, 
                 n_heads=8, d_head=64, d_elem=32, rbf_dim=50, K=3,
                 n_props=len(TARGET_PROPS)):
        super().__init__()
        d_model = n_heads * d_head # 512 
        
        # Frozen TopoSSL encoder
        self.ssl_model = TopoSSL(
            num_features=53, hidden_dim=512, num_layers=5,
            proj_dim=256, pool='mean'
        )
        ckpt = torch.load(metalic_ckpt, map_location='cpu')
        if 'encoder_state_dict' in ckpt:
            self.ssl_model.load_state_dict(ckpt['encoder_state_dict'])
        elif 'model_state_dict' in ckpt:
            self.ssl_model.load_state_dict(ckpt['model_state_dict'])
        for p in self.ssl_model.parameters():
            p.requires_grad = False
        
        # Attention projections - input is D_AUG 
        self.W_q = nn.Linear(D_AUG, d_model, bias=False)
        self.W_k = nn.Linear(D_AUG, d_model, bias=False)
        self.W_v = nn.Linear(D_AUG, d_model, bias=False)
        self.W_p = nn.Linear(n_elem, D_AUG) # project P into QK space
        
        self.W_b = nn.Parameter(torch.zeros(n_elem, n_heads, rbf_dim))
        nn.init.normal_(self.W_b, std=0.01)
        
        self.W_o = nn.Linear(d_model, 128, bias=False)
        #self.W_ctx = nn.Linear(128, D_AUG)  # project h_ctx back for iterative update
        self.elem_embed = nn.Embedding(n_elem, d_elem)
        
        # Element repair head [N, n_elem]
        self.head = nn.Sequential(
            nn.Linear(D_AUG + 128 + d_elem, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_elem),
        )
        
        # Regression head [n_props]
        self.regression_head = nn.Sequential(
            nn.Linear(D_AUG + 128 + d_elem, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_props),
        )
        
        self.n_heads = n_heads
        self.d_head = d_head
        self.K = K
        self.n_elem = n_elem
        self.n_props = n_props
        self.scale = d_head ** -0.5
        
    def _attend_all(self, g, h, W_b_soft):
        N = h.shape[0]
        H, Dh = self.n_heads, self.d_head
        Q = self.W_q(h).view(N, H, Dh)
        K = self.W_k(h).view(N, H, Dh)
        V = self.W_v(h).view(N, H, Dh)
        g.ndata.update({'_q': Q, '_k': K, '_v': V, '_wb': W_b_soft})

        def compute_scores(edges):
            scores = (edges.dst['_q'] * edges.src['_k']).sum(-1) * self.scale
            if 'rbf' in edges.data:
                bias = (edges.dst['_wb'] * edges.data['rbf'].unsqueeze(1)).sum(-1)
                scores = scores + bias
            return {'_score': scores}

        g.apply_edges(compute_scores)
        g.edata['_alpha'] = edge_softmax(g, g.edata['_score'])
        g.apply_edges(lambda e: {'_av': e.src['_v'] * e.data['_alpha'].unsqueeze(-1)})
        g.update_all(fn.copy_e('_av', '_m'), fn.sum('_m', '_h'))
        h_ctx = g.ndata['_h'].reshape(N, -1)

        for k in ['_q', '_k', '_v', '_wb', '_h']:   g.ndata.pop(k, None)
        for k in ['_score', '_alpha', '_av', '_m']:  g.edata.pop(k, None)
        return h_ctx

    def forward(self, g, metal_mask):
        """
        g:          batched DGL graph
        metal_mask: [N_total] bool — True at TM atom positions
        Returns:
            logits  [N_total, n_elem]   — for element repair loss
            reg_out [B, n_props]        — for regression loss
        """
        dev = next(self.parameters()).device
        N, B = g.num_nodes(), g.batch_size

        _, h_node = self.ssl_model.encoder(g)              # [N, 2560]
        h_aug = torch.cat([
            h_node,
            g.ndata['tab_feat'].to(dev),                   # [N, 3]
            g.ndata['dft_feat'].to(dev),                   # [N, 2]
        ], dim=-1)                                          # [N, 2565]

        P = torch.ones(N, self.n_elem, device=dev) / self.n_elem
        #h_iter = h_aug                                                  # evolves each iteration
        x = None
        for _ in range(self.K):
            e_soft   = P @ self.elem_embed.weight
            W_b_soft = torch.einsum('nz,zhr->nhr', P, self.W_b)
            P_hard = F.one_hot(P.argmax(-1), self.n_elem).float()
            P_ste = P + (P_hard - P).detach()
            h_iter = h_aug + self.W_p(P_ste)
            h_ctx    = self.W_o(self._attend_all(g, h_iter, W_b_soft))  # [N, 128]
            x        = torch.cat([h_aug, h_ctx, e_soft], dim=-1)        # [N, 2725]
            logits   = self.head(x)                                      # [N, n_elem]
            P        = torch.softmax(logits, dim=-1)                     # residual update

        # Mean-pool metal nodes per graph → regression
        offsets = torch.zeros(B, dtype=torch.long, device=dev)
        if B > 1:
            offsets[1:] = g.batch_num_nodes().cumsum(0)[:-1]
        reg_outs = []
        for b in range(B):
            s = offsets[b].item()
            e = s + g.batch_num_nodes()[b].item()
            h_metal = x[s:e][metal_mask[s:e]]
            feat = h_metal.mean(0) if h_metal.shape[0] > 0 \
                   else torch.zeros(x.shape[-1], device=dev)
            reg_outs.append(self.regression_head(feat))
        reg_out = torch.stack(reg_outs)   # [B, n_props]

        return logits, reg_out
    
class tmQMDataset(Dataset):
    def __init__(self, pt_path, target_stats=None):
        data = torch.load(pt_path)
        self.records = data['records']
        self.target_stats = target_stats or self._compute_stats()
        
    def _compute_stats(self):
        vals = {p: [] for p in TARGET_PROPS}
        for r in self.records:
            if not r['corrupted_sites']:
                for p, v in r['targets'].items():
                    if p in vals:
                        vals[p].append(v)
        return {p: {'mean': float(np.mean(v or [0])),
                    'std': float(np.std(v or [0])) + 1e-8} for p,v in vals.items()}
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        g = r['graph']
        
        # Ground-truth elements: restore z_old at corrputed sites 
        z_full = g.ndata['atomic_num'].clone()
        for site, z_o in zip(r['corrupted_sites'], r['z_old']):
            z_full[site] = z_o  
        
        elem_targets= torch.tensor(
            [Z_TO_IDX.get(z.item(), -1) for z in z_full], dtype=torch.long
        )
        metal_mask = torch.tensor(
            [z.item() in TRANSITION_METALS for z in z_full], dtype=torch.bool
        )
        
        #Normalized regression targets
        target_vec = torch.full((len(TARGET_PROPS),), float('nan'))
        if not r['corrupted_sites']:
            for i, p in enumerate(TARGET_PROPS):
                if p in r['targets']:
                    s = self.target_stats[p]
                    target_vec[i] = (r['targets'][p] - s['mean']) / s['std']

        corr_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        for site in r['corrupted_sites']:
            corr_mask[site] = True

        return {'graph': g, 'elem_targets': elem_targets,
                'metal_mask': metal_mask, 'target_vec': target_vec,
                'is_clean': len(r['corrupted_sites']) == 0,
                'corr_mask': corr_mask}
        
def collate_fn(batch):
    return (
        dgl.batch([s['graph']        for s in batch]),
        torch.cat( [s['elem_targets'] for s in batch]),
        torch.cat( [s['metal_mask']   for s in batch]),
        torch.stack([s['target_vec']  for s in batch]),
        torch.tensor([s['is_clean']   for s in batch]),
        torch.cat( [s['corr_mask']   for s in batch]),
    )
    
def run_epoch(model, loader, optimizer, device, target_stats, class_weights, train=True):
    model.train(train)
    tot_loss = tot_rep = tot_reg = 0.0
    tot_correct = tot_valid = 0
    corr_correct = corr_total = 0
    clean_correct = clean_total = 0
    all_pred, all_tgt = [], []

    with (torch.enable_grad() if train else torch.no_grad()):
        for graphs, elem_tgt, metal_mask, target_vec, is_clean, corr_mask in tqdm(loader, leave=False):
            graphs, elem_tgt, metal_mask = \
                graphs.to(device), elem_tgt.to(device), metal_mask.to(device)
            target_vec, is_clean = target_vec.to(device), is_clean.to(device)
            corr_mask = corr_mask.to(device)

            # dft_feat dropout: 50% of training steps zero out DFT features
            # forces model to rely on tab_feat + attention, matching 3DED inference
            if train and random.random() < 0.5:
                graphs.ndata['dft_feat'] = torch.zeros_like(graphs.ndata['dft_feat'])

            logits, reg_out = model(graphs, metal_mask)

            valid    = elem_tgt >= 0
            L_repair = F.cross_entropy(logits[valid], elem_tgt[valid], weight=class_weights)

            # repair accuracy (all valid nodes)
            preds = logits[valid].argmax(dim=-1)
            tot_correct += (preds == elem_tgt[valid]).sum().item()
            tot_valid   += valid.sum().item()

            # corrupted-site accuracy (only nodes that were actually corrupted)
            corr_valid = valid & corr_mask
            if corr_valid.any():
                corr_preds = logits[corr_valid].argmax(dim=-1)
                corr_correct += (corr_preds == elem_tgt[corr_valid]).sum().item()
                corr_total   += corr_valid.sum().item()

            # clean-site accuracy (nodes that were NOT corrupted)
            clean_valid = valid & ~corr_mask
            if clean_valid.any():
                clean_preds = logits[clean_valid].argmax(dim=-1)
                clean_correct += (clean_preds == elem_tgt[clean_valid]).sum().item()
                clean_total   += clean_valid.sum().item()

            L_reg = torch.tensor(0.0, device=device)
            if is_clean.any():
                pred = reg_out[is_clean]
                tgt  = target_vec[is_clean]
                mask = ~torch.isnan(tgt)
                if mask.any():
                    L_reg = F.mse_loss(pred[mask], tgt[mask])
                all_pred.append(reg_out[is_clean].detach().cpu())
                all_tgt.append(target_vec[is_clean].cpu())

            loss = L_repair + LAMBDA_REG * L_reg

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tot_loss += loss.item()
            tot_rep  += L_repair.item()
            tot_reg  += L_reg.item()

    n           = len(loader)
    repair_acc  = tot_correct / max(tot_valid, 1)
    corr_acc    = corr_correct / max(corr_total, 1)
    clean_acc   = clean_correct / max(clean_total, 1)

    # per-property MAE on original scale
    prop_mae = {}
    if all_pred:
        pred_cat = torch.cat(all_pred)   # [n_clean, 4]
        tgt_cat  = torch.cat(all_tgt)    # [n_clean, 4]
        for i, p in enumerate(TARGET_PROPS):
            mask = ~torch.isnan(tgt_cat[:, i])
            if mask.any():
                s         = target_stats[p]
                pred_orig = pred_cat[mask, i] * s['std'] + s['mean']
                tgt_orig  = tgt_cat[mask,  i] * s['std'] + s['mean']
                prop_mae[p] = (pred_orig - tgt_orig).abs().mean().item()

    return tot_loss/n, tot_rep/n, tot_reg/n, repair_acc, corr_acc, clean_acc, prop_mae


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading datasets …")
    train_ds = tmQMDataset(f'{DATA_DIR}/train_tmqm.pt')
    val_ds   = tmQMDataset(f'{DATA_DIR}/val_tmqm.pt',
                           target_stats=train_ds.target_stats)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    # Compute class weights (inverse frequency, normalized to mean=1)
    print("Computing class weights …")
    label_cnt = Counter()
    for r in train_ds.records:
        z_vec = (r['graph'].ndata['h'][:,0] * 100).round().long().tolist()
        for z in z_vec:
            idx = Z_TO_IDX.get(z, -1)
            if idx >= 0:
                label_cnt[idx] += 1
    total_labels = sum(label_cnt.values())
    class_weights = torch.ones(N_ELEM, device=device)
    for idx, cnt in label_cnt.items():
        class_weights[idx] = total_labels / (N_ELEM * cnt)
    class_weights = class_weights / class_weights.mean()
    print(f"  Class weights: {class_weights}")

    print("Building model …")
    model = IterativeRefinerV2(METALLIC_CKPT).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable: {sum(p.numel() for p in trainable):,} (encoder frozen)")

    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    stats = train_ds.target_stats
    best_val, patience_cnt = float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        # Subsample training data each epoch for speed (different subset every time)
        if len(train_ds) > RECORDS_PER_EPOCH:
            indices = random.sample(range(len(train_ds)), RECORDS_PER_EPOCH)
            epoch_loader = DataLoader(Subset(train_ds, indices), BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_fn, num_workers=4)
        else:
            epoch_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_fn, num_workers=4)

        tr = run_epoch(model, epoch_loader, optimizer, device, stats, class_weights, train=True)
        va = run_epoch(model, val_loader,   optimizer, device, stats, class_weights, train=False)
        scheduler.step()

        tr_loss, tr_rep, tr_reg, tr_acc, tr_corr, tr_clean, _        = tr
        va_loss, va_rep, va_reg, va_acc, va_corr, va_clean, va_mae   = va

        mae_str = '  '.join(f'{p}={va_mae.get(p, float("nan")):.4f}'
                            for p in TARGET_PROPS)
        print(f"Epoch {epoch:3d}"
              f"  train loss={tr_loss:.4f} rep={tr_rep:.4f} reg={tr_reg:.4f}"
              f"  acc={tr_acc:.3f} corr={tr_corr:.3f} clean={tr_clean:.3f}"
              f"\n         val   loss={va_loss:.4f} rep={va_rep:.4f} reg={va_reg:.4f}"
              f"  acc={va_acc:.3f} corr={va_corr:.3f} clean={va_clean:.3f}"
              f"\n         MAE: {mae_str}")

        # monitor on val corrupted-site accuracy (the metric that matters)
        monitor = -va_corr
        if monitor < best_val:
            best_val, patience_cnt = monitor, 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_stats':     stats,
                'config': {'n_elem': N_ELEM, 'K': 3,
                           'n_props': len(TARGET_PROPS),
                           'target_props': TARGET_PROPS},
            }, f'{OUT_DIR}/checkpoint_best.pt')
            print(f"  saved best val corr_acc={va_corr:.3f} (overall={va_acc:.3f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop @ epoch {epoch}")
                break

    print(f"Done. Best val acc: {-best_val:.3f}")


if __name__ == '__main__':
    main()