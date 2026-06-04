import json, os, random
import torch
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset 

from ase_topo_verifier import TopoVerifier
from ase_finetune import ensure_node_features
from ase_model import TopoSSL

ELEM_EMB = {6:0, 7:1, 8:2, 17:3, 35:4} # z -> emb index

class CEPDataset(Dataset):
    def __init__(self, path, mol_indcies):
        super().__init__()
        data = torch.load(path)
        self.graphs = data['graphs']
        self.coord_graphs = data['coord_graphs']
        mol_set = set(mol_indcies)
        mask = [i for i, m in enumerate(data['mol_idx']) if m in mol_set]
        self.mol_idx = [data['mol_idx'][i] for i in mask]
        self.site_idx = [data['site_idx'][i] for i in mask]
        self.z_new = [data['z_new'][i] for i in mask]
        self.delta_s = [data['delta_s'][i] for i in mask]
        
    def __len__(self):
        return len(self.mol_idx)
    
    def __getitem__(self, i):
        g = self.graphs[self.mol_idx[i]]
        g_coord = self.coord_graphs[self.mol_idx[i]]
        site = self.site_idx[i]
        z_new = ELEM_EMB[self.z_new[i]]
        delta_s = self.delta_s[i]
        return g, g_coord, site, z_new, delta_s
    
def cep_collate_fn(batch):
    graphs, coord_graphs, sites, z_news, delta_ss = zip(*batch)
    graphs = [ensure_node_features(g, 53) for  g in graphs]
    coord_graphs = [ensure_node_features(g, 53) for g in coord_graphs]
    batched_g = dgl.batch(graphs)
    batched__coord_g = dgl.batch(coord_graphs)
    return (batched_g,
            batched__coord_g,
            torch.tensor(sites, dtype=torch.long),
            torch.tensor(z_news, dtype=torch.long),
            torch.tensor(delta_ss, dtype=torch.float))
    
class CEPModelV3(nn.Module):
    """hiterarchical two scale multi-head element-conditional attention.
    Innter shell atten d <= INNER_CUTOFF: coordination-sphere neighbors
    Outer shell d >= INNER_CUTOFF: long-range environment neightbors 
    
    Both scales share the same Q/K/V projections but have separate W_b matrices
    A learned sigmoid gate \alpha mixes inner and outer conext vectors.
    Can extend to multi-head with W_o aggregations
    """
    INNER_CUTOFF = 2.5 # separates coordination shell from outer shell
    
    def __init__(self, 
                 encoder_checkpoint_path,
                 hidden_dim=512,
                 num_layers=5,
                 proj_dim=256,
                 n_elem_types=5, # C,N,O,Cl,Br,
                 n_heads=8,
                 d_head=64, # per head atten dim; total = n_heads * d_head = 256
                 d_elem=16,
                 rbf_dim=50
                 ):
        super().__init__()
        
        # frozen encoder 
        self.ssl_model = TopoSSL(
            num_features=53,
            hidden_dim=512,
            num_layers=5,
            proj_dim=256,
            pool='mean'
        )
        ckpt = torch.load(encoder_checkpoint_path, map_location='cpu')
        self.ssl_model.load_state_dict(ckpt['encoder_state_dict'])
        for p in self.ssl_model.parameters():
            p.requires_grad = False
            
        D = hidden_dim * num_layers # 2560
        d_model = n_heads * d_head # 8 * 64 = 512
        
        # Multi-heda QKV projections 
        self.W_q = nn.Linear(D, d_model, bias=False)
        self.W_k = nn.Linear(D, d_model, bias=False)
        self.W_v = nn.Linear(D, d_model, bias=False)
        
        # Element-conditioned distance bias: one per scale 
        # W_b_inner[z, h, :]
        self.W_b_inner = nn.Parameter(torch.zeros(n_elem_types, n_heads, rbf_dim))
        self.W_b_outer = nn.Parameter(torch.zeros(n_elem_types, n_heads, rbf_dim))
        nn.init.normal_(self.W_b_inner, std=0.01)
        nn.init.normal_(self.W_b_outer, std=0.01)
        
        # Head aggregation: [n_heads * d_head -> 128]
        self.W_o = nn.Linear(d_model, 128, bias=False)
        
        # sclae mixing gate: learned sigmoid scalar 
        self.scale_gate = nn.Linear(D, 1)
        
        # Element embedding 
        self.elem_embed = nn.Embedding(n_elem_types, d_elem)
        
        # Regression head
        head_in = D + D + 128 + d_elem
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.scale = d_head ** -0.5
        self.rbf_dim = rbf_dim
        
        # RBF centers for distance decoding 
        rbf_centers = torch.linspace(0.5, 6.0, rbf_dim)
        self.register_buffer('rbf_centers', rbf_centers)
        
    def _decode_dist(self, phi):
        '''
        Approximate raw distance d from RBF features via weighted mean
        phi: [n_nbr. rbf_dim] -> [n_nbr]
        '''
        return (phi * self.rbf_centers).sum(-1) / (phi.sum(-1) + 1e-8)

    def _attend(self, q_b, K, V, phi, W_b_z):
        """Multi-head element-condiotioned attention for one scale.

        q_b : [d_model]
        K, V : [n_nbr, d_model]
        phi: [n_nbr, rbf_dim]
        W_b_z: [n_heads, rbf_dim]
        """
        H, Dh = self.n_heads, self.d_head
        n_nbr = K.shape[0]

        # Reshape to [H, n_nbr, Dh]
        q_h = q_b.view(H, Dh)
        K_h = K.view(n_nbr, H, Dh).permute(1, 0, 2)
        V_h = V.view(n_nbr, H, Dh).permute(1, 0, 2)  # [H, n_nbr, Dh]

        # Attention scores: [H, n_nbr]
        scores = torch.einsum('hd,hnd->hn', q_h, K_h) * self.scale

        # Element-conditioned distance bias: W_b_z [H, rbf_dim] x phi [n_nbr, rbf_dim]
        bias = torch.einsum('hr,nr->hn', W_b_z, phi)
        scores = scores + bias

        alpha = torch.softmax(scores, dim=-1)

        # weighted sum: [H, Dh]
        ctx_h = torch.einsum('hn,hnd->hd', alpha, V_h)

        return ctx_h.reshape(self.d_model)

    def forward(self, g, g_coord, site_idx, z_new_idx):
        B   = site_idx.shape[0]
        dev = site_idx.device

        # 1. Encode
        h_graph, h_node = self.ssl_model.encoder(g)   # [B, 2560], [N_total, 2560]

        # 2. Global site indices
        offsets = torch.zeros(B, dtype=torch.long, device=dev)
        offsets[1:] = g.batch_num_nodes().cumsum(0)[:-1].to(dev)
        g_site = site_idx + offsets   # [B]

        # 3. Site embedding + projections
        h_site = h_node[g_site]       # [B, 2560]
        Q = self.W_q(h_site)          # [B, d_model]
        h_node_K = self.W_k(h_node)   # [N_total, d_model]
        h_node_V = self.W_v(h_node)   # [N_total, d_model]

        # 4. Scale gate (uses raw site embedding, not projected)
        alpha_gate = torch.sigmoid(self.scale_gate(h_site)).squeeze(-1)  # [B]

        # 5. Per-sample hierarchical attention
        h_ctx = torch.zeros(B, self.d_model, device=dev)
        for b in range(B):
            _, nbr_nodes, edge_ids = g_coord.out_edges(g_site[b].item(), form='all')
            if nbr_nodes.numel() == 0:
                continue

            K   = h_node_K[nbr_nodes]              # [n_nbr, d_model]
            V   = h_node_V[nbr_nodes]              # [n_nbr, d_model]
            phi = g_coord.edata['rbf'][edge_ids]   # [n_nbr, rbf_dim]
            d   = self._decode_dist(phi)            # [n_nbr]

            z   = z_new_idx[b].item()
            W_bi = self.W_b_inner[z]               # [n_heads, rbf_dim]
            W_bo = self.W_b_outer[z]               # [n_heads, rbf_dim]

            # Split into inner / outer shells
            inner_mask = d <= self.INNER_CUTOFF
            outer_mask = ~inner_mask

            q_b = Q[b]  # [d_model]

            if inner_mask.any():
                h_inner = self._attend(q_b,
                                       K[inner_mask], V[inner_mask],
                                       phi[inner_mask], W_bi)
            else:
                h_inner = torch.zeros(self.d_model, device=dev)

            if outer_mask.any():
                h_outer = self._attend(q_b,
                                       K[outer_mask], V[outer_mask],
                                       phi[outer_mask], W_bo)
            else:
                h_outer = torch.zeros(self.d_model, device=dev)

            # Learned mixing: α * inner + (1-α) * outer
            a = alpha_gate[b]
            h_ctx[b] = a * h_inner + (1.0 - a) * h_outer

        # 6. Head aggregation
        h_ctx = self.W_o(h_ctx)    # [B, 128]

        # 7. Element embedding
        e = self.elem_embed(z_new_idx)   # [B, 16]

        # 8. Regression
        x = torch.cat([h_graph, h_site, h_ctx, e], dim=-1)  # [B, 5264]
        return self.head(x).squeeze(-1)  # [B]
    
if __name__ == '__main__':
    CF_PATH    = 'data/processed_data_2/counterfactual/train_cf_v2.pt'
    OUT_DIR    = 'ase_checkpoints_cep_v3'
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS     = 300
    LR         = 1e-4
    BATCH_SIZE = 256
    PATIENCE   = 20

    meta   = torch.load(CF_PATH)
    n_mols = max(meta['mol_idx']) + 1
    mol_ids = list(range(n_mols))
    random.shuffle(mol_ids)
    split = int(0.8 * n_mols)
    train_dataset = CEPDataset(CF_PATH, mol_ids[:split])
    val_dataset   = CEPDataset(CF_PATH, mol_ids[split:])
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  collate_fn=cep_collate_fn, num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=cep_collate_fn, num_workers=0)

    model = CEPModelV3('ase_checkpoints_metallic/checkpoint_best.pt').to(DEVICE)

    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                            eta_min=1e-6)

    best_val_loss, patience_counter = float('inf'), 0
    train_losses, val_losses = [], []
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss, n_batches = 0.0, 0
        for g, g_coord, site, z_new, delta_s in tqdm(train_loader,
                                                      desc=f"Epoch {epoch:3d} [train]",
                                                      leave=False):
            g, site, z_new, delta_s = (g.to(DEVICE), site.to(DEVICE),
                                        z_new.to(DEVICE), delta_s.to(DEVICE))
            g_coord = g_coord.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(g, g_coord, site, z_new), delta_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        val_loss_total, val_mae_total, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for g, g_coord, site, z_new, delta_s in tqdm(val_loader,
                                                          desc=f"Epoch {epoch:3d} [val]  ",
                                                          leave=False):
                g, site, z_new, delta_s = (g.to(DEVICE), site.to(DEVICE),
                                            z_new.to(DEVICE), delta_s.to(DEVICE))
                g_coord = g_coord.to(DEVICE)
                pred = model(g, g_coord, site, z_new)
                val_loss_total += criterion(pred, delta_s).item()
                val_mae_total  += (pred - delta_s).abs().mean().item()
                n_val += 1
        scheduler.step()
        val_loss = val_loss_total / n_val
        val_mae  = val_mae_total  / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | MAE={val_mae:.4f}"
              + (" ← best" if val_loss < best_val_loss else ""))

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'config': {'hidden_dim': 512, 'num_layers': 5, 'proj_dim': 256,
                                   'n_heads': 4, 'd_head': 64, 'elem_vocab': ELEM_EMB}},
                       f'{OUT_DIR}/checkpoint_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.savefig(f'{OUT_DIR}/loss_curve.png', dpi=150)
    plt.close()
    print(f"Best val loss: {best_val_loss:.4f}")
            
            
        
        
    