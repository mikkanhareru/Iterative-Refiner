import json, os, random
import torch 
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F

RARE_ELEMENT = {28, 29, 30, 44, 45, 46, 47}
ELEM_BINS =  [1, 6, 7, 8, 9, 15, 16, 17, 35, 26, 27]                                   

N_VOCAB = 13
Z_TO_IDX = {z: i for i, z in enumerate(ELEM_BINS)}
OUT_DIR = 'ase_checkpoints_elem_prior'

def build_histogram(nbr_z):
    """nbr_z: 1D tensor / vector of neighbor atomic number -> float histogram [N_VOCAB]
    """
    hist = torch.zeros(N_VOCAB, dtype=torch.float)
    for z_val in nbr_z.tolist():
        if z_val in Z_TO_IDX:
            idx = Z_TO_IDX.get(z_val, N_VOCAB - 1)
        elif z_val in RARE_ELEMENT:
            idx = 11
        else:
            idx = 12
        hist[idx] += 1.0
    return hist

def z_to_idx(z_val):
    """Map atomic number to vocab class index (for target label)"""
    if z_val in Z_TO_IDX:
        return Z_TO_IDX[z_val]
    elif z_val in RARE_ELEMENT:
        return 11 # combine rare metal together
    else:
        return 12

class ElementPrior(nn.Module):
    def __init__(self, n_vocab=N_VOCAB, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vocab, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_vocab)
        )
    def forward(self, x): #x:[B, N_VOCAB] histograms
        return F.log_softmax(self.net(x), dim=-1) # [B, N_VOCAB] log-probs

    def score(self, x, y_idx): # logP(z_i | neighbors)
        return self(x)[torch.arange(len(y_idx)), y_idx] # [B]

if __name__ ==  '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    data = torch.load('data/processed_data_2/metallic_graphs/train_data.pt')
    graphs = data['graphs']
    labels = data['labels']

    X, Y = [], []

    for g, label in tqdm(zip(graphs, labels), total=len(graphs)):
        if label != 1:
            continue
        z = g.ndata['atomic_num'].cpu() #[N]
        src, dst = g.edges()
        src, dst = src.cpu(), dst.cpu()
        
        for i in range(g.num_nodes()):
            nbr_z = z[dst[src == i]] # get neightbor elements of atom i
            X.append(build_histogram(nbr_z))
            Y.append(z_to_idx(z[i].item()))
            
    X = torch.stack(X) #[N_total_atoms, N_VOCAB]
    Y = torch.tensor(Y, dtype=torch.long) #[N_total_atoms]
    print(f"Extracted {len(Y)} atoms")
    print(f"Class distribution: {torch.bincount(Y)}")
    dataset = TensorDataset(X, Y)
    n_trian = int(0.85 * len(dataset))
    n_val = len(dataset) - n_trian
    train_dataset, val_dataset = random_split(dataset, [n_trian, n_val],
                                generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size = 4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100 
    LR = 1e-3
    PATIENCE  = 20

    model = ElementPrior(n_vocab=N_VOCAB, hidden=64).to(DEVICE)

    # inverse-frequency weights, capped at 50
    counts = torch.bincount(Y, minlength=N_VOCAB).float()
    weights = (Y.numel() / (N_VOCAB * counts)).clamp(max=50).to(DEVICE)
    criterion = nn.NLLLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses  = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches
        
        model.eval()
        val_loss_total, n_val = 0.0, 0
        n_correct = 0 
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logprobs = model(x_batch)
                val_loss_total += criterion(logprobs, y_batch).item()
                n_correct += (logprobs.argmax(dim=1)==y_batch).sum().item()
                n_val += 1
        val_loss = val_loss_total / n_val
        val_acc = n_correct / len(val_dataset)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.4f}"
            + (" ← best" if val_loss < best_val_loss else ""))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'config': {'n_vocab': N_VOCAB, 'hidden': 64},
                        'elem_bins': ELEM_BINS, 'merge_to_rare': list(RARE_ELEMENT)},
                    f'{OUT_DIR}/checkpoint_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Best val loss: {best_val_loss:.4f}")
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.savefig(f'{OUT_DIR}/loss_curve.png', dpi=150)
    plt.close()
    print(f"Best val loss: {best_val_loss:.4f}")