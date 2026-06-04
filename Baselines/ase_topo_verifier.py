import json
import torch 
import torch.nn.functional as F
import dgl
import numpy as np
from ase_finetune import SSLClassifier, ensure_node_features
from ase_sa import reconstruct_molecular_graph

class TopoVerifier:
    def __init__(self, checkpoint_path, ssl_checkpoint_path, bond_config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.multiplier = json.load(open(bond_config_path))['bond_multiplier']
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = ckpt['config']
        
        # build architecture using base SSL checkpoint
        self.model = SSLClassifier(
            ssl_checkpoint_path=ssl_checkpoint_path,
            num_classes=cfg['num_classes'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            proj_dim=cfg['proj_dim'],
            freeze_encoder=True
        ).to(self.device)
    
        # orverride all weights with fine-tuned metallic checkpoint
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
    def score(self, pos, atomic_numbers) -> float:
        """accepts numpy arrays or lists 

        Args:
            pos (_type_): atomic position 
            atomic_numbers (_type_): atomic number

        Returns:
            float: S_topo in [0, 1]
        """
        z = torch.tensor(np.array(atomic_numbers), dtype=torch.long)
        p = torch.tensor(np.array(pos), dtype=torch.float)
        
        _, g = reconstruct_molecular_graph(z, p, multiplier=self.multiplier)
        g = ensure_node_features(g, num_features=53).to(self.device)
        with torch.no_grad():
            logits = self.model(g)
            return  F.softmax(logits, dim=1)[0, 1].item()
    
    def score_batch(self, graphs) -> list:
        # accepts list of pre-built DGL geometry graphs, for sanity check only
        graphs = [ensure_node_features(g, 53) for g in graphs]
        batched = dgl.batch(graphs).to(self.device)
        with torch.no_grad():
            logits = self.model(batched)
            return F.softmax(logits, dim=1)[:, 1].tolist()

if __name__ == '__main__':
    verifier = TopoVerifier(
        checkpoint_path='ase_checkpoints_metallic/checkpoint_best.pt',
        ssl_checkpoint_path='ase_checkpoints_tuned/checkpoint_best.pt',
        bond_config_path='results/graph_calibrate/bond_config.json',
        device = 'cuda'
    )
    data  = torch.load('data/processed_data_2/metallic_graphs/test_data.pt')
    g, label = data['graphs'][0], data['labels'][0]
    #pos = g.ndata['h'][:,0].cpu().numpy()
    print(f"Label={label}, model.eval mode: {not verifier.model.training}")
    scores = verifier.score_batch([g])
    print(f"S_topo = {scores[0]:.4f}  (label={'clean' if label==1 else 'corrupted'})")
        