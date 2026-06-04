import dgl
import torch 
import numpy as np
from ase import Atoms

def gaussian_rbf(distances, centers, width):
    distances = np.array(distances).reshape(-1, 1)
    centers = np.array(centers).reshape(1, -1)
    return np.exp(-((distances - centers) ** 2) / (2 * width ** 2))

def build_coord_graph(atoms, coord_cutoff=3.5, rbf_params=None):
    """Build coordination DGL graph using fixed distance cutoff.
    Captures metal-ligand proximity (at ~2.0 Å) that is invisible to the
    bond graph (mult=0.85 excludes metal-ligand bonds).
    
    Used ONLY for CEP attention — not for S_topo encoder.
    No node features needed; only edge connectivity + rbf distance features.

    Parameters:
    - atoms     : ASE Atoms
    - coord_cutoff : float, max distance for coordination edges (default 3.5 Å)
    - rbf_params: same format as MolecularGraphBuilder (50 centers, 0–6 Å)

    Returns: DGL graph with g.edata['rbf'] shape [n_edges, 50]
    """
    if rbf_params is None:
        rbf_params = {'start': 0.0, 'stop': 6.0, 'num_centers': 50, 'width': 0.3}
        
    n = len(atoms)
    pos = atoms.get_positions()
    src_list, dst_list, dists = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist <= coord_cutoff:
                src_list.append(i)
                dst_list.append(j)
                dists.append(dist)
    g = dgl.graph((src_list, dst_list), num_nodes=n)
    
    centers = np.linspace(rbf_params['start'], rbf_params['stop'],
                          rbf_params['num_centers'])
    rbf = gaussian_rbf(dists, centers, rbf_params['width']) #[E, 50]
    g.edata['rbf'] = torch.tensor(rbf, dtype=torch.float32)
    
    return g
if __name__ == '__main__':
    pos = [[0,0,0], [2,0,0], [-2,0,0], [0,2,0], [0,-2,0]]
    atoms = Atoms('CoNNNN', positions=pos)
    g = build_coord_graph(atoms)
    print(g)
    print(g.edata['rbf'].shape)
    _, nbrs, eids = g.out_edges(0, form='all')
    print(f"Co neighbors:{nbrs.tolist()}")
