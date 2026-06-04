import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, json, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import dgl
from tqdm import tqdm
import matplotlib.pyplot as plt 
from ase import Atoms
import warnings
warnings.filterwarnings('ignore')


# Reuse utilities from baseline
from ase_sa import (reconstruct_molecular_graph, filter_fragment,
                    element_label, save_composition_figure,
                    save_score_change_figure, print_score_distribution,
                    print_score_change)

from ase_topo_verifier import TopoVerifier
from ase_element_prior import ElementPrior, build_histogram, z_to_idx, N_VOCAB, ELEM_BINS
from ase_train_counterfactual_v3 import CEPModelV3 as CEPModel
from ase_finetune import ensure_node_features
from ase_coord_graph import build_coord_graph
    
# CEP candidate elements (same vocab as training)
ELEM_EMB  = {6:0, 7:1, 8:2, 17:3, 35:4}       # z → CEP vocab index
CEP_CANDS = list(ELEM_EMB.keys())               # [6, 7, 8, 17, 35]

TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))

ELEMENT_NAMES = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F',
                 15:'P', 16:'S', 17:'Cl', 26:'Fe', 27:'Co',
                 28:'Ni', 29:'Cu', 30:'Zn', 35:'Br', 53:'I'}

METAL_CANDS = [
    22, 23, 24, 25, 26, 27, 28, 29, 30,   # Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
    42, 44, 45, 46, 47, 48,                # Mo, Ru, Rh, Pd, Ag, Cd
    74, 78, 79,                            # W, Pt, Au
]

class MaskedGNN(nn.Module):
    def __init__(self, encoder_checkpoint, hidden_dim=512, num_layers=5, n_vocab=N_VOCAB):
        super().__init__()
        from ase_model import TopoSSL
        self.ssl_model = TopoSSL(hidden_dim=hidden_dim, num_layers=num_layers, proj_dim=256)
        ckpt = torch.load(encoder_checkpoint, map_location='cpu')
        self.ssl_model.load_state_dict(ckpt['encoder_state_dict'])
        for p in self.ssl_model.parameters():
            p.requires_grad = False
        D = hidden_dim * num_layers
        self.head = nn.Linear(D, n_vocab)

    def forward(self, g, site_idx):
        dev = site_idx.device
        B   = site_idx.shape[0]
        g   = g.to(dev)
        _, h_node = self.ssl_model.encoder(g)
        offsets = torch.zeros(B, dtype=torch.long, device=dev)
        offsets[1:] = g.batch_num_nodes().cumsum(0)[:-1].to(dev)
        g_site = site_idx + offsets
        h_site = h_node[g_site]
        return F.log_softmax(self.head(h_site), dim=-1)

    def score(self, g, site_idx, y_idx):
        return self(g, site_idx)[torch.arange(len(y_idx), device=y_idx.device), y_idx]


SSL_TUNED_CKPT = 'ase_checkpoints_tuned/checkpoint_best.pt'
METALLIC_CKPT = 'ase_checkpoints_metallic/checkpoint_best.pt'
CEP_CKPT = 'ase_checkpoints_cep_v3/checkpoint_best.pt'
ELEM_PRIOR_CKPT = 'ase_checkpoints_elem_prior/checkpoint_best.pt'
MASKED_CKPT = 'ase_checkpoints_masked_elem/checkpoint_best.pt'
BOND_CONFIG = 'results/graph_calibrate/bond_config.json'

def load_all_models(device):
    print("Loading TopoVerifier...")
    topo_verifier = TopoVerifier(
        checkpoint_path=METALLIC_CKPT,
        ssl_checkpoint_path=SSL_TUNED_CKPT,
        bond_config_path=BOND_CONFIG,
        device=str(device)
    )
    
    print("Loading CEPModel")
    cep_model = CEPModel(encoder_checkpoint_path=METALLIC_CKPT)
    cep_ckpt = torch.load(CEP_CKPT, map_location='cpu')
    cep_model.load_state_dict(cep_ckpt['model_state_dict'])
    cep_model.to(device).eval()
    
    print("Loading ElementPrior...")
    elem_prior = ElementPrior(n_vocab=N_VOCAB, hidden=64)
    ep_ckpt = torch.load(ELEM_PRIOR_CKPT, map_location='cpu')
    elem_prior.load_state_dict(ep_ckpt['model_state_dict'])
    elem_prior.to(device).eval()
    
    print("Loading ablation MaskedGNN...")
    masked_GNN = MaskedGNN(encoder_checkpoint=METALLIC_CKPT, hidden_dim=512,
                           num_layers=5, n_vocab=N_VOCAB)
    masked_ckpt = torch.load(MASKED_CKPT, map_location='cpu')
    masked_GNN.load_state_dict(masked_ckpt['model_state_dict'])
    masked_GNN.to(device).eval()
    
    print("All models loaded")
    return topo_verifier, cep_model, elem_prior, masked_GNN

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# topo_verifer, cep_model, elem_prior, masked_GNN = load_all_models(DEVICE)

def identify_edit_sites(z: torch.Tensor, g=None,
                        min_degree_suspicious: int = 3) -> List[int]:
    """Return metal sites + suspicious non-metals.

    1. Transition metals (always)
    2. Cl/Br with degree >= min_degree_suspicious (Type B)
    3. C/N/O with >= min_NO_neighbors N/O atoms within metal coordination
       distance [1.7, 2.8] Å (Type A: metal→C/N/O disguise)
       Uses raw positions — not graph bonds — because C/N/O radii are too
       small to form bonds at Zn/Co-N distances after element swap.
    """
    sites = set(i for i in range(len(z)) if z[i].item() in TRANSITION_METALS)

    if g is not None:
        src, _ = g.edges()
        src = src.cpu()
        for i in range(len(z)):
            zi = z[i].item()
            degree = int((src == i).sum())
            if zi in (17, 35):
                if degree >= min_degree_suspicious:
                    sites.add(i)
            elif zi in (6, 7, 8):
                if degree == 0:
                    sites.add(i)

    return list(sites)



def guided_proposal(g, edit_sites: List[int], z: torch.Tensor,
                    cep_model, elem_prior, device, pos=None, g_coord=None,
                    beta_cep: float = 1.0, masked_gnn=None):
    """Unified proposal:
    - Metal sites   → CEP candidates {C, N, O, Cl, Br} (repair over-assigned metals)
    - Suspicious non-metals → METAL_CANDS ranked by ElementPrior (repair misassigned Cl/Br)
    Returns (site_idx: int, z_new: int) or (None, None).
    """
    if len(edit_sites) == 0:
        return None, None

    g_dev = ensure_node_features(g, 53).to(device)
    all_sites, all_z_new, all_scores = [], [], []

    # --- coord graph for CEP attention (use prebuilt if provided) ---
    if g_coord is None:
        if pos is not None:
            atoms   = Atoms(numbers=z.cpu().numpy(), positions=pos.cpu().numpy())
            g_coord = build_coord_graph(atoms).to(device)
        else:
            g_coord = g_dev

    # --- Metal sites: CEP scores ---
    metal_sites = [s for s in edit_sites if z[s].item() in TRANSITION_METALS]
    if metal_sites:
        pairs = [(s, z_new, z_idx)
                 for s in metal_sites
                 for z_new, z_idx in ELEM_EMB.items()]
        sites_t  = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        z_idx_t  = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=device)
        batched_g = dgl.batch([g_dev] * len(pairs))
        batched_coord = dgl.batch([g_coord] * len(pairs))
        with torch.no_grad():
            delta_s = cep_model(batched_g, batched_coord, sites_t, z_idx_t)  # [n_pairs]
        for i, (s, z_new, _) in enumerate(pairs):
            all_sites.append(s)
            all_z_new.append(z_new)
            all_scores.append(delta_s[i].item())

    # --- Suspicious non-metal sites: ElementPrior scores over METAL_CANDS ---
    suspicious_sites = [s for s in edit_sites if z[s].item() not in TRANSITION_METALS]
    if suspicious_sites:
        src_cpu, dst_cpu = g.edges()
        src_cpu, dst_cpu = src_cpu.cpu(), dst_cpu.cpu()
        for s in suspicious_sites:
            nbr_z = z[dst_cpu[src_cpu == s]]
            for z_metal in METAL_CANDS:
                y_idx = z_to_idx(z_metal)
                if y_idx < 0:
                    continue
                with torch.no_grad():
                    if masked_gnn is not None:
                        site_t = torch.tensor([s], dtype=torch.long, device=device)
                        y_t = torch.tensor([y_idx], dtype=torch.long, device=device)
                        log_p = masked_gnn.score(g.to(device), site_t, y_t).item()
                    else:
                        hist  = build_histogram(nbr_z).unsqueeze(0).to(device)
                        log_p = elem_prior.score(
                            hist, torch.tensor([y_idx], dtype=torch.long, device=device)
                        ).item()
                all_sites.append(s)
                all_z_new.append(z_metal)
                all_scores.append(log_p)   # log p(z_metal | env) — higher is better

    if not all_sites:
        return None, None

    scores_t = torch.tensor(all_scores, dtype=torch.float)
    weights  = torch.softmax(scores_t * beta_cep, dim=0)
    chosen   = torch.multinomial(weights, num_samples=1).item()
    return all_sites[chosen], all_z_new[chosen]

def compute_elem_energy(elem_prior, g, edit_sites: List[int],
                        z: torch.Tensor, device, masked_gnn=None) -> float:
    """E_elem = mean(-log p(z_i | env(i))) over all edit sites, clipped per site.

    Uses mean (not sum) so E_elem is independent of number of edit sites and
    stays in [0, MAX_PER_SITE], comparable in scale to E_topo ∈ [-1, 0].
    """
    if len(edit_sites) == 0:
        return 0.0
    MAX_PER_SITE = 5.0   # clip: -log(p) > 5 means p < 0.007, effectively zero

    src_cpu, dst_cpu = g.edges()
    src_cpu, dst_cpu = src_cpu.cpu(), dst_cpu.cpu()

    hists, y_idxs = [], []
    for site in edit_sites:
        nbr_z = z[dst_cpu[src_cpu == site]]
        hists.append(build_histogram(nbr_z))
        y_idxs.append(z_to_idx(z[site].item()))

    X = torch.stack(hists).to(device)
    Y = torch.tensor(y_idxs, dtype=torch.long, device=device)

    with torch.no_grad():
        if masked_gnn is not None:
            site_t = torch.tensor(edit_sites, dtype=torch.long, device=device)
            y_t = torch.tensor(y_idxs, dtype=torch.long, device=device)
            log_probs = masked_gnn.score(g.to(device),site_t, y_t)
        else:
            X = torch.stack(hists).to(device)
            Y = torch.tensor(y_idxs, dtype=torch.long, device=device)
            log_probs = elem_prior.score(X, Y)        # [n_sites], log p(z_i | env)

    per_site = (-log_probs).clamp(max=MAX_PER_SITE)
    return per_site.mean().item()                  # in [0, MAX_PER_SITE]

def compute_sa_energy_update(
    mol_data: Dict,
    proposed_Z: torch.Tensor,
    original_Z: torch.Tensor,
    topo_verifier: TopoVerifier,
    elem_prior: ElementPrior,
    multiplier: float,
    lambda_elem: float = 1.0,
    lambda_penalty: float = 0.05,
    
) -> Tuple[float, Dict]:
    """Combined energy E = E_topo + λ_elem * E_elem + λ_penalty * n_changes."""
    pos = mol_data['positions'] # torch.Tensor [N, 3]
    device = topo_verifier.device
        
    # 1 Reconstruct graph from proposed_Z + positions
    try: 
        _, g_geometry = reconstruct_molecular_graph(proposed_Z, pos, multiplier)
    except Exception:
        return 1e6, {'topology_score': 0.0, 'E_elem': 0.0, 'E_penalty': 0.0, 'n_changes': 0}
    
    g_geometry = ensure_node_features(g_geometry, 53)
    
    # 2. E_topo = -S_topo via TopoVerifier.score_batch
    s_topo = topo_verifier.score_batch([g_geometry])[0]
    E_topo = -s_topo
    
    # 3. Identify edit sites with graph so suspicious Cl/Br are included in E_elem
    edit_sites = identify_edit_sites(proposed_Z, g_geometry)
    
    # 4. E_elem from ElementPrior (bag-of-neighbors MLP)
    E_elem = compute_elem_energy(elem_prior, g_geometry, edit_sites, proposed_Z.cpu(), device, masked_gnn=None)
    
    # 5. Parsimony penalty: discourage unnecessary changes
    n_changes = int((proposed_Z != original_Z).sum().item())
    E_penalty = lambda_penalty * n_changes
    
    energy = E_topo + lambda_elem * E_elem + E_penalty
    metrics =  {
        'topology_score': s_topo,
        'E_topo': E_topo,
        'E_elem': E_elem,
        'E_penalty': E_penalty,
        'n_changes': n_changes,
    }
    return energy, metrics

def simulated_annealing_updated(
    mol_data: Dict,
    topo_verifier: TopoVerifier,
    cep_model,
    elem_prior: ElementPrior,
    device: torch.device,
    multiplier: float,
    t_init: float = 100.0,
    t_min: float = 1.0,
    max_steps: int = 1000,
    lambda_elem: float = 1.0,
    lambda_penalty: float = 0.05,
    beta_cep: float = 1.0,
    forced_edit_sites:Optional[list[int]] = None,
    verbose: bool = False,
    masked_gnn = None
) -> Tuple[Optional[torch.Tensor], float, Dict]:
    Z_current = mol_data['atomic_numbers'].clone()
    original_Z = Z_current.clone()
    connectivity = mol_data['connectivity']
    
    if filter_fragment(connectivity, max_fragments=3):
        if verbose:
            print("Too fragmented, skipping")
        return None, float('inf'), {}
    
    # Build graph first — needed to detect suspicious Cl/Br for gate + edit sites
    try:
        _, g_current = reconstruct_molecular_graph(Z_current, mol_data['positions'], multiplier)
        g_current = ensure_node_features(g_current, 53)
    except Exception:
        return None, float('inf'), {}

    edit_sites = forced_edit_sites if forced_edit_sites is not None \
                 else identify_edit_sites(Z_current, g_current)
    if len(edit_sites) == 0:
        if verbose:
            print("no edit sites found, skipping")
        return None, float('inf'), {}

    # Initial energy
    try:
        E_current, current_metrics = compute_sa_energy_update(
            mol_data, Z_current, original_Z, topo_verifier, elem_prior,
            multiplier, lambda_elem, lambda_penalty
        )
    except Exception as e:
        if verbose:
            print(f"failed to initialize: {e}")
        return None, float('inf'), {}

    best_Z = Z_current.clone()
    best_E = E_current
    best_metrices = current_metrics.copy()
    
    T = t_init
    history = {
        'energies': [E_current],
        'topology_scores': [current_metrics['topology_score']],
        'E_elem': [current_metrics['E_elem']],
        'temperatures': [T],
        'accepted': []
    }
    
    if verbose:
        metals = {z.item(): ELEMENT_NAMES.get(z.item(), f'Z={z.item()}')
                  for z in Z_current[edit_sites]}
        print(f" Init: metals={metals}, E={E_current:.4f},"
              f"S_topo={current_metrics['topology_score']:.4f},"
              f"E_elem={current_metrics['E_elem']:.4f}")

    # Build g_coord ONCE — positions don't change during SA
    _atoms_tmp = Atoms(numbers=Z_current.cpu().numpy(),
                       positions=mol_data['positions'].cpu().numpy())
    g_coord_fixed = build_coord_graph(_atoms_tmp).to(device)

    for step in range(max_steps):
        T = max(t_min, t_init / (step + 1))

        # CEP guided proposal: sample (site, z_new) from CEP score distribution
        site_idx, z_new = guided_proposal(g_current, edit_sites, Z_current, cep_model, elem_prior, device,
                                          g_coord=g_coord_fixed, beta_cep=beta_cep, masked_gnn=masked_gnn)
        if site_idx is None:
            continue
        
        proposed_Z = Z_current.clone()
        proposed_Z[site_idx] = z_new
        
        # Evaluate propsoed energy:
        E_proposed, proposed_metrices = compute_sa_energy_update(mol_data, proposed_Z, original_Z, topo_verifier, elem_prior,
                                                                 multiplier, lambda_elem, lambda_penalty)

        # Metropolis acceptance
        delta_E = E_proposed - E_current
        accept_prob = float(np.exp(-delta_E / T) if delta_E > 0 else 1.0)
        
        if random.random() < accept_prob:
            try: 
                _, g_proposed = reconstruct_molecular_graph(proposed_Z, mol_data['positions'], multiplier)
                g_proposed = ensure_node_features(g_proposed, 53)
            except Exception:
                pass
            else:
                Z_current = proposed_Z
                E_current = E_proposed
                current_metrics = proposed_metrices
                g_current = g_proposed
                history['accepted'].append(step)

        # Update edit sites after every step (Z_current may have changed on accept)
        if forced_edit_sites is None:
            edit_sites = identify_edit_sites(Z_current, g_current)
        
        if E_current < best_E:
            best_Z = Z_current.clone()
            best_E = E_current
            best_metrices = current_metrics.copy()
            
            if verbose:
                    metals = {z.item(): ELEMENT_NAMES.get(z.item(), f'Z={z.item()}')
                              for z in Z_current[identify_edit_sites(Z_current, g_current)]}
                    print(f"  Step {step:4d}: NEW BEST  metals={metals}, "
                          f"E={best_E:.4f}, S_topo={best_metrices['topology_score']:.4f}, "
                          f"E_elem={best_metrices['E_elem']:.4f}")
    
        history['energies'].append(E_current)
        history['topology_scores'].append(current_metrics['topology_score'])
        history['E_elem'].append(current_metrics['E_elem'])
        history['temperatures'].append(T)
        
        # Early stop: high topo score + low elem energy + few changes
        if (step >= 50 
            and best_metrices['E_elem'] < 0.5
            and best_metrices['n_changes'] <= len(edit_sites)):
            if verbose:
                print(f"early stop triggered at step {step}")
            break
        
        if T <= t_min:
            break
    
    return best_Z, best_E, history

def process_molecules(
        input_file: str,
        output_dir: str,
        t_init: float = 100.0,
        t_min: float = 1.0,
        max_steps: int = 1000,
        lambda_elem: float = 1.0,
        lambda_penalty: float = 0.05,
        beta_cep: float = 1.0,
        multiplier: float = 0.85,
        device_str: str = 'cuda',
        report_scores: bool = False,
        verbose: bool = False,
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    topo_verifier, cep_model, elem_prior, masked_GNN = load_all_models(device)
    
    data = torch.load(input_file)
    molecules = data['graphs']
    print(f'Found {len(molecules)} molecules')
    
    fixed_molecules = []
    skipped_count = 0
    unchanged_count = 0
    fixed_count = 0
    failed_count = 0
    accepted_orig_scores, accepted_fixed_scores = [], []
    center_metal_counts_before, center_metal_counts_after = Counter(), Counter()
    
    for idx, mol in enumerate(tqdm(molecules, desc="Processing")):
        # Gate: build initial graph to detect suspicious Cl/Br (Type B) + real metals
        try:
            _, g_init = reconstruct_molecular_graph(
                mol['atomic_numbers'], mol['positions'], multiplier
            )
            g_init = ensure_node_features(g_init, 53)
            edit_sites_orig = identify_edit_sites(mol['atomic_numbers'], g_init)
        except Exception as e:
            if verbose:
                print(f"  Molecule {idx}: graph build failed — {e}")
            failed_count += 1
            continue

        if len(edit_sites_orig) == 0:
            skipped_count += 1
            continue
        if verbose:
            print(f"Molecules{idx}: {mol.get('mol_id', idx)} "
                  f"({len(edit_sites_orig)} edit sites)")

        try:
            initial_E, initial_metrics = compute_sa_energy_update(
                mol, mol['atomic_numbers'], mol['atomic_numbers'],
                topo_verifier, elem_prior, multiplier, lambda_elem, lambda_penalty
            )
        except Exception as e:
            if verbose:
                print(f"initial scoring failed: {e}")
            failed_count += 1
            continue
        
        # Run SA
        best_Z, best_E, history = simulated_annealing_updated(
            mol_data=mol,
            topo_verifier=topo_verifier,
            cep_model=cep_model,
            elem_prior=elem_prior,
            device=device,
            multiplier=multiplier,
            t_init=t_init,
            t_min=t_min,
            max_steps=max_steps,
            lambda_elem=lambda_elem,
            lambda_penalty=lambda_penalty,
            beta_cep=beta_cep,
            verbose=verbose,
        )
        
        if best_Z is None:
            failed_count += 1
            continue
        
        n_changes = int((best_Z != mol['atomic_numbers']).sum().item())
        
        # Accept the proposal only if energy strictly improved and something changed
        if best_E < initial_E and n_changes > 0:
            try:
                _, g_fixed = reconstruct_molecular_graph(best_Z, mol['positions'], multiplier)
                g_fixed = ensure_node_features(g_fixed, 53)
                fixed_score = topo_verifier.score_batch([g_fixed])[0]
                orig_score = initial_metrics['topology_score']
                
                mol_fixed = dict(mol)
                mol_fixed['atomic_numbers'] = best_Z
                mol_fixed['sa_history'] = history
                mol_fixed['original_energy'] = initial_E
                mol_fixed['fixed_energy'] = best_E
                mol_fixed['original_score'] = orig_score
                mol_fixed['fixed_score'] = fixed_score
                mol_fixed['n_changes'] = n_changes
                
                for site in edit_sites_orig:
                    z_before = int(mol['atomic_numbers'][site].item())
                    z_after = int(best_Z[site].item())
                    center_metal_counts_before[element_label(z_before)] += 1
                    center_metal_counts_after[element_label(z_after)] += 1
                    
                fixed_molecules.append(mol_fixed)
                fixed_count += 1
                accepted_orig_scores.append(orig_score)
                accepted_fixed_scores.append(fixed_score)
                
                if verbose:
                    print(f"  Fixed ({n_changes} changes): "
                          f"E {initial_E:.4f} -> {best_E:.4f}, "
                          f"S_topo {orig_score:.4f} -> {fixed_score:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f"Post-SA eval failed:{e}") 
                failed_count += 1
                
        else:
            unchanged_count += 1
            if verbose:
                reason = "no energy improvement" if best_E >= initial_E else "no changes made"
                print(f"  Unchanged ({reason})")
                
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    composition_statistics = {
        'center_metal_counts_before': dict(center_metal_counts_before),
        'center_metal_counts_after':  dict(center_metal_counts_after),
        'n_total': len(molecules),
        'n_skipped_no_metal': skipped_count,
        'n_fixed': fixed_count,
        'n_unchanged': unchanged_count,
        'n_failed': failed_count,
    }
    output_data = {
        'graphs': fixed_molecules,
        'sa_config': {
            't_init': t_init, 't_min': t_min, 'max_steps': max_steps,
            'lambda_elem': lambda_elem, 'lambda_penalty': lambda_penalty,
            'beta_cep': beta_cep, 'multiplier': multiplier,
        },
        'statistics': composition_statistics,
    }

    torch.save(output_data, str(output_path / 'sa_result.pt'))
    with open(output_path / 'composition_stats.json', 'w') as f:
        json.dump(composition_statistics, f, indent=2)
    save_composition_figure(composition_statistics, output_path / 'composition_stats.png')
    if accepted_orig_scores:
        save_score_change_figure(accepted_orig_scores, accepted_fixed_scores,
                                 output_path / 'topology_score_change.png')
    if report_scores:
        print_score_distribution(accepted_orig_scores, title="Fixed: original S_topo")
        print_score_distribution(accepted_fixed_scores, title="Fixed: repaired S_topo")
        print_score_change(accepted_orig_scores, accepted_fixed_scores, title="S_topo change")

    processed = len(molecules) - skipped_count
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total molecules       : {len(molecules)}")
    print(f"Skipped (no metals)   : {skipped_count}")
    print(f"Processed             : {processed}")
    print(f"Fixed (energy↓)       : {fixed_count}  ({fixed_count/max(processed,1)*100:.1f}%)")
    print(f"Unchanged             : {unchanged_count}")
    print(f"Failed                : {failed_count}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CEP-guided SA for metallic element repair")
    parser.add_argument('--input',          type=str,   required=True)
    parser.add_argument('--output',         type=str,   required=True)
    parser.add_argument('--t_init',         type=float, default=100.0)
    parser.add_argument('--t_min',          type=float, default=1.0)
    parser.add_argument('--max_steps',      type=int,   default=1000)
    parser.add_argument('--lambda_elem',    type=float, default=0.1)
    parser.add_argument('--lambda_penalty', type=float, default=0.5)
    parser.add_argument('--beta_cep',       type=float, default=1.0,
                        help='CEP proposal sharpness (higher = greedier)')
    parser.add_argument('--multiplier',     type=float, default=0.85,
                        help='Bond length multiplier (calibrated s*=0.85)')
    parser.add_argument('--device',         type=str,   default='cuda')
    parser.add_argument('--report_scores',  action='store_true')
    parser.add_argument('--verbose',        action='store_true')
    args = parser.parse_args()

    process_molecules(
        input_file=args.input,
        output_dir=args.output,
        t_init=args.t_init,
        t_min=args.t_min,
        max_steps=args.max_steps,
        lambda_elem=args.lambda_elem,
        lambda_penalty=args.lambda_penalty,
        beta_cep=args.beta_cep,
        multiplier=args.multiplier,
        device_str=args.device,
        report_scores=args.report_scores,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()

                    