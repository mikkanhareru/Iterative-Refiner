import torch
import torch.nn as nn
import numpy as np
import random
import json
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import dgl
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList

from ase_process_topo import MolecularGraphBuilder, gaussian_rbf
from ase_model import TopoSSL
from ase_finetune import SSLClassifier

# Chemical Elements 
CENTER_ATOMS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30
                    ] #transition metals

LIGANDS = [1, 6, 7, 8, 15, 16, 9, 17, 35, 53]

# Elements name for reporting 
ELEMENT_NAMES = {1: 'H', 6:'C', 7:'N', 8:'O', 9:'F', 
15:'P', 16:'S', 17:'Cl', 27:'Co', 28:'Ni', 30:'Zn', 
35:'Br', 53:'I'}


def element_label(z: int) -> str:
    return ELEMENT_NAMES.get(int(z), f"Z={int(z)}")


def save_composition_figure(composition_statistics: Dict, output_path: Path) -> None:
    center_before = composition_statistics.get('center_transition_metal_counts_before', {})
    center_after = composition_statistics.get('center_transition_metal_counts_after', {})
    ligand_before = composition_statistics.get('ligand_type_counts_before', {})
    ligand_after = composition_statistics.get('ligand_type_counts_after', {})

    def aligned_items(before: Dict, after: Dict):
        keys = sorted(set(before.keys()) | set(after.keys()))
        b = [int(before.get(k, 0)) for k in keys]
        a = [int(after.get(k, 0)) for k in keys]
        return keys, b, a

    def top_k_items(before: Dict, after: Dict, k: int = 12):
        total = Counter(before) + Counter(after)
        keys = [name for name, _ in total.most_common(k)]
        b = [int(before.get(k1, 0)) for k1 in keys]
        a = [int(after.get(k1, 0)) for k1 in keys]
        return keys, b, a

    center_names, center_b, center_a = aligned_items(center_before, center_after)
    ligand_names, ligand_b, ligand_a = top_k_items(ligand_before, ligand_after, k=12)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Center transition-metal counts
    if len(center_names) > 0:
        x = np.arange(len(center_names))
        w = 0.4
        axes[0].bar(x - w/2, center_b, width=w, label='Before SA')
        axes[0].bar(x + w/2, center_a, width=w, label='After SA')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(center_names, rotation=30, ha='right')
    axes[0].set_title('Center Transition-Metal Counts')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.25)

    # Ligand type counts (top-k)
    if len(ligand_names) > 0:
        x2 = np.arange(len(ligand_names))
        w2 = 0.4
        axes[1].bar(x2 - w2/2, ligand_b, width=w2, label='Before SA')
        axes[1].bar(x2 + w2/2, ligand_a, width=w2, label='After SA')
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(ligand_names, rotation=30, ha='right')
    axes[1].set_title('Ligand Type Counts (Top 12 by Total Frequency)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.25)

    n_attempted = composition_statistics.get('n_sa_attempted', 0)
    n_succeeded = composition_statistics.get('n_sa_succeeded', 0)
    fig.suptitle(f"SA Composition Summary (attempted={n_attempted}, succeeded={n_succeeded})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_score_change_figure(before_scores: List[float], after_scores: List[float], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(before_scores) == 0 or len(after_scores) == 0 or len(before_scores) != len(after_scores):
        axes[0].text(0.5, 0.5, "No valid paired score data", ha='center', va='center')
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        fig.suptitle("Topology Score Change Distribution")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    before = np.array(before_scores, dtype=float)
    after = np.array(after_scores, dtype=float)
    delta = after - before

    # Left panel: before vs after distributions
    bins = np.linspace(0.0, 1.0, 31)
    axes[0].hist(before, bins=bins, alpha=0.6, label='Before SA')
    axes[0].hist(after, bins=bins, alpha=0.6, label='After SA')
    axes[0].set_title('Topology Score Distribution')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.25)

    # Right panel: score delta distribution
    delta_min = float(delta.min())
    delta_max = float(delta.max())
    if np.isclose(delta_min, delta_max):
        delta_bins = np.linspace(delta_min - 1e-3, delta_max + 1e-3, 21)
    else:
        delta_bins = np.linspace(delta_min, delta_max, 31)
    axes[1].hist(delta, bins=delta_bins, color='tab:green', alpha=0.8)
    axes[1].axvline(0.0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Score Delta (After - Before)')
    axes[1].set_xlabel('Delta')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.25)

    improved = int((delta > 0).sum())
    unchanged = int((delta == 0).sum())
    worsened = int((delta < 0).sum())
    fig.suptitle(
        f"Topology Score Change Distribution (n={len(delta)}, improved={improved}, unchanged={unchanged}, worsened={worsened})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def ensure_node_features(g, num_features=53):
    if 'h' in g.ndata:
        return g
    num_nodes = g.num_nodes()
    h = torch.zeros(num_nodes, num_features)
    
    # Feature 0: Atomic number (normalized)
    if 'atomic_num' in g.ndata:
        h[:, 0] = g.ndata['atomic_num'].float() / 100.0
    
    # Feature 1: Degree
    if 'degree' in g.ndata:
        h[:, 1] = g.ndata['degree'].float()
    
    # Feature 2: In ring
    if 'in_ring' in g.ndata:
        h[:, 2] = g.ndata['in_ring'].float()
    
    g.ndata['h'] = h
    return g
    
    

def find_spatial_neighbors(center_idx:int, positions:torch.Tensor, distance_threshold:float=3.0, exclude_self:bool=True):
    center_pos = positions[center_idx]

    # calcualte distance to all atoms
    distance = torch.norm(positions - center_pos, dim=1)

    # Find the atoms within the threshold
    mask = distance < distance_threshold

    if exclude_self:
        mask[center_idx] = False

    neighbor_idxs = torch.where(mask)[0].tolist()

    return neighbor_idxs

def identify_center_atom(
        atomic_numbers: torch.Tensor,
        connectivity: torch.Tensor,
        positions: torch.Tensor,
        coord_distance_threshold: float = 3.0
) -> Tuple[Optional[int], List[int]]:
    """Same as batch processing script build for SA
        Returns: 
                (center_idx, neighbor_idxs)
    """
    n_atoms = len(atomic_numbers)

    #transition metals
    transition_metals = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))

    # Strategy 1: find atom with maximum degree
    degrees = connectivity.sum(dim=1)
    max_degree = degrees.max().item()

    if max_degree >= 3:
        candidates = torch.where(degrees == max_degree)[0].tolist()

        # detect if the candidate is in metal
        metal_candiates = [i for i in candidates if atomic_numbers[i].item() in transition_metals]
        if metal_candiates:
            center_idx = metal_candiates[0]
            neighbor_idxs = torch.where(connectivity[center_idx] > 0)[0].tolist()
            return center_idx, neighbor_idxs
        
        center_idx = candidates[0]
        neighbor_idxs = torch.where(connectivity[center_idx] > 0)[0].tolist()
        return center_idx, neighbor_idxs
    
    # strategy 2: Find isolated transition metals 
    for i in range(n_atoms):
        if atomic_numbers[i].item() in transition_metals:
            degree_i = degrees[i].item()

            # find isolated atom
            if degree_i <= 1:
                print(f"found isolated transition metal at {i} (Z={atomic_numbers[i].item()}, degree={degree_i})")

                neighbors = find_spatial_neighbors(
                    center_idx=i,
                    positions=positions,
                    distance_threshold=coord_distance_threshold,
                    exclude_self=True 
                )

                if len(neighbors) >= 2:
                    return i, neighbors

    # strategy 3: Spatial proximity for any transition metal
    for i in range(n_atoms):
        if atomic_numbers[i].item() in transition_metals:
            neighbors = find_spatial_neighbors(
                center_idx=i,
                positions=positions,
                distance_threshold=coord_distance_threshold,
                exclude_self=True
            )
        if len(neighbors) > 0: # type: ignore
            return i, neighbors # type: ignore
        
    # strategy 4: find the highest atomic num 
    center_idx = torch.argmax(atomic_numbers).item()
    neighbors = find_spatial_neighbors(
        center_idx=center_idx, # pyright: ignore[reportArgumentType]
        positions=positions,
        distance_threshold=coord_distance_threshold,
        exclude_self=True
    )
    if len(neighbors) > 0:
        return center_idx, neighbors # type: ignore

    return None, []

def compute_element_penalty(
        atomic_numbers: torch.Tensor,
        center_idx: Optional[int],
        neighbor_idxs: List[int]
):
    """Compute penalty for chemically implausible element assignemnts
        Atomic_numbers: [N] atomic numbers
        center_idx: index of center atom
        neighbor_idx: indices of neighbor atoms 
        Returns: 
            penalty_value (0 = good chemistry, higher = worse)
    """
    if center_idx is None or len(neighbor_idxs) == 0:
        return 0.0

    penalty = 0.0 

    # transition metals should be at the center 
    transition_metals = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))\
    
    # Halogens should not be at the center
    halogens = {9, 17, 35, 53}

    # Typical ligands
    ligands_atoms = {1, 6, 7, 8, 15, 16, 9, 17, 35, 53}

    # check center atom 
    center_Z = atomic_numbers[center_idx].item()

    # Penalty 1: center should be transition metal
    if center_Z not in transition_metals:
        penalty += 1.0


    # Penalty 2: Halogen can not be at center
    if center_Z in halogens:
        penalty += 2.0

    # penalty 3: common misclassify carbon as center
    if center_Z == 6:
        penalty += 0.5

    # penalty 4: check neighbor chemistry 
    for neighbor_idx in neighbor_idxs:
        neighbor_Z = atomic_numbers[neighbor_idx].item()

        #unusal ligands
        if neighbor_Z not in ligands_atoms and neighbor_Z not in transition_metals:
            penalty += 1
    
    # penalty 5: penalizes poor electron donors, rewards good electron donors
    L_good = {7, 8, 16, 15, 17, 35, 53}
    L_poor = {1, 6, 9}
    
    if center_Z in transition_metals:
        for neighbor_idx in neighbor_idxs:
            neighbor_Z = atomic_numbers[neighbor_idx].item()
            if neighbor_Z in L_poor:
                penalty += 0.5 
    
    # penalty 6: penalize mixed donor types 
    if center_Z in transition_metals and len(neighbor_idxs) >= 3:
        neighbor_elements = [atomic_numbers[n].item() for n in neighbor_idxs]
        unique_elements = set(neighbor_elements)
        if len(unique_elements) > 1:
            # small penalty proportiona lto how mixed it is 
            penalty += (len(unique_elements) - 1) * 0.3

    return penalty

def filter_fragment(connectivity: torch.Tensor, max_fragments:int=3) -> bool:
    n_atoms = connectivity.shape[0]
    visited = set()
    n_fragments = 0
    
    for start_atom in range(n_atoms):
        if start_atom in visited:
            continue
        
        n_fragments += 1
        queue = [start_atom]
        visited.add(start_atom)
        
        while queue:
            current = queue.pop(0)
            neighbors = torch.where(connectivity[current]>0)[0]
            
            for neighbor in neighbors:
                neighbor = int(neighbor.item())
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return n_fragments > max_fragments

def propose_chemcially_valid_move(
    current_Z: torch.Tensor,
    center_idx: int,
    neighbor_idxs: List[int],
    p_center: float = 0.5
) -> torch.Tensor:
    proposed_Z = current_Z.clone()
    
    if random.random() < p_center and len(CENTER_ATOMS) > 0:
        new_Z = random.choice(CENTER_ATOMS)
        proposed_Z[center_idx] = new_Z
    elif len(neighbor_idxs) > 0 and len(LIGANDS) > 0:
        neighbor = random.choice(neighbor_idxs)
        new_Z = random.choice(LIGANDS)
        proposed_Z[neighbor] = new_Z
        
    return proposed_Z

def reconstruct_molecular_graph(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    multiplier: float = 1.0,
    num_features: int = 53
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    
    #get atoms
    atoms = Atoms(
        numbers=atomic_numbers.cpu().numpy(),
        positions=positions.cpu().numpy()
    )
    
    # build graph data
    builder = MolecularGraphBuilder(atoms, multiplier=multiplier)
    builder.build_graph(
        apply_correction=False,
        use_aromatic_check=True,
        detect_rings=True,
        verbose=False
    )
    
    # Get graph data
    graph = builder.get_graph()
    n_atoms = len(atoms)
    
    # outline graph
    src = [i for i, j in builder.bonds]
    dst = [j for i, j in builder.bonds]
    
    g_outline = dgl.graph((src + dst, dst + src), num_nodes=n_atoms)
    
    # craete 'h' feature for outline
    h_outline = torch.zeros(n_atoms, num_features)
    h_outline[:,0] = atomic_numbers.float() / 100.0
    if 'degrees' in graph:
        h_outline[:,1] = torch.tensor(graph['degrees'], dtype=torch.float)
    if 'in_ring' in graph:
        h_outline[:,2] = torch.tensor(graph["in_ring"], dtype=torch.float)
    
    g_outline.ndata['h'] = h_outline
    g_outline.ndata['atomic_num'] = atomic_numbers
    
    # create geometry graph
    g_geometry = dgl.graph((src + dst, dst + src), num_nodes=n_atoms)
    
    # Create 'h' feature for geometry
    h_geometry = h_outline.clone()
    if 'rbf_features' in graph and len(graph['rbf_features']) > 0:
        rbf = graph['rbf_features']
        rbf_tensor = torch.tensor(rbf, dtype=torch.float)
        g_geometry.edata['rbf'] = torch.cat([rbf_tensor, rbf_tensor], dim=0)
        
        # Average RBF features for each atom node
        for i in range(n_atoms):
            neighbors = builder.connectivity.get(i, [])
            if neighbors:
                neighbor_rbfs = []
                for j in neighbors:
                    bond = tuple(sorted([i, j]))
                    if bond in builder.bond_lengths:
                        idx = builder.bonds.index(bond)
                        neighbor_rbfs.append(rbf_tensor[idx])
                if neighbor_rbfs:
                    avg_rbf = torch.stack(neighbor_rbfs).mean(dim=0)
                    h_geometry[i, 3:53] = avg_rbf
        
    g_geometry.ndata['h'] = h_geometry
    g_geometry.ndata['atomic_num'] = atomic_numbers
    if 'degrees' in graph:
        g_geometry.ndata['degree'] = torch.tensor(graph['degrees'], dtype=torch.long)
    if 'in_ring' in graph:
        g_geometry.ndata['in_ring'] = torch.tensor(graph['in_ring'], dtype=torch.float)
        
    return g_outline, g_geometry

def compute_topology_score(
    model: nn.Module,
    g_outline: dgl.DGLGraph,
    g_geometry: dgl.DGLGraph,
    device: torch.device
) -> float:
    model.eval()
    
    with torch.no_grad():
        g_outline = g_outline.to(device)
        g_geometry = g_geometry.to(device)
        
        logits = model(g_geometry)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            score = torch.sigmoid(logits.view(-1))[0].item()
        else:
            probs = torch.softmax(logits, dim=1)
            score = probs[:, 1][0].item()
    return score


def compute_topology_probs(
    model: nn.Module,
    g_geometry: dgl.DGLGraph,
    device: torch.device
) -> List[float]:
    model.eval()
    with torch.no_grad():
        g_geometry = g_geometry.to(device)
        logits = model(g_geometry)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            p1 = torch.sigmoid(logits.view(-1))[0].item()
            return [1.0 - p1, p1]
        probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu().tolist()


def print_score_distribution(scores: List[float], title: str) -> None:
    if len(scores) == 0:
        print(f"{title}: no data")
        return

    arr = np.array(scores, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f"{title}:")
    print(f"  n={len(arr)} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")
    print(f"  p10={p10:.4f} p25={p25:.4f} p50={p50:.4f} p75={p75:.4f} p90={p90:.4f}")


def print_score_change(before_scores: List[float], after_scores: List[float], title: str = "Score change") -> None:
    if len(before_scores) == 0 or len(after_scores) == 0:
        print(f"{title}: no paired data")
        return
    if len(before_scores) != len(after_scores):
        print(f"{title}: invalid paired data (before={len(before_scores)}, after={len(after_scores)})")
        return

    delta = np.array(after_scores, dtype=float) - np.array(before_scores, dtype=float)
    print(f"{title}:")
    print(f"  mean_delta={delta.mean():.4f} median_delta={np.median(delta):.4f}")
    print(f"  improved={(delta > 0).sum()}/{len(delta)} unchanged={(delta == 0).sum()} worsened={(delta < 0).sum()}")
    print(f"  min_delta={delta.min():.4f} max_delta={delta.max():.4f}")


def compute_sa_energy(
    mol_data: Dict,
    proposed_Z: torch.Tensor,
    original_Z: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    lambda_penalty: float = 0.05,
    lambda_element: float = 1.0,
    multiplier: float = 1.0
) -> Tuple[float, Dict]:
    """Energy function with element penalty: 
    E = -topology score + λ*n_change + λ*element_penalty
    element penalty obtained from compute_element_penalty()
    Returns:
        energy: total energy
        metrics: dictionary with individual components 
    """
    g_outline, g_geometry = reconstruct_molecular_graph(atomic_numbers=proposed_Z,
                                                        positions=mol_data['positions'],
                                                        multiplier=multiplier)
    topology_score = compute_topology_score(model, g_outline, g_geometry, device)
    
    # change penalty, we want the number of changed atoms to be as less as possible
    n_changes = (proposed_Z != original_Z).sum().item()
    change_penalty = lambda_penalty * n_changes
    
    element_penalty = compute_element_penalty(
        atomic_numbers=proposed_Z,
        center_idx=mol_data['center_atom_idx'],
        neighbor_idxs=mol_data['neighbor_idxs']
    )
    
    # energy function
    energy = -topology_score + change_penalty + element_penalty*lambda_element
    
    metrics = {
        'topology_score': topology_score,
        'change_penalty': change_penalty,
        'element_penalty': element_penalty,
        "n_changes": n_changes
    }
    
    return energy, metrics

def simulated_annealing(
        mol_data: Dict,
        model: nn.Module,
        device: torch.device,
        t_init: float = 100.0,
        t_min: float = 1.0,
        max_steps: int = 1000,
        lambda_penalty: float = 0.05,
        lambda_element: float = 1.0,
        p_center: float = 0.3,
        multiplier: float = 1.0,
        verbose: bool = False
) -> Tuple[Optional[torch.Tensor], float, Dict]:
    """
    Main SA loop, penalizes wrong eements when the topology is correct
    Returns:
        (best_Z, best_energy, history) tuple
    """
    positions = mol_data['positions']
    Z_current = mol_data['atomic_numbers'].clone()
    connectivity = mol_data['connectivity']
    center_idx = mol_data.get('center_atom_idx', None)
    neighbor_idxs = mol_data.get('neighbor_idxs', [])

    if filter_fragment(connectivity, max_fragments=3):
        if verbose:
            print(f"too fragmented, filtered")
        return None, float('inf'), {}
    
    if center_idx is None or len(neighbor_idxs) == 0:
        if verbose:
            print(f"No center atom found or neighbor found, filtered")
        return None, float('inf'), {}
    
    try: 
        E_current, current_metrics = compute_sa_energy(mol_data, Z_current, Z_current, model, device,
                                                       lambda_penalty, lambda_element, multiplier)
    except Exception as e:
        if verbose:
            print(f"failed to initailize: {e}")
        return None, float('inf'), {}
    
    best_Z = Z_current.clone()
    best_E = E_current
    best_metrics = current_metrics.copy()

    # SA loop:
    T = t_init
    history = {
        'energies': [E_current],
        'topology_scores': [current_metrics['topology_score']],
        'element_penalties': [current_metrics['element_penalty']],
        'temperatures': [T],
        'accepted': []
    }

    if verbose: 
        center_elem = ELEMENT_NAMES.get(Z_current[center_idx].item(), f"Z={Z_current[center_idx].item()}")
        print(f"  Initial: Center={center_elem}, E={E_current:.4f}, "
              f"Topo={current_metrics['topology_score']:.4f}, "
              f"Elem penalty={current_metrics['element_penalty']:.2f}")
    
    for step in range(max_steps):
        T = max(t_min, t_init / (step + 1))

        # skip reidentfy center, use original center throughout 
        # come back if needed, check result 
        center_idx_current = center_idx
        neighbor_idxs_currect= neighbor_idxs

        # propose new elements 
        proposed_Z  = propose_chemcially_valid_move(Z_current, center_idx_current, neighbor_idxs_currect, p_center)

        # reconstruct center (around) strucuture, recompute E 
        try:
                E_proposed, proposed_metrics = compute_sa_energy(
                mol_data, proposed_Z, Z_current, model, device, lambda_penalty,
                lambda_element, multiplier
            )
        except Exception as e:
            # invalid strucutre, reject 
            continue
            
        # Metropolis acceptence criterion
        delta_E = E_proposed - E_current
        accept_prob = np.exp(-delta_E / T) if delta_E > 0 else 1.0

        if random.random() < accept_prob:
            # Accept 
            Z_current = proposed_Z
            E_current = E_proposed
            current_metrics = proposed_metrics
            history['accepted'].append(step)

            # Track best solution
            if E_current < best_E:
                best_Z = Z_current.clone()
                best_E = E_current
                best_metrics = current_metrics.copy()

                if verbose and step % 100 == 0:
                    center_elem = ELEMENT_NAMES.get(best_Z[center_idx].item(), f"Z={best_Z[center_idx].item()}") # type: ignore
                    print(f"  Step {step}: NEW BEST! Center={center_elem}, E={best_E:.4f}, "
                          f"Topo={best_metrics['topology_score']:.4f}, "
                          f"Elem={best_metrics['element_penalty']:.2f}")
                    
        history['energies'].append(E_current)
        history['topology_scores'].append(current_metrics['topology_score'])
        history['element_penalties'].append(current_metrics['element_penalty'])
        history['temperatures'].append(T)    

        # early stop
        min_explore_steps = max(50, len(neighbor_idxs) * 10) # at leat 10 steps per ligands
        if step >= min_explore_steps and best_metrics['element_penalty'] == 0.0 and best_metrics['topology_score'] > 0.93:
            if verbose:
                print(f"early stop at setp {step}: chemical plausible strucutre found")
            break
        
        # stop if exceed min temperature
        if T <= t_min:
            break

    if verbose:
        orig_elem = ELEMENT_NAMES.get(
            mol_data['atomic_numbers'][center_idx].item(),
            f"Z={mol_data['atomic_numbers'][center_idx].item()}"
        )
        final_elem = ELEMENT_NAMES.get(
            best_Z[center_idx].item(),
            f"Z={best_Z[center_idx].item()}"
        ) # type: ignore
        ligand_changes = []
        for n_idx in neighbor_idxs:
            before_z = int(mol_data['atomic_numbers'][n_idx].item())
            after_z = int(best_Z[n_idx].item())
            if before_z != after_z:
                before_name = ELEMENT_NAMES.get(before_z, f"Z={before_z}")
                after_name = ELEMENT_NAMES.get(after_z, f"Z={after_z}")
                ligand_changes.append(f"{n_idx}:{before_name}->{after_name}")

        print(f"  Final: {orig_elem} → {final_elem}, E={best_E:.4f}, "
              f"Topo={best_metrics['topology_score']:.4f}, "
              f"Elem penalty={best_metrics['element_penalty']:.2f}")
        if ligand_changes:
            print(f"  Ligand changes: {', '.join(ligand_changes)}")
        else:
            print("  Ligand changes: none")
        print(f"  Accepted {len(history['accepted'])}/{step+1} proposals")

    return best_Z, best_E, history

def process_suspicious_molecules(
        input_file: str,
        output_file: str,
        model_path: str,
        ssl_checkpoint: str,
        threshold: float=0.7,
        threshold_mode: str='below',
        t_init: float=100.0,
        t_min: float=0.1,
        max_steps: int=1000,
        lambda_penalty: float=1.0,
        lambda_element: float=1.0,
        device: str = 'cuda',
        report_scores: bool = False,
        verbose: bool = True
): 
    """Process molecules base on SA score"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu') # type: ignore
    print(f"using device: {device}")

    # fintune checkpoint
    print(f"loading fine-tuned model path {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    hidden_dim = 512
    num_layers = 5
    proj_dim = 256
    print(f'hidden dim = {hidden_dim}')
    print(f"num layers = {num_layers}")
    print(f"proj_dim = {proj_dim}")

    print(f"initializing model strucutre from SSL cehckpoint: {ssl_checkpoint}")
    model = SSLClassifier(
        ssl_checkpoint_path=ssl_checkpoint,
        num_classes=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        proj_dim=proj_dim,
        freeze_encoder=False,
        dropout=0.3
    )

    if 'encoder_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
        model.ssl_model.load_state_dict(checkpoint['encoder_state_dict'])
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("Checkpoint missing required weights: expected encoder/classifier_state_dict or model_state_dict")
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    model = model.to(device)
    model.eval()

    print(f"loading data from {input_file}")
    data = torch.load(input_file)
    molecules = data['graphs']

    print(f"found {len(molecules)} molecules")
    print("topology score class index = 1 (fixed)")

    print(f"\nIdentifying molecules for SA (mode={threshold_mode}, threshold={threshold})...")
    suspicious_indices = []
    all_scores = []
    all_scores_cls0 = []
    all_scores_cls1 = []

    for idx, mol in enumerate(tqdm(molecules, desc="Scoring")):
        try: 
            g_geometry = ensure_node_features(mol['g_homo_geometry'])
            g_outline = ensure_node_features(mol['g_homo_outline'])

            probs = compute_topology_probs(model, g_geometry, device)
            if len(probs) >= 2:
                all_scores_cls0.append(probs[0])
                all_scores_cls1.append(probs[1])
            score = probs[1] if len(probs) > 1 else probs[-1]
            all_scores.append(score)

            if threshold == 0.0:
                should_process = True
            elif threshold_mode == 'below':
                should_process = score < threshold
            else:
                should_process = score > threshold

            if should_process:
                suspicious_indices.append((idx, score))
        except Exception as e:
            if verbose:
                print(f"failed to score molecule {idx}: {e}")

    if report_scores:
        print_score_distribution(all_scores, title="All molecule topology score distribution")
        if len(all_scores_cls0) > 0 and len(all_scores_cls1) > 0:
            print_score_distribution(all_scores_cls0, title="All molecule class-0 probability distribution")
            print_score_distribution(all_scores_cls1, title="All molecule class-1 probability distribution")

    print(f"Processing {len(suspicious_indices)} molecules with SA")
    if threshold == 0.0:
        print("  (threshold=0.0: applying SA to ALL molecules)")
    
    if len(suspicious_indices) == 0:
        print("No molecules to process!")
        return
    
    # Second pass: apply enhanced SA
    print(f"\nApplying enhanced SA with element penalties...")
    print(f"  lambda_element = {lambda_element} (element penalty weight)")
    
    fixed_molecules = []
    fixed_count = 0
    failed_count = 0
    high_score_fixes = 0  # Track fixes for high-scoring molecules
    accepted_orig_scores = []
    accepted_fixed_scores = []
    transition_metal_set = set(CENTER_ATOMS)
    center_metal_counts_before = Counter()
    center_metal_counts_after = Counter()
    ligand_type_counts_before = Counter()
    ligand_type_counts_after = Counter()
    sa_attempted = 0
    sa_succeeded = 0
    
    for idx, orig_score in tqdm(suspicious_indices, desc="Fixing"):
        mol = molecules[idx]
        sa_attempted += 1
        
        if verbose:
            print(f"\nMolecule {idx}: {mol['mol_id']} (score={orig_score:.4f})")
        
        # Apply enhanced SA
        fixed_Z, fixed_energy, history = simulated_annealing(
            mol_data=mol,
            model=model,
            device=device, # type: ignore
            t_init=t_init,
            t_min=t_min,
            max_steps=max_steps,
            lambda_penalty=lambda_penalty,
            lambda_element=lambda_element,
            verbose=verbose
        )
        
        if fixed_Z is not None:
            # Get final topology score
            try:
                sa_succeeded += 1
                g_outline_fixed, g_geometry_fixed = reconstruct_molecular_graph(
                    fixed_Z, mol['positions'], multiplier=1.0
                )
                fixed_score = compute_topology_score(
                    model, g_outline_fixed, g_geometry_fixed, device # type: ignore
                )

                center_idx = mol.get('center_atom_idx')
                neighbor_idxs = mol.get('neighbor_idxs', [])
                if center_idx is not None:
                    z_before = int(mol['atomic_numbers'][center_idx].item())
                    z_after = int(fixed_Z[center_idx].item())
                    if z_before in transition_metal_set:
                        center_metal_counts_before[element_label(z_before)] += 1
                    if z_after in transition_metal_set:
                        center_metal_counts_after[element_label(z_after)] += 1
                    for n_idx in neighbor_idxs:
                        lig_before = int(mol['atomic_numbers'][n_idx].item())
                        lig_after = int(fixed_Z[n_idx].item())
                        ligand_type_counts_before[element_label(lig_before)] += 1
                        ligand_type_counts_after[element_label(lig_after)] += 1
                
                # Calculate improvements
                element_improved = history['element_penalties'][0] > history['element_penalties'][-1]
                score_improved = fixed_score > orig_score
                
                # Accept if either improved
                if element_improved or score_improved:
                    mol_fixed = mol.copy()
                    mol_fixed['atomic_numbers'] = fixed_Z
                    mol_fixed['g_homo_outline'] = g_outline_fixed
                    mol_fixed['g_homo_geometry'] = g_geometry_fixed
                    mol_fixed['sa_history'] = history
                    mol_fixed['original_score'] = orig_score
                    mol_fixed['fixed_score'] = fixed_score
                    mol_fixed['original_energy'] = history['energies'][0]
                    mol_fixed['fixed_energy'] = fixed_energy
                    
                    fixed_molecules.append(mol_fixed)
                    fixed_count += 1
                    accepted_orig_scores.append(orig_score)
                    accepted_fixed_scores.append(fixed_score)
                    
                    # Track high-score fixes
                    if orig_score > 0.93:
                        high_score_fixes += 1
                    
                    if verbose:
                        print(f"  ✓ Fixed: Score {orig_score:.4f}→{fixed_score:.4f}, "
                              f"Energy {history['energies'][0]:.4f}→{fixed_energy:.4f}")
                else:
                    if verbose:
                        print(f"  ✗ No improvement")
                    failed_count += 1
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to reconstruct: {e}")
                failed_count += 1
        else:
            if verbose:
                print(f"  ✗ SA failed")
            failed_count += 1
    
    # Save results
    print(f"\nSaving {len(fixed_molecules)} fixed molecules to {output_file}")
    output_data = {
        'graphs': fixed_molecules,
        'sa_config': {
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'T_init': t_init,
            'T_min': t_min,
            'max_steps': max_steps,
            'lambda_penalty': lambda_penalty,
            'lambda_element': lambda_element,
        },
        'statistics': {
            'n_suspicious': len(suspicious_indices),
            'n_fixed': fixed_count,
            'n_failed': failed_count,
            'n_high_score_fixes': high_score_fixes,
        }
    }

    composition_statistics = {
        'n_sa_attempted': sa_attempted,
        'n_sa_succeeded': sa_succeeded,
        'center_transition_metal_counts_before': dict(center_metal_counts_before),
        'center_transition_metal_counts_after': dict(center_metal_counts_after),
        'ligand_type_counts_before': dict(ligand_type_counts_before),
        'ligand_type_counts_after': dict(ligand_type_counts_after),
    }
    output_data['composition_statistics'] = composition_statistics

    output_path = Path(output_file)
    # Always treat output path as a directory and store all artifacts inside it.
    output_path.mkdir(parents=True, exist_ok=True)
    main_save_path = output_path / 'sa_result.pt'
    composition_json_path = output_path / 'composition_stats.json'
    composition_fig_path = output_path / 'composition_stats.png'

    torch.save(output_data, str(main_save_path))
    with open(composition_json_path, 'w', encoding='utf-8') as f:
        json.dump(composition_statistics, f, indent=2)
    save_composition_figure(composition_statistics, composition_fig_path)
    score_change_fig_path = output_path / 'topology_score_change_distribution.png'
    save_score_change_figure(accepted_orig_scores, accepted_fixed_scores, score_change_fig_path)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total molecules processed: {len(suspicious_indices)}")
    print(f"Successfully fixed: {fixed_count}")
    print(f"Failed to fix: {failed_count}")
    print(f"Success rate: {fixed_count / len(suspicious_indices) * 100:.1f}%")
    print(f"\nHigh-score fixes (>0.93): {high_score_fixes}")
    print(f"  → These had correct topology but wrong element!")
    print(f"Composition stats saved: {composition_json_path}")
    print(f"Composition figure saved: {composition_fig_path}")
    print(f"Score-change figure saved: {score_change_fig_path}")
    if report_scores:
        print_score_distribution(accepted_orig_scores, title="Accepted molecules original score distribution")
        print_score_distribution(accepted_fixed_scores, title="Accepted molecules fixed score distribution")
        print_score_change(accepted_orig_scores, accepted_fixed_scores, title="Accepted molecules score change")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SA with element-aware penalties"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input .pt file with molecular data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .pt file for fixed molecules')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--ssl_checkpoint', type=str, required=True,
                        help='Path to SSL pretrained checkpoint (needed to initialize model)')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Score threshold (0.0 = process all molecules)')
    parser.add_argument('--threshold_mode', type=str, default='below', choices=['below', 'above'],
                        help='Select molecules with score below/above threshold')
    parser.add_argument('--t_init', type=float, default=10.0,
                        help='Initial temperature for SA')
    parser.add_argument('--t_min', type=float, default=0.1,
                        help='Minimum temperature for SA')
    parser.add_argument('--max_steps', type=int, default=1500,
                        help='Maximum SA steps per molecule')
    parser.add_argument('--lambda_penalty', type=float, default=0.05,
                        help='Penalty for atom changes')
    parser.add_argument('--lambda_element', type=float, default=1.0,
                        help='Penalty weight for wrong elements')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')
    parser.add_argument('--report_scores', action='store_true',
                        help='Print topology score distributions and before/after score changes')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    process_suspicious_molecules(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model,
        ssl_checkpoint=args.ssl_checkpoint,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        t_init=args.t_init,
        t_min=args.t_min,
        max_steps=args.max_steps,
        lambda_penalty=args.lambda_penalty,
        lambda_element=args.lambda_element,
        device=args.device,
        report_scores=args.report_scores,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()


        
