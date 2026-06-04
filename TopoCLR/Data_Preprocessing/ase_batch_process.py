import torch
import dgl
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from ase.io import read
from typing import List, Dict, Optional
import sys
import warnings
from collections import deque
warnings.filterwarnings('ignore')

from ase_process_topo import MolecularGraphBuilder as StructuralBuilder

def extract_single_molecule(atoms, builder):
    """Use in the case where multiple molecules or segments are in one file

    Args:
        atoms (_type_): ASE atoms object
        builder (_type_): StructuralBuilder with connectivity 
    Returns:
        Ase atoms containing oly the largest connected cmponent
    """
    if len(atoms) == 0:
        return None

    adjacency = builder.connectivity

    # find all connected components using BFS
    visited = set()
    components = []

    for start_node in range(len(atoms)):
        if start_node in visited:
            continue
    
        # BFS to find connected component 
        component = []
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)
    
    if not components:
        return None
    
    # largest component
    largest_component = max(components, key=len)

    # sort to maintain order consistency
    largest_component.sort()

    # createnew atom object with only the main molecule
    main_molecule = atoms[largest_component]

    return main_molecule

def create_homogeneous_outline_graph(atoms, all_bonds: List) -> dgl.DGLGraph:
    """Create homo DGL graph withall bonds

    Args:
        atoms (_type_): Atoms object
        all_bonds (List): List of (i, j) tuples for all bonds

    Returns:
        dgl.DGLGraph: Homogeneous graph
    """
    n_atoms = len(atoms)

    src = [i for i, j in all_bonds] + [j for i, j in all_bonds]
    dst = [j for i, j in all_bonds] + [i for i, j in all_bonds]

    g = dgl.graph((src, dst), num_nodes=n_atoms)

    # node features
    g.ndata['atomic_num'] = torch.tensor(
        [atom.number for atom in atoms],
        dtype=torch.long
    )

    return g

def create_homogeneous_geometry_graph(atoms, builder_structural) -> dgl.DGLGraph:
    """Create homogenoeous DGL graph with RBF geometirc features.

    Args:
        atoms: ASE atoms object
        builder_structural: Structural Builder 

    Returns:
        dgl.DGLGraph: DGL homogeneous graph with geometric features
    """
    graph = builder_structural.get_graph()
    n_atoms = len(atoms)

    # Create edges
    src = [i for i, j, _ in graph['edge_list']]
    dst = [j for i, j, _ in graph['edge_list']]
    g = dgl.graph((src + dst, dst + src), num_nodes=n_atoms)

    # Add node features
    g.ndata['atomic_num'] = torch.tensor(
        [atom.number for atom in atoms],
        dtype=torch.long
    )

    # Add RBF features
    if 'rbf_features' in graph:
        rbf = graph['rbf_features']
        rbf_tensor = torch.tensor(rbf, dtype=torch.float)
        g.edata['rbf'] = torch.cat([rbf_tensor, rbf_tensor], dim=0)

    # Add degree features
    if 'degrees' in graph:
        g.ndata['degree'] = torch.tensor(graph['degrees'], dtype=torch.long)

    # Add ring features
    if 'in_ring' in graph:
        g.ndata['in_ring'] = torch.tensor(graph['in_ring'], dtype=torch.float)
    
    return g

def process_single_molecule(pdb_file: Path, verbose: bool=False):
    """Process a single PDB file to create different graph views
    Args:
        pdb_file: 
        verbose: whether to print progress
    
    Returns: 
        Dictionary with:
            g_homo_outline: homo outline graph
            g_homo_geometry: homo geometry graph
            mol_id: Molecule identifier
            ground_truth_coords
    """
    try: 
        # Read molecule
        atoms = read(pdb_file)
        mol_id = pdb_file.stem

        MIN_ATOMS = 10
        MAX_ATOMS = 300
        filtered = {
            'no_atoms':0,
            'too_less': 0,
            'too_much': 0
        }

        if len(atoms) == 0:
            filtered['no_atoms'] += 1
            if verbose:
                print(f"skipping {mol_id}: No atoms found")
            return None
        
        if len(atoms) >= MAX_ATOMS:
            filtered['too_much'] += 1
            if verbose:
                print(f"skipping {mol_id}: skipping due to too much nodes")
            return None
        
        if len(atoms) <= MIN_ATOMS:
            filtered['too_less'] += 1
            if verbose:
                print(f"skipping {mol_id}: skipping due to too less nodes")
            return None

        original_atom_count = len(atoms)

        builder_tmp = StructuralBuilder(atoms, multiplier=1.0)
        builder_tmp.build_connectivity()

        atoms = extract_single_molecule(atoms, builder_tmp)
        
        if atoms is None or len(atoms) == 0:
            if verbose:
                print(f"Skipping {mol_id}: No valid atoms in main component")
            return None
        
        if verbose and len(atoms) < original_atom_count:
            print(f"{mol_id}: extracted main molecule ({len(atoms)}/{original_atom_count} atoms)")

        # build graph:
        builder_structural = StructuralBuilder(atoms, multiplier=1.0)
        builder_structural.build_graph(
            apply_correction=False,
            detect_rings=True,
            verbose=False
        )

        # Get all bonds (union of all bond types)
        all_bonds = builder_structural.bonds
        
        # two views graph
        g_homo_outline = create_homogeneous_outline_graph(atoms, all_bonds)
        g_homo_geometry = create_homogeneous_geometry_graph(atoms, builder_structural)

        # Extract ground truth coordnates
        ground_truth_coords = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float
        ) #[N, 3]

        result = {
            'g_homo_outline': g_homo_outline,
            'g_homo_geometry': g_homo_geometry,
            'mol_id': mol_id,
            'ground_truth_coords':ground_truth_coords
        }

        if verbose:
            print(f"Processed {mol_id}: {len(atoms)} atoms, {len(all_bonds)} bonds")
            print(f"  No atoms: {filtered['no_atoms']}")
            print(f"  Too small (<= {MIN_ATOMS}): {filtered['too_small']}")
            print(f"  Too large (>= {MAX_ATOMS}): {filtered['too_large']}")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"Failed to process {pdb_file.name}: {e}")
        return None
    
def batch_process_directory(
        input_dir: str,
        output_dir: str,
        batch_size: int = 10000,
        n_workers : int = 0,
        verbose: bool = True
):
    """Process all PDB files in directory and save in batches

    Args:
        input_dir (str): Directory contrining PDB files
        output_dir (str): Directory to save .pt processed files
        batch_size (int): Defaults to 10000.
        n_workers (int): Parallel workers. Defaults to 0.
        verbose (bool): Wheter to print progress. Defaults to True.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PDB files
    pdb_files = sorted(input_path.glob('*.pdb'))
    n_files = len(pdb_files)

    if n_files == 0:
        print(f"No PDB files found in {input_dir}")
        return
    
    print(f"Found {n_files} PDB files")
    print(f"Processing in batches of {batch_size}...")
    print("="*60)

    # Process in batches
    current_batch = []
    batch_idx = 0
    success_count = 0
    failed_count = 0

    for pdb_file in tqdm(pdb_files, desc = "Processing"):
        result = process_single_molecule(pdb_file, verbose=False)

        if result is not None:
            current_batch.append(result)
            success_count += 1 
        else:
            failed_count += 1 
    
        # save batch 
        if len(current_batch) >= batch_size:
            output_file = output_path / f'pretrain_{batch_idx:04d}.pt'
            torch.save({'graphs': current_batch}, output_file)

            if verbose: 
                print(f"Saved batch {batch_idx}: {len(current_batch)} molecules to {output_file.name}")

            current_batch = []
            batch_idx += 1
    
    if len(current_batch) > 0:
        output_file = output_path / f'pretrain_{batch_idx:04d}.pt'
        torch.save({'graphs': current_batch}, output_file)

        if verbose: 
            print(f"saved final batch {batch_idx}: {len(current_batch)}")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total PDB files: {n_files}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Saved {batch_idx + 1} batch files")
    print(f"Output directory: {output_path}")
    print("="*60)      

def verify_processed_data(output_dir: str):
    """check the processed result by loading few examples

    Args:
        output_dir (str): Directory containing .pt files
    """
    output_path = Path(output_dir)
    pt_files = sorted(output_path.glob('pretrain_*.pt'))

    if len(pt_files) == 0:
        print(f"no .pt files found in {output_dir}")
        return
    
    print("\n" + "="*60)
    print("varifying")
    print("="*60)

    total_molecules = 0
    for pt_file in pt_files:
        data = torch.load(pt_file)
        n_graphs = len(data['graphs'])
        total_molecules += n_graphs
        print(f"{pt_file.name}: {n_graphs} molecules")

    print(f"\nTotal molecules: {total_molecules}")

    # check the first molecule
    if len(pt_files) > 0:
        print("\nChecking first molecule....")
        data = torch.load(pt_files[0])
        first_graph = data['graphs'][0]

        print(f"Molecule ID: {first_graph['mol_id']}")
        print(f"Outline graph: {first_graph['g_homo_outline']}")
        print(f"Geometry graph: {first_graph['g_homo_geometry']}")
        print(f"  Node features: {list(first_graph['g_homo_geometry'].ndata.keys())}")
        print(f"Ground truth coords shape: {first_graph['ground_truth_coords'].shape}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Batch process PDB 3DED data to create different views graphs"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='input directory')
    parser.add_argument('--output', type=str, required=True, help='output directory for .pt files')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Number of molecules per .pt file')
    parser.add_argument('--n_workers', type=int, default=0, 
                        help='Number of parallel workers')
    parser.add_argument('--verify', action='store_true', 
                        help='Verify processed data')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    args = parser.parse_args()

    batch_process_directory(
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        verbose=args.verbose
    )
    
    # Verify if requested
    if args.verify:
        verify_processed_data(args.output)

if __name__ == '__main__':
    main()


