import glob, json, random, os, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ase.io import read
from ase_process_topo import MolecularGraphBuilder
from ase_batch_process import extract_single_molecule

random.seed(42)
torch.manual_seed(42)

MULTIPLIER = json.load(open('results/graph_calibrate/bond_config.json'))['bond_multiplier']
PDB_FILES = glob.glob("data/mettalic/*.pdb")
random.shuffle(PDB_FILES)

n = len(PDB_FILES)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
TEST_FILES = PDB_FILES[n_train + n_val:] 

TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))
TYPE_A = [6, 7, 8]
TYPE_B = [17, 35]

OUT_DIR = Path('data/processed_data_2/sa_benchmark')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Test files: {len(TEST_FILES)}")

def build_connectivity(builder, n_atoms):
    conn = torch.zeros(n_atoms, n_atoms, dtype=torch.float)
    for (i, j) in builder.bonds:
        conn[i, j] = 1.0
        conn[j, i] = 1.0
    return conn

def make_mol_dict(path, atoms, z_gt, z_corrupted, corrupted_sites, corruption_type,
                  builder, multiplier):
    n = len(atoms)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    connectivity = build_connectivity(builder, n)
    return {
        'mol_id':            path,
        'atomic_numbers':    z_corrupted.clone(),
        'gt_atomic_numbers': z_gt.clone(),
        'positions':         pos,
        'connectivity':      connectivity,
        'corrupted_sites':   corrupted_sites,       # oracle forced_edit_sites
        'corruption_type':   corruption_type,
    }
    
# build benchmark
records_a = [] # single-site Type A
records_b = [] # singel-site Type B
records_multi = [] 
skipped = 0 

for path in tqdm(TEST_FILES, desc='Preparing benchmark'):
    try: 
        atoms = read(path)
        builder_tmp = MolecularGraphBuilder(atoms, multiplier=MULTIPLIER)
        builder_tmp.build_connectivity()
        atoms = extract_single_molecule(atoms, builder_tmp)
        if atoms is None or len(atoms) < 5:
            skipped += 1
            continue
    except Exception:
        skipped += 1
        continue
    
    n_atoms = len(atoms)
    z_gt = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    
    metal_indices = [i for i in range(n_atoms) if z_gt[i].item() in TRANSITION_METALS]
    if not metal_indices:
        skipped += 1
        continue
    
    try: 
        builder = MolecularGraphBuilder(atoms, multiplier=MULTIPLIER)
        builder.build_graph(apply_correction=False, use_aromatic_check=False, 
                            detect_rings=False, verbose=False)
    
    except Exception:
        skipped += 1
        continue
    
    site = random.choice(metal_indices)
    z_a = z_gt.clone()
    z_a[site] = random.choice(TYPE_A)
    records_a.append(make_mol_dict(path, atoms, z_gt, z_a,
                                   [site], 'type_a', builder, MULTIPLIER))
    
    site = random.choice(metal_indices)
    z_b = z_gt.clone()
    z_b[site] = random.choice(TYPE_B)
    records_b.append(make_mol_dict(path, atoms, z_gt, z_b,
                                   [site], 'type_b', builder, MULTIPLIER))
    
    if len(metal_indices) >= 2:
        n_corrupt = random.choice([2, min(3, len(metal_indices))])
        sites = random.sample(metal_indices, n_corrupt)
        z_m = z_gt.clone()
        for s in sites:
            z_m[s] = random.choice(TYPE_A + TYPE_B)
        records_multi.append(make_mol_dict(path, atoms, z_gt, z_m,
                                           sites, 'multi', builder, MULTIPLIER))
print(f"\nSkipped: {skipped}")
print(f"Type A records : {len(records_a)}")
print(f"Type B records : {len(records_b)}")
print(f"Multi records  : {len(records_multi)}")

# --- Save ---
for name, records in [('single_type_a', records_a),
                       ('single_type_b', records_b),
                       ('multi_site',    records_multi)]:
    out = OUT_DIR / f'{name}.pt'
    torch.save({'graphs': records,
                'config': {'multiplier': MULTIPLIER,
                           'source': 'csd_test_split',
                           'n_records': len(records)}}, str(out))
    print(f"Saved {len(records)} records → {out}")
    
