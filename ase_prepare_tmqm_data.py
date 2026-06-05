import csv, os, random, torch
import numpy as np
from tqdm import tqdm
from ase.data import atomic_numbers as ASE_Z
import dgl

import json
from ase_process_topo import gaussian_rbf
from ase_sa import reconstruct_molecular_graph
from ase_finetune import ensure_node_features

BOND_CFG   = json.load(open('results/graph_calibrate/bond_config.json'))
MULTIPLIER = BOND_CFG['bond_multiplier']

random.seed(42)
np.random.seed(42)

TMQM_DIR  = 'data/tmQM/tmQM'
XYZ_FILES = [f'{TMQM_DIR}/tmQM_X{i}.xyz' for i in [1, 2, 3]]
BO_FILES  = [f'{TMQM_DIR}/tmQM_X{i}.BO'  for i in [1, 2, 3]]
Q_FILE    = f'{TMQM_DIR}/tmQM_X.q'
Y_CSV     = f'{TMQM_DIR}/tmQM_y.csv'
OUT_DIR   = 'data/processed_data_2/tmqm'

TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))
TYPE_A = [6, 7, 8]    # C, N, O
TYPE_B = [17, 35]     # Cl, Br

TARGET_PROPS = [
    'Electronic_E',    # total electronic energy (Ha)
    'Dispersion_E',    # DFT-D3 dispersion energy (Ha)
    'Dipole_M',        # dipole moment (D)
    'Metal_q',         # Mulliken charge on metal centre
    'HL_Gap',          # HOMO–LUMO gap (Ha)
    'HOMO_Energy',     # HOMO energy (Ha)
    'LUMO_Energy',     # LUMO energy (Ha)
    'Polarizability',  # molecular polarizability (a₀³)
]

# Z -> (d_electrons for M²⁺, ionic_radius_Å M²⁺ 6-coord, Pauling EN, IE1 eV)
TABULATED_CLEAN = {
    1:  {"symbol": "H",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.20, "ie1_eV": 13.5984},
    5:  {"symbol": "B",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.04, "ie1_eV": 8.2980},
    6:  {"symbol": "C",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.55, "ie1_eV": 11.2603},
    7:  {"symbol": "N",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 3.04, "ie1_eV": 14.5341},
    8:  {"symbol": "O",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 3.44, "ie1_eV": 13.6181},
    9:  {"symbol": "F",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 3.98, "ie1_eV": 17.4228},
    14: {"symbol": "Si", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.90, "ie1_eV": 8.1517},
    15: {"symbol": "P",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.19, "ie1_eV": 10.4867},
    16: {"symbol": "S",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.58, "ie1_eV": 10.3600},
    17: {"symbol": "Cl", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 3.16, "ie1_eV": 12.9676},
    33: {"symbol": "As", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.18, "ie1_eV": 9.7886},
    34: {"symbol": "Se", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.55, "ie1_eV": 9.7524},
    35: {"symbol": "Br", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.96, "ie1_eV": 11.8138},
    52: {"symbol": "Te", "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.10, "ie1_eV": 9.0098},
    53: {"symbol": "I",  "d_electrons_M2plus": 0,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.66, "ie1_eV": 10.4513},

    # 3d transition metals
    21: {"symbol": "Sc", "d_electrons_M2plus": 1,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.36, "ie1_eV": 6.5615},
    22: {"symbol": "Ti", "d_electrons_M2plus": 2,  "ionic_radius_M2plus_6coord_ls_A": 1.00, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.54, "ie1_eV": 6.8281},
    23: {"symbol": "V",  "d_electrons_M2plus": 3,  "ionic_radius_M2plus_6coord_ls_A": 0.79, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.63, "ie1_eV": 6.7462},
    24: {"symbol": "Cr", "d_electrons_M2plus": 4,  "ionic_radius_M2plus_6coord_ls_A": 0.73, "ionic_radius_M2plus_6coord_hs_A": 0.80, "pauling_en": 1.66, "ie1_eV": 6.7665},
    25: {"symbol": "Mn", "d_electrons_M2plus": 5,  "ionic_radius_M2plus_6coord_ls_A": 0.67, "ionic_radius_M2plus_6coord_hs_A": 0.83, "pauling_en": 1.55, "ie1_eV": 7.4340},
    26: {"symbol": "Fe", "d_electrons_M2plus": 6,  "ionic_radius_M2plus_6coord_ls_A": 0.61, "ionic_radius_M2plus_6coord_hs_A": 0.78, "pauling_en": 1.83, "ie1_eV": 7.9025},
    27: {"symbol": "Co", "d_electrons_M2plus": 7,  "ionic_radius_M2plus_6coord_ls_A": 0.65, "ionic_radius_M2plus_6coord_hs_A": 0.745,"pauling_en": 1.88, "ie1_eV": 7.8810},
    28: {"symbol": "Ni", "d_electrons_M2plus": 8,  "ionic_radius_M2plus_6coord_ls_A": 0.69, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.91, "ie1_eV": 7.6399},
    29: {"symbol": "Cu", "d_electrons_M2plus": 9,  "ionic_radius_M2plus_6coord_ls_A": 0.73, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.90, "ie1_eV": 7.7264},
    30: {"symbol": "Zn", "d_electrons_M2plus": 10, "ionic_radius_M2plus_6coord_ls_A": 0.74, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.65, "ie1_eV": 9.3942},

    # 4d transition metals
    # For several 4d entries, an octahedral M2+ Shannon value is not consistently available
    # in the same source convention, so these are left as None rather than guessed.
    39: {"symbol": "Y",  "d_electrons_M2plus": 1,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.22, "ie1_eV": 6.2173},
    40: {"symbol": "Zr", "d_electrons_M2plus": 2,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.33, "ie1_eV": 6.6339},
    41: {"symbol": "Nb", "d_electrons_M2plus": 3,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.60, "ie1_eV": 6.7589},
    42: {"symbol": "Mo", "d_electrons_M2plus": 4,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.16, "ie1_eV": 7.0924},
    43: {"symbol": "Tc", "d_electrons_M2plus": 5,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.90, "ie1_eV": 7.2800},
    44: {"symbol": "Ru", "d_electrons_M2plus": 6,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.20, "ie1_eV": 7.3605},
    45: {"symbol": "Rh", "d_electrons_M2plus": 7,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.28, "ie1_eV": 7.4589},
    46: {"symbol": "Pd", "d_electrons_M2plus": 8,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.20, "ie1_eV": 8.3369},
    47: {"symbol": "Ag", "d_electrons_M2plus": 9,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.93, "ie1_eV": 7.5762},
    48: {"symbol": "Cd", "d_electrons_M2plus": 10, "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.69, "ie1_eV": 8.9938},

    # 5d transition metals
    72: {"symbol": "Hf", "d_electrons_M2plus": 2,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.30, "ie1_eV": 6.8251},
    73: {"symbol": "Ta", "d_electrons_M2plus": 3,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.50, "ie1_eV": 7.5496},
    74: {"symbol": "W",  "d_electrons_M2plus": 4,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.36, "ie1_eV": 7.8640},
    75: {"symbol": "Re", "d_electrons_M2plus": 5,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 1.90, "ie1_eV": 7.8335},
    76: {"symbol": "Os", "d_electrons_M2plus": 6,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.20, "ie1_eV": 8.4382},
    77: {"symbol": "Ir", "d_electrons_M2plus": 7,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.20, "ie1_eV": 8.9670},
    78: {"symbol": "Pt", "d_electrons_M2plus": 8,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.28, "ie1_eV": 8.9588},
    79: {"symbol": "Au", "d_electrons_M2plus": 9,  "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.54, "ie1_eV": 9.2256},
    80: {"symbol": "Hg", "d_electrons_M2plus": 10, "ionic_radius_M2plus_6coord_ls_A": None, "ionic_radius_M2plus_6coord_hs_A": None, "pauling_en": 2.00, "ie1_eV": 10.4375},
}

def _extract_raw(entry):
    return [
        float(entry['d_electrons_M2plus']),
        float(entry['pauling_en']),
        float(entry['ie1_eV']),
    ]

N_TAB = 3

# Build lookup: Z -> raw [3] float list
TABULATED_RAW = {z:_extract_raw(e) for z, e in TABULATED_CLEAN.items()}

# compute normalisation stats over all knwon elements 
_tab_arr = np.array(list(TABULATED_RAW.values()), dtype=np.float32) #[n_elem, 3]
_TAB_MEAN = _tab_arr.mean(axis=0) #[3]
_TAB_STD = _tab_arr.std(axis=0) + 1e-8 #[3]

def get_tab_feat(z):
    raw = np.array(TABULATED_RAW.get(z, _TAB_MEAN.tolist()), dtype=np.float32)
    return (raw - _TAB_MEAN) / _TAB_STD

def load_targets(y_csv):
    out = {}
    with open(y_csv, newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            code = row['CSD_code'].strip()
            out[code] = {
                p: float(row[p]) for p in TARGET_PROPS
                if row.get(p, '').strip() not in ('', 'nan')
            }
    return out

def iter_tmqm(xyz_files, q_file, bo_files):
    """
    Generator — yields one molecule at a time as:
        (csd_code, symbols, positions, charges, bond_pairs)
    All iterators advance in lockstep (X.q covers X1+X2+X3 in order).
    """
    q_f = open(q_file, 'r')

    for xyz_path, bo_path in zip(xyz_files, bo_files):
        bo_f = open(bo_path, 'r')
        with open(xyz_path, 'r') as xyz_f:
            while True:
                n_line = xyz_f.readline()
                if not n_line:
                    break
                if not n_line.strip():
                    continue
                n_atoms  = int(n_line.strip())
                comment  = xyz_f.readline()
                csd_code = comment.split('|')[0].split('=')[1].strip()

                syms, pos = [], []
                for _ in range(n_atoms):
                    parts = xyz_f.readline().split()
                    syms.append(parts[0])
                    pos.append([float(x) for x in parts[1:4]])

                # skip trailing lines from previous mol + header (until CSD_code)
                while True:
                    line = q_f.readline()
                    if not line or line.startswith('CSD_code'):
                        break
                charges = []
                for _ in range(n_atoms):
                    charges.append(float(q_f.readline().split()[1]))

                # skip trailing blank + header (CSD_code line consumed here)
                while True:
                    line = bo_f.readline()
                    if not line or line.startswith('CSD_code'):
                        break
                # read until next blank line — robust to atom count mismatches
                bond_pairs = []
                while True:
                    line = bo_f.readline()
                    if not line or not line.strip():
                        break
                    parts = line.split()
                    i = int(parts[0]) - 1          # 1-indexed → 0-indexed
                    j_off = 3
                    while j_off + 2 <= len(parts) - 1:
                        j  = int(parts[j_off + 1]) - 1
                        bo = float(parts[j_off + 2])
                        if i < j:                  # store once per pair
                            bond_pairs.append((i, j, bo))
                        j_off += 3

                yield csd_code, syms, pos, charges, bond_pairs
        bo_f.close()

    q_f.close()
    
def build_graph(z_arr, pos_arr, charges_arr, bos_arr):
    """Build graph using covalent-radius heuristic (same as CSD benchmark).
    DFT features (charges, total BO) attached separately from .BO data."""
    N = len(z_arr)
    z_t   = torch.tensor(z_arr, dtype=torch.long)
    pos_t = torch.tensor(pos_arr, dtype=torch.float32)

    # Heuristic graph — matches CSD eval pipeline exactly
    _, g = reconstruct_molecular_graph(z_t, pos_t, MULTIPLIER)
    g = ensure_node_features(g, 53)

    # tab_feat [N, 3]
    g.ndata['tab_feat'] = torch.tensor(
        [get_tab_feat(z) for z in z_arr], dtype=torch.float32)

    # dft_feat [N, 2]: charges from .q, total BO from .BO
    total_bo = np.zeros(N, dtype=np.float32)
    for i, j, bo in bos_arr:
        if bo > 0.15:
            total_bo[i] += bo
            total_bo[j] += bo

    dft = torch.zeros(N, 2, dtype=torch.float32)
    if charges_arr is not None:
        dft[:, 0] = torch.tensor(np.clip(charges_arr / 2.0, -1.5, 1.5))
    dft[:, 1] = torch.tensor(np.clip(total_bo / 6.0, 0.0, 1.5))
    g.ndata['dft_feat'] = dft

    return g

def make_record(z_clean, pos_arr, site_indices, z_new_list,
                charges_arr, bond_pairs, csd_code, targets):
    z_arr = z_clean.numpy().copy()
    for site, z_new in zip(site_indices, z_new_list):
        z_arr[site] = z_new
    try:
        g = build_graph(z_arr, pos_arr, charges_arr, bond_pairs)
    except Exception as e:
        if random.random() < 0.001:   # sample 0.1% of failures
            print(f"  [make_record FAIL] {csd_code}: {type(e).__name__}: {e}")
        return None
    N = len(z_arr)
    node_labels = torch.zeros(N, dtype=torch.long)
    for site in site_indices:
        node_labels[site] = 1
    return {
        'graph': g, 'node_labels': node_labels,
        'corrupted_sites': list(site_indices),
        'z_old': [int(z_clean[s]) for s in site_indices],
        'z_new': list(z_new_list),
        'csd_code': csd_code, 'targets': targets,
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tmQM_y.csv targets …")
    all_targets = load_targets(Y_CSV)
    print(f"  {len(all_targets)} molecules with targets")

    # ── Pass 1: scan all molecules, filter, collect ──────────────────────────
    print("\nPass 1: scanning tmQM …")
    valid = []   # (csd_code, z_arr, pos_arr, chg_arr, bond_pairs, metal_sites, targets)
    n_total = n_no_metal = n_too_small = 0

    for csd_code, syms, pos, charges, bos in tqdm(
            iter_tmqm(XYZ_FILES, Q_FILE, BO_FILES), total=108541, desc='scan'):
        n_total += 1

        # Filter H atoms
        keep = [i for i, s in enumerate(syms) if s != 'H']
        keep_set = set(keep)
        remap = {old: new for new, old in enumerate(keep)}
        if len(keep) < 5:
            n_too_small += 1
            continue

        syms_f  = [syms[i]    for i in keep]
        pos_f   = [pos[i]     for i in keep]
        chg_f   = [charges[i] for i in keep]
        bonds_f = [(remap[i], remap[j], bo)
                    for i, j, bo in bos           
                    if i in keep_set and j in keep_set]

        z_arr = np.array([ASE_Z.get(s, 0) for s in syms_f], dtype=np.int32)
        metals = [i for i, z in enumerate(z_arr) if z in TRANSITION_METALS]
        if not metals:
            n_no_metal += 1
            continue

        valid.append((
            csd_code,
            z_arr,
            np.array(pos_f,  dtype=np.float32),
            np.array(chg_f,  dtype=np.float32),
            bonds_f,
            metals,
            all_targets.get(csd_code, {}),
        ))

    print(f"  Scanned  : {n_total}")
    print(f"  No metal : {n_no_metal}")
    print(f"  Too small: {n_too_small}")
    print(f"  Valid    : {len(valid)}")

    # ── Split 70 / 15 / 15 ──────────────────────────────────────────────────
    random.shuffle(valid)
    n    = len(valid)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    splits = {
        'train': valid[:n_tr],
        'val':   valid[n_tr : n_tr + n_va],
        'test':  valid[n_tr + n_va:],
    }

    # ── Pass 2: build records ────────────────────────────────────────────────
    for split, mols in splits.items():
        print(f"\n[{split}] {len(mols)} molecules → building records …")
        records = []
        n_skip  = 0

        for csd_code, z_arr, pos_arr, chg_arr, bond_pairs, metal_sites, tgts in tqdm(mols, desc=split):
            z_clean = torch.tensor(z_arr, dtype=torch.long)

            # Clean record (node_labels all zero)
            try:
                g_cln = build_graph(z_arr, pos_arr, chg_arr, bond_pairs)
                N = g_cln.num_nodes()
                records.append({
                    'graph': g_cln,
                    'node_labels': torch.zeros(N, dtype=torch.long),
                    'corrupted_sites': [], 'z_old': [], 'z_new': [],
                    'csd_code': csd_code, 'targets': tgts,
                })
            except Exception:
                n_skip += 1
                continue
                

            # Try ALL metal sites × ALL corruption types (heuristic rejects most)
            for site in metal_sites:
                for z_new in TYPE_A + TYPE_B:
                    rec = make_record(z_clean, pos_arr, [site], [z_new],
                                      chg_arr, bond_pairs, csd_code, tgts)
                    if rec: records.append(rec)

            # Multi-site: two metals simultaneously (3 attempts)
            if len(metal_sites) >= 2:
                for _ in range(3):
                    sites  = random.sample(metal_sites, 2)
                    z_news = [random.choice(TYPE_A + TYPE_B) for _ in sites]
                    rec    = make_record(z_clean, pos_arr, sites, z_news,
                                         chg_arr, bond_pairs, csd_code, tgts)
                    if rec: records.append(rec)
                    
            # Corrupt ligands
            ligand_sites = []
            for site in metal_sites:
                for nbr in range(len(z_arr)):
                    if nbr == site:
                        continue
                    d = np.linalg.norm(pos_arr[site] - pos_arr[nbr])
                    if d < 2.8 and z_arr[nbr] in (7, 8): # N or O with bonding distance
                        ligand_sites.append((nbr, 6)) # corrupt to C 
            
            # single ligand corruption (deduplicate — same atom may neighbor multiple metals)
            seen_lig = set()
            for site, z_new in ligand_sites:
                if site in seen_lig:
                    continue
                seen_lig.add(site)
                rec = make_record(z_clean, pos_arr, [site], [z_new],
                                  chg_arr, bond_pairs, csd_code, tgts)
                if rec: records.append(rec)

            # combined: metal + its ligands corrupted together
            for m_site in metal_sites:
                nearby_ligs = [(s, z) for s, z in ligand_sites
                               if np.linalg.norm(pos_arr[m_site] - pos_arr[s]) < 2.8]
                if nearby_ligs:
                    all_sites = [m_site] + [s for s, z in nearby_ligs]
                    all_znew = [random.choice(TYPE_A)] + [z for s, z in nearby_ligs]
                    rec = make_record(z_clean, pos_arr, all_sites, all_znew,
                                      chg_arr, bond_pairs, csd_code, tgts)
                    if rec: records.append(rec)

        n_clean = sum(1 for r in records if not r['corrupted_sites'])
        n_corr  = len(records) - n_clean
        print(f"  Records : {len(records)}  (clean {n_clean} / corrupted {n_corr})")
        print(f"  Skipped : {n_skip}")

        out_path = os.path.join(OUT_DIR, f'{split}_tmqm.pt')
        torch.save({
            'records':      records,
            'target_props': TARGET_PROPS,
        }, out_path)
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()