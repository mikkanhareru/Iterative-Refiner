import torch
import torch.nn.functional as F
import dgl
import numpy as np
import json, csv, argparse, itertools
from pathlib import Path
from tqdm import tqdm
from ase import Atoms as AseAtoms 

from ase_sa_update import identify_edit_sites, load_all_models, METAL_CANDS
from ase_sa import reconstruct_molecular_graph
from ase_finetune import ensure_node_features
from ase_process_topo import MolecularGraphBuilder
from ase_element_prior import build_histogram, z_to_idx

MULTIPLIER = json.load(open('results/graph_calibrate/bond_config.json'))['bond_multiplier']

def _get_site_candidates(corrupted_Z, g_base, edit_sites, elem_prior, top_k, device):
    """Use ElementPrior to rank METAL_CANDS per site; return top_k per site.

    Batches all sites in a single forward pass.
    Metals with the same vocab class (e.g. RARE_ELEMENT) get the same score
    and are ranked by their order in METAL_CANDS.
    """
    src_cpu, dst_cpu = g_base.edges()
    src_cpu, dst_cpu = src_cpu.cpu(), dst_cpu.cpu()

    hists = []
    for site in edit_sites:
        nbr_z = corrupted_Z[dst_cpu[src_cpu == site]]
        hists.append(build_histogram(nbr_z))

    X = torch.stack(hists).to(device)            # [n_sites, N_VOCAB]
    with torch.no_grad():
        log_probs = elem_prior(X)                # [n_sites, N_VOCAB]

    site_cands = []
    for k in range(len(edit_sites)):
        ranked = sorted(METAL_CANDS,
                        key=lambda z: log_probs[k, z_to_idx(z)].item(),
                        reverse=True)
        site_cands.append(ranked[:top_k])
    return site_cands


MAX_COMBOS = 125   # hard ceiling on reconstruct calls per molecule

def joint_search(corrupted_Z, pos, edit_sites, topo_verifier, elem_prior, device,
                 top_k=5, g_base=None):
    """Exhaustive combo search with proper per-combo graph reconstruction.

    Prunes METAL_CANDS (18) → top_k (default 5) per site via ElementPrior.
    top_k is automatically reduced so total combos never exceed MAX_COMBOS,
    preventing hangs when guided site-detection returns spurious extra sites.

    g_base: pre-built DGL geometry graph for corrupted_Z (avoids a redundant
            reconstruct call when the caller already built it).
    """
    n_sites = len(edit_sites)

    if g_base is None:
        try:
            _, g_base = reconstruct_molecular_graph(corrupted_Z, pos, multiplier=MULTIPLIER)
            g_base    = ensure_node_features(g_base, 53)
        except Exception:
            return None, -1.0

    # Adapt top_k so combos <= MAX_COMBOS regardless of how many sites were found.
    # e.g. n_sites=1→top_k=5, n_sites=3→top_k=5 (125), n_sites=5→top_k=2 (32)
    effective_top_k = max(1, min(top_k, int(MAX_COMBOS ** (1.0 / n_sites))))

    site_cands   = _get_site_candidates(corrupted_Z, g_base, edit_sites,
                                        elem_prior, effective_top_k, device)
    combos       = list(itertools.product(*site_cands))
    CHUNK        = 64     # smaller than before; reconstruct is heavier than clone
    all_scores   = []
    valid_combos = []

    for start in range(0, len(combos), CHUNK):
        chunk  = combos[start:start+CHUNK]
        graphs = []
        for combo in chunk:
            z_cand = corrupted_Z.clone()
            for k, site in enumerate(edit_sites):
                z_cand[site] = combo[k]
            try:
                _, g = reconstruct_molecular_graph(z_cand, pos, multiplier=MULTIPLIER)
                g    = ensure_node_features(g, 53)
                graphs.append(g)
                valid_combos.append(combo)
            except Exception:
                pass   # skip combos that fail graph build
        if graphs:
            all_scores.extend(topo_verifier.score_batch(graphs))

    if not all_scores:
        return None, -1.0

    best_c     = int(np.argmax(all_scores))
    best_combo = valid_combos[best_c]
    assignment = {edit_sites[k]: best_combo[k] for k in range(n_sites)}
    return assignment, all_scores[best_c]



def compute_bond_f1(repaired_Z, pos, gt_connectivity):
    n        = len(repaired_Z)
    gt_bonds = {(i,j) for i in range(n) for j in range(i+1,n) if gt_connectivity[i,j] > 0}
    atoms    = AseAtoms(numbers=repaired_Z, positions=pos)
    builder  = MolecularGraphBuilder(atoms, multiplier=MULTIPLIER)
    builder.build_graph(apply_correction=False, use_aromatic_check=False,
                        detect_rings=False, verbose=False)
    pred     = {(min(a,b), max(a,b)) for a,b in builder.bonds}
    if not gt_bonds and not pred: return 1.0
    if not gt_bonds or  not pred: return 0.0
    tp = len(pred & gt_bonds)
    p  = tp/len(pred); r = tp/len(gt_bonds)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0


# -----------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------
def evaluate_split(benchmark_file, output_dir, oracle=False, n_mols=None,
                   device_str='cuda', verbose=False, top_k=5):

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    topo_verifier, _, elem_prior, _ = load_all_models(device)

    data      = torch.load(benchmark_file)
    molecules = data['graphs']
    if n_mols: molecules = molecules[:n_mols]

    split_name = Path(benchmark_file).stem
    mode       = ('oracle' if oracle else 'guided') + '_joint'
    print(f"\n[Joint Search] {split_name}  mode={mode}  n={len(molecules)}")

    results = []

    for idx, mol in enumerate(tqdm(molecules, desc=f"{split_name}/{mode}")):
        corrupted_Z     = mol['atomic_numbers']
        gt_Z            = mol['gt_atomic_numbers']
        pos             = mol['positions']
        connectivity    = mol['connectivity']
        corrupted_sites = mol['corrupted_sites']
        corruption_type = mol['corruption_type']
        mol_id          = mol.get('mol_id', str(idx))

        # Build g_base ONCE — reused for s_before, identify_edit_sites, joint_search.
        # This avoids 2-3 redundant reconstruct_molecular_graph calls per molecule.
        try:
            _, g_base = reconstruct_molecular_graph(corrupted_Z, pos, multiplier=MULTIPLIER)
            g_base    = ensure_node_features(g_base, 53)
            s_before  = topo_verifier.score_batch([g_base])[0]
        except Exception:
            g_base   = None
            s_before = float('nan')

        # Identify edit sites
        if oracle:
            edit_sites = list(corrupted_sites)
        else:
            try:
                edit_sites = identify_edit_sites(corrupted_Z, g_base) if g_base is not None else []
            except:
                edit_sites = []

        if len(edit_sites) == 0:
            repaired_Z = corrupted_Z.clone().numpy()
            s_after    = s_before
        else:
            assignment, s_after = joint_search(
                corrupted_Z, pos, edit_sites, topo_verifier, elem_prior, device,
                top_k=top_k, g_base=g_base
            )
            repaired_Z = corrupted_Z.clone().numpy()
            if assignment:
                for site, z_new in assignment.items():
                    repaired_Z[site] = z_new

            if verbose:
                changes = {s: (int(corrupted_Z[s]), int(repaired_Z[s]))
                           for s in edit_sites if corrupted_Z[s] != repaired_Z[s]}
                eff_k    = max(1, min(top_k, int(MAX_COMBOS ** (1.0 / len(edit_sites)))))
                print(f"  [{mol_id}]  K={len(edit_sites)}  top_k={eff_k}  combos={eff_k**len(edit_sites)}"
                      f"  S:{s_before:.3f}→{s_after:.3f}  {changes}")

        n_sites   = len(corrupted_sites)
        n_correct = sum(int(repaired_Z[s]) == int(gt_Z[s].item()) for s in corrupted_sites)
        site_acc  = n_correct / n_sites if n_sites > 0 else float('nan')
        was_fixed = int(not np.array_equal(repaired_Z, corrupted_Z.numpy()))
        delta     = (float(s_after) - float(s_before)
                     if not np.isnan(s_before) else float('nan'))
        success   = int(delta > 0.5) if not np.isnan(delta) else 0

        try:    bond_f1 = compute_bond_f1(repaired_Z, pos.numpy(), connectivity.numpy())
        except: bond_f1 = float('nan')

        results.append({
            'mol_id': mol_id, 'corruption_type': corruption_type,
            'n_corrupted_sites': n_sites, 'n_correct_sites': n_correct,
            'site_accuracy': site_acc,
            's_before':   round(float(s_before), 6) if not np.isnan(s_before) else None,
            's_after':    round(float(s_after),  6) if not np.isnan(s_after)  else None,
            'delta_stopo':round(float(delta),    6) if not np.isnan(delta)    else None,
            'bond_f1':    round(float(bond_f1),  6) if not np.isnan(bond_f1)  else None,
            'was_fixed': was_fixed, 'success': success,
        })

    _save_and_print(results, split_name, mode, output_dir)


def _save_and_print(results, split_name, mode, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f'{split_name}_{mode}_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    def sm(vals):
        v = [x for x in vals if x is not None and not np.isnan(x)]
        return float(np.mean(v)) if v else float('nan')

    n_total = len(results); n_fixed = sum(r['was_fixed'] for r in results)
    summary = {
        'split': split_name, 'mode': mode,
        'n_total': n_total, 'n_fixed': n_fixed,
        'fix_rate':           n_fixed/n_total if n_total else float('nan'),
        'mean_site_accuracy': sm([r['site_accuracy'] for r in results]),
        'mean_delta_stopo':   sm([r['delta_stopo']   for r in results]),
        'mean_s_before':      sm([r['s_before']      for r in results]),
        'mean_s_after':       sm([r['s_after']       for r in results]),
        'mean_bond_f1':       sm([r['bond_f1']       for r in results]),
        'success_rate':       sm([r['success']       for r in results]),
    }
    for t in sorted(set(r['corruption_type'] for r in results)):
        sub = [r for r in results if r['corruption_type'] == t]
        summary[f'n_{t}']                = len(sub)
        summary[f'site_accuracy_{t}']    = sm([r['site_accuracy'] for r in sub])
        summary[f'success_rate_{t}']     = sm([r['success']       for r in sub])
        summary[f'mean_delta_stopo_{t}'] = sm([r['delta_stopo']   for r in sub])
        summary[f'mean_bond_f1_{t}']     = sm([r['bond_f1']       for r in sub])

    json_path = out / f'{split_name}_{mode}_summary.json'
    with open(json_path, 'w') as f: json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\n  {split_name.upper()}  [{mode}]\n{'='*60}")
    print(f"  Total molecules       : {n_total}")
    print(f"  Changed something     : {n_fixed}  ({n_fixed/n_total*100:.1f}%)")
    print(f"  Top-1 Site Accuracy   : {summary['mean_site_accuracy']*100:.1f}%")
    print(f"  Mean ΔS_topo          : {summary['mean_delta_stopo']:+.4f}")
    print(f"  S_topo before → after : {summary['mean_s_before']:.4f} → {summary['mean_s_after']:.4f}")
    print(f"  Mean Bond F1          : {summary['mean_bond_f1']*100:.1f}%")
    print(f"  Success@budget(>0.5)  : {summary['success_rate']*100:.1f}%")
    print(f"{'='*60}")
    print(f"  JSON → {json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split',         default='multi_site',
                   choices=['single_type_a','single_type_b','multi_site','all'])
    p.add_argument('--benchmark_dir', default='data/processed_data_2/sa_benchmark')
    p.add_argument('--output',        default='results/joint_search')
    p.add_argument('--oracle',        action='store_true')
    p.add_argument('--n_mols',        type=int,  default=None)
    p.add_argument('--device',        default='cuda')
    p.add_argument('--verbose',       action='store_true')
    p.add_argument('--top_k',         type=int,  default=5,
                   help='ElementPrior top-k candidates per site (default 5). '
                        'Combos = top_k^n_sites. Reduce for speed (3→27 combos max).')
    args = p.parse_args()

    splits = (['single_type_a','single_type_b','multi_site']
              if args.split == 'all' else [args.split])

    for split in splits:
        bfile = Path(args.benchmark_dir) / f'{split}.pt'
        if not bfile.exists(): print(f"Not found: {bfile}"); continue
        evaluate_split(str(bfile), args.output,
                       oracle=args.oracle, n_mols=args.n_mols,
                       device_str=args.device, verbose=args.verbose,
                       top_k=args.top_k)

if __name__ == '__main__':
    main()