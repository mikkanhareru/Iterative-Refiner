import torch 
import json
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ase import Atoms as AseAtoms

from ase_sa_update import (
    simulated_annealing_updated,
    identify_edit_sites,
    load_all_models
)
from ase_sa import reconstruct_molecular_graph
from ase_process_topo import MolecularGraphBuilder
from ase_finetune import ensure_node_features

MULTIPLIER = json.load(open('results/graph_calibrate/bond_config.json'))['bond_multiplier']

# Benchmark: Bond F1
def compute_bond_f1(repaired_Z, pos, gt_connectivity, multiplier):
    n = len(repaired_Z) 
    
    gt_bonds = set()
    for i in range(n):
        for j in range(i + 1, n): 
            if gt_connectivity[i, j] > 0: # 1.0 = bond exist
                gt_bonds.add((i,j))
                
    atoms = AseAtoms(
        numbers=repaired_Z.cpu().numpy(),
        positions=pos.cpu().numpy(),
    )
    builder = MolecularGraphBuilder(atoms, multiplier=MULTIPLIER)
    builder.build_graph(
        apply_correction=False, use_aromatic_check=False,
        detect_rings=False, verbose=False
    )
    pred_bonds = set()
    for (i, j) in builder.bonds:
        a, b = (i, j) if i < j else (j, i)
        pred_bonds.add((a, b))
        
    if not gt_bonds and not pred_bonds:
        return 1.0, 1.0, 1.0
    if not pred_bonds or not gt_bonds:
        return 0.0, 0.0, 0.0
    
    tp = len(pred_bonds & gt_bonds)
    precision = tp / len(pred_bonds)
    recall = tp / len(gt_bonds)
    f1 = 2* precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 
    return precision, recall, f1

# Core Evaluation Loop
def evaluate_split(
    benchmark_file: str,
    output_dir: str,
    oracle: bool = False,
    max_steps: int = 1000,
    n_mols: int = None,
    lambda_elem: float = 0.1,
    lambda_penalty: float = 0.5,
    beta_cep: float = 1.0,
    device_str: str = 'cuda',
    verbose: bool = False,
    use_masked_gnn: bool = False,
    use_attn_cep: bool = False
    ):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    topo_verifer, cep_model, elem_prior, masked_GNN = load_all_models(device)
    masked_gnn = masked_GNN if use_masked_gnn else None
    
    data = torch.load(benchmark_file)
    molecules = data['graphs']
    if n_mols is not None:
        molecules = molecules[:n_mols]
        
    split_name = Path(benchmark_file).stem
    _elem_suffix = '_masked_gnn' if use_masked_gnn else ('_attn' if use_attn_cep else '')
    mode = ('oracle' if oracle else 'guided') + _elem_suffix
    print(f"\nEvaluating: {split_name} mode={mode} n={len(molecules)}")
    
    results = []
    
    for idx, mol in enumerate(tqdm(tqdm(molecules, desc=f"{split_name}/{mode}"))):
        corrupted_Z = mol['atomic_numbers']
        gt_Z = mol['gt_atomic_numbers']
        pos = mol['positions']
        connectivity = mol['connectivity']
        corrputed_sites = mol['corrupted_sites']
        corruption_type = mol['corruption_type']
        mol_id = mol.get('mol_id', str(idx))
        
        try: 
            s_before = topo_verifer.score(pos, corrupted_Z)
        except Exception:
            s_before = float('nan')
        
        forced = corrputed_sites if oracle else None
        best_Z, best_E, history = simulated_annealing_updated(
            mol_data = mol,
            topo_verifier=topo_verifer,
            cep_model=cep_model,
            elem_prior=elem_prior,
            device = device,
            multiplier=MULTIPLIER,
            max_steps=max_steps,
            lambda_elem=lambda_elem,
            lambda_penalty=lambda_penalty,
            beta_cep=beta_cep,
            forced_edit_sites=forced,
            verbose=verbose,
            masked_gnn = masked_gnn
        )
        
        if best_Z is None:
            repaired_Z = corrupted_Z
            was_fixed = False
        else:
            n_changes = int((best_Z != corrupted_Z).sum().item())
            repaired_Z = best_Z if n_changes > 0 else corrupted_Z
            was_fixed = (n_changes > 0)
            
        try:
            s_after = topo_verifer.score(pos, repaired_Z)
        except Exception:
            s_after = float('nan')
            
        n_sites = len(corrputed_sites)
        n_correct = sum(
            int(repaired_Z[s].item()) == int(gt_Z[s].item())
            for s in corrputed_sites
        )
        site_acc = n_correct / n_sites if n_sites > 0 else float('nan')
        
        try: 
            _, _, bond_f1 = compute_bond_f1(repaired_Z, pos, connectivity, MULTIPLIER)
        except Exception:
            bond_f1 = float('nan')
        
        delta_stopo = (s_after - s_before 
                       if not (np.isnan(s_before) or np.isnan(s_after))
                       else float('nan'))
        success = bool(delta_stopo > 0.5) if not np.isnan(delta_stopo) else False
        
        results.append({
            'mol_id':            mol_id,
            'corruption_type':   corruption_type,
            'n_corrupted_sites': n_sites,
            'n_correct_sites':   n_correct,
            'site_accuracy':     site_acc,
            's_before':          round(float(s_before), 6) if not np.isnan(s_before) else None,
            's_after':           round(float(s_after),  6) if not np.isnan(s_after)  else None,
            'delta_stopo':       round(float(delta_stopo), 6) if not np.isnan(delta_stopo) else None,
            'bond_f1':           round(float(bond_f1),  6) if not np.isnan(bond_f1)  else None,
            'was_fixed':         int(was_fixed),
            'success':           int(success),
        })

    # Save per-mol CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f'{split_name}_{mode}_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Aggregate
    def safe_mean(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    n_total = len(results)
    n_fixed = sum(r['was_fixed'] for r in results)

    summary = {
        'split':              split_name,
        'mode':               mode,
        'n_total':            n_total,
        'n_fixed':            n_fixed,
        'fix_rate':           n_fixed / n_total if n_total > 0 else float('nan'),
        'mean_site_accuracy': safe_mean([r['site_accuracy'] for r in results]),
        'mean_delta_stopo':   safe_mean([r['delta_stopo']   for r in results]),
        'mean_s_before':      safe_mean([r['s_before']      for r in results]),
        'mean_s_after':       safe_mean([r['s_after']       for r in results]),
        'mean_bond_f1':       safe_mean([r['bond_f1']       for r in results]),
        'success_rate':       safe_mean([r['success']       for r in results]),
    }

    corr_types = sorted(set(r['corruption_type'] for r in results))
    for t in corr_types:
        sub = [r for r in results if r['corruption_type'] == t]
        summary[f'n_{t}']                = len(sub)
        summary[f'site_accuracy_{t}']    = safe_mean([r['site_accuracy'] for r in sub])
        summary[f'success_rate_{t}']     = safe_mean([r['success']       for r in sub])
        summary[f'mean_delta_stopo_{t}'] = safe_mean([r['delta_stopo']   for r in sub])
        summary[f'mean_bond_f1_{t}']     = safe_mean([r['bond_f1']       for r in sub])

    json_path = out_dir / f'{split_name}_{mode}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()}  [{mode}]")
    print(f"{'='*60}")
    print(f"  Total molecules       : {n_total}")
    print(f"  SA changed something  : {n_fixed}  ({n_fixed/n_total*100:.1f}%)")
    print(f"  Top-1 Site Accuracy   : {summary['mean_site_accuracy']*100:.1f}%")
    print(f"  Mean ΔS_topo          : {summary['mean_delta_stopo']:+.4f}")
    print(f"  S_topo before → after : {summary['mean_s_before']:.4f} → {summary['mean_s_after']:.4f}")
    print(f"  Mean Bond F1          : {summary['mean_bond_f1']*100:.1f}%")
    print(f"  Success@budget(>0.5)  : {summary['success_rate']*100:.1f}%")
    if len(corr_types) > 1:
        print()
        for t in corr_types:
            sa = summary.get(f'site_accuracy_{t}',    float('nan'))
            sr = summary.get(f'success_rate_{t}',     float('nan'))
            ds = summary.get(f'mean_delta_stopo_{t}', float('nan'))
            bf = summary.get(f'mean_bond_f1_{t}',     float('nan'))
            n  = summary.get(f'n_{t}', 0)
            print(f"  [{t}] n={n}  site_acc={sa*100:.1f}%  "
                  f"success={sr*100:.1f}%  ΔS={ds:+.4f}  bond_f1={bf*100:.1f}%")
    print(f"{'='*60}")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")

    # ΔS_topo histogram
    deltas = [r['delta_stopo'] for r in results if r['delta_stopo'] is not None]
    if deltas:
        plt.figure(figsize=(8, 4))
        plt.hist(deltas, bins=50, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.3)
        plt.axvline(0,   color='red',   linestyle='--', linewidth=1.5, label='No change')
        plt.axvline(0.5, color='green', linestyle='--', linewidth=1.5, label='Success (>0.5)')
        plt.xlabel('ΔS_topo  (after − before)')
        plt.ylabel('Count')
        plt.title(f'{split_name} [{mode}]  — ΔS_topo distribution')
        plt.legend()
        plt.tight_layout()
        fig_path = out_dir / f'{split_name}_{mode}_delta_stopo.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Plot → {fig_path}")

    return summary


# ---------------------------------------------------------------------------
# FPR on clean molecules
# ---------------------------------------------------------------------------

def evaluate_clean_fpr(
    benchmark_file: str,
    output_dir: str,
    max_steps: int = 500,
    n_mols: int = None,
    lambda_elem: float = 0.1,
    lambda_penalty: float = 0.5,
    beta_cep: float = 1.0,
    device_str: str = 'cuda',
):
    """Run SA on clean (gt_atomic_numbers) structures.
    FPR = fraction where SA makes at least one change.
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    topo_verifier, cep_model, elem_prior, _ = load_all_models(device)

    data = torch.load(benchmark_file)
    molecules = data['graphs']
    if n_mols is not None:
        molecules = molecules[:n_mols]

    n_changed = 0
    n_total = 0

    for mol in tqdm(molecules, desc="FPR (clean)"):
        clean_mol = dict(mol)
        clean_mol['atomic_numbers'] = mol['gt_atomic_numbers'].clone()

        best_Z, _, _ = simulated_annealing_updated(
            mol_data=clean_mol,
            topo_verifier=topo_verifier,
            cep_model=cep_model,
            elem_prior=elem_prior,
            device=device,
            multiplier=MULTIPLIER,
            max_steps=max_steps,
            lambda_elem=lambda_elem,
            lambda_penalty=lambda_penalty,
            beta_cep=beta_cep,
            forced_edit_sites=None,
        )

        n_total += 1
        if best_Z is not None:
            n_changes = int((best_Z != clean_mol['atomic_numbers']).sum().item())
            if n_changes > 0:
                n_changed += 1

    fpr = n_changed / n_total if n_total > 0 else float('nan')
    print(f"\nFPR on clean mols: {n_changed}/{n_total} = {fpr*100:.1f}%")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = Path(benchmark_file).stem
    with open(out_dir / f'{split_name}_clean_fpr.json', 'w') as f:
        json.dump({'n_total': n_total, 'n_changed': n_changed, 'fpr': fpr}, f, indent=2)

    return fpr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',
                        type=str, default='single_type_a',
                        choices=['single_type_a', 'single_type_b', 'multi_site', 'all'])
    parser.add_argument('--benchmark_dir', type=str, default='data/processed_data_2/sa_benchmark')
    parser.add_argument('--output',        type=str, default='results/sa_evaluation_v3')
    parser.add_argument('--oracle',        action='store_true',
                        help='Use known corrupted sites (oracle ablation)')
    parser.add_argument('--fpr',           action='store_true',
                        help='Also measure FPR on clean structures')
    parser.add_argument('--max_steps',     type=int,   default=1000)
    parser.add_argument('--n_mols',        type=int,   default=None)
    parser.add_argument('--lambda_elem',   type=float, default=0.1)
    parser.add_argument('--lambda_penalty',type=float, default=0.5)
    parser.add_argument('--beta_cep',      type=float, default=1.0)
    parser.add_argument('--device',        type=str,   default='cuda')
    parser.add_argument('--verbose',       action='store_true')
    parser.add_argument('--elem_model', choices=['elem_prior', 'masked_gnn'], default='elem_prior')
    parser.add_argument('--attn', action='store_true', help='Use CEP v3 with active coord attention (cep_v3 checkpoint)')
    args = parser.parse_args()

    splits = (
        ['single_type_a', 'single_type_b', 'multi_site']
        if args.split == 'all' else [args.split]
    )

    all_summaries = []
    for split in splits:
        bfile = Path(args.benchmark_dir) / f'{split}.pt'
        if not bfile.exists():
            print(f"Not found: {bfile}")
            continue

        summary = evaluate_split(
            benchmark_file=str(bfile),
            output_dir=args.output,
            oracle=args.oracle,
            max_steps=args.max_steps,
            n_mols=args.n_mols,
            lambda_elem=args.lambda_elem,
            lambda_penalty=args.lambda_penalty,
            beta_cep=args.beta_cep,
            device_str=args.device,
            verbose=args.verbose,
            use_masked_gnn=(args.elem_model == 'masked_gnn'),
            use_attn_cep=args.attn
        )
        all_summaries.append(summary)

        if args.fpr:
            evaluate_clean_fpr(
                benchmark_file=str(bfile),
                output_dir=args.output,
                max_steps=min(args.max_steps, 500),
                n_mols=args.n_mols,
                lambda_elem=args.lambda_elem,
                lambda_penalty=args.lambda_penalty,
                beta_cep=args.beta_cep,
                device_str=args.device,
            )

    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print("COMBINED RESULTS")
        print(f"{'='*80}")
        print(f"{'Split':<22} {'Mode':<8} {'SiteAcc':>9} {'ΔS_topo':>9} {'BondF1':>9} {'Success':>9}")
        print('-' * 80)
        for s in all_summaries:
            print(f"{s['split']:<22} {s['mode']:<8} "
                  f"{s['mean_site_accuracy']*100:>8.1f}% "
                  f"{s['mean_delta_stopo']:>+9.4f} "
                  f"{s['mean_bond_f1']*100:>8.1f}% "
                  f"{s['success_rate']*100:>8.1f}%")
        print(f"{'='*80}")

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'combined_summary.json', 'w') as f:
            json.dump(all_summaries, f, indent=2)


if __name__ == '__main__':
    main()
