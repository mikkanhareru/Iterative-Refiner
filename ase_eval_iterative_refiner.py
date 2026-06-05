# ase_eval_iterative_refiner.py
import torch
import json, os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

from ase_train_iterative_refiner import IterativeRefiner, ELEM_VOCAB
from ase_train_refiner_v2 import IterativeRefinerV2
from ase_train_refiner_v3 import IterativeRefinerV3
from ase_prepare_tmqm_data import get_tab_feat, TRANSITION_METALS
from ase_topo_verifier import TopoVerifier
from ase_sa import reconstruct_molecular_graph
from ase_finetune import ensure_node_features
from ase_process_topo import MolecularGraphBuilder
from ase import Atoms

METALLIC_CKPT  = 'ase_checkpoints_metallic/checkpoint_best.pt'
BENCHMARK_DIR  = 'data/processed_data_2/sa_benchmark'
MULTIPLIER     = json.load(open('results/graph_calibrate/bond_config.json'))['bond_multiplier']
ELEM_VOCAB_SET = set(ELEM_VOCAB)


def get_bonds(z, pos, multiplier):
    atoms   = Atoms(numbers=z.numpy(), positions=pos.numpy())
    builder = MolecularGraphBuilder(atoms, multiplier=multiplier)
    builder.build_connectivity()
    return set(tuple(sorted(b)) for b in builder.bonds)


def bond_f1(z_pred, z_gt, pos, multiplier):
    try:
        pred_bonds = get_bonds(z_pred, pos, multiplier)
        gt_bonds   = get_bonds(z_gt,   pos, multiplier)
        tp = len(pred_bonds & gt_bonds)
        fp = len(pred_bonds - gt_bonds)
        fn = len(gt_bonds   - pred_bonds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_split(split_name, model, topo_verifier, device, n_mols=None, use_v2=False):
    records = torch.load(Path(BENCHMARK_DIR) / f'{split_name}.pt')['graphs']
    if n_mols:
        records = records[:n_mols]
    
    site_correct, site_total = 0, 0
    delta_s_list, bond_f1_list, success_list = [], [], []

    model.eval()
    with torch.no_grad():
        for rec in tqdm(records, desc=split_name):
            z_corr = rec['atomic_numbers']        # [N] corrupted
            z_gt   = rec['gt_atomic_numbers']     # [N] ground truth
            pos    = rec['positions']             # [N, 3]
            sites  = rec['corrupted_sites']

            # Build graph from corrupted structure
            try:
                _, g = reconstruct_molecular_graph(z_corr, pos, MULTIPLIER)
                g    = ensure_node_features(g, 53)
                if use_v2:
                    N_g = g.num_nodes()
                    g.ndata['tab_feat'] = torch.tensor(
                        [get_tab_feat(int(z)) for z in z_corr.tolist()],
                        dtype=torch.float32)
                    g.ndata['dft_feat'] = torch.zeros(N_g, 2, dtype=torch.float32)
                    # Add RBF edge features for V3 coord_scorer
                    if g.num_edges() > 0:
                        src, dst = g.edges()
                        pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else np.array(pos)
                        dists = np.linalg.norm(pos_np[src.numpy()] - pos_np[dst.numpy()], axis=1)
                        centers = np.linspace(0.0, 6.0, 50)
                        rbf = np.exp(-((dists.reshape(-1,1) - centers.reshape(1,-1))**2) / (2 * 0.3**2))
                        g.edata['rbf'] = torch.tensor(rbf, dtype=torch.float32)
                    else:
                        g.edata['rbf'] = torch.zeros(0, 50, dtype=torch.float32)
            except Exception:
                continue

            # Run model
            with torch.no_grad():
                if use_v2:
                    metal_mask = torch.tensor(
                        [int(z) in TRANSITION_METALS for z in z_corr.tolist()],
                        dtype=torch.bool)
                    logits, _ = model(g.to(device), metal_mask.to(device))
                else:
                    logits = model(g.to(device),
                                   use_elem_cond=not args.no_elem_cond,
                                   use_dist_bias=not args.no_dist_bias)
            pred_idx = logits.argmax(dim=-1).cpu()    # [N]

            # Apply predictions: only update where model predicts a different element
            z_repaired = z_corr.clone()
            for i in range(len(z_corr)):
                if z_corr[i].item() in ELEM_VOCAB_SET:
                    pred_z = ELEM_VOCAB[pred_idx[i].item()]
                    if pred_z != z_corr[i].item():
                        z_repaired[i] = pred_z

            # Site accuracy
            for site in sites:
                site_total += 1
                if z_repaired[site].item() == z_gt[site].item():
                    site_correct += 1

            # ΔS_topo
            try:
                s_before = topo_verifier.score(pos.numpy(), z_corr.numpy())
                s_after  = topo_verifier.score(pos.numpy(), z_repaired.numpy())
                ds = s_after - s_before
            except Exception as e:
                print(f"delta_S exception: {e}")
                ds = 0.0

            delta_s_list.append(ds)
            success_list.append(1 if ds > 0.5 else 0)
            bond_f1_list.append(bond_f1(z_repaired, z_gt, pos, MULTIPLIER))

    n = len(delta_s_list)
    return {
        'split':    split_name,
        'n':        n,
        'site_acc': site_correct / site_total if site_total > 0 else 0.0,
        'delta_s':  float(np.mean(delta_s_list)),
        'bond_f1':  float(np.mean(bond_f1_list)),
        'success':  float(np.mean(success_list)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',         default='all',
                        choices=['single_type_a', 'single_type_b', 'multi_site', 'all'])
    parser.add_argument('--ckpt_dir',      default='ase_checkpoints_iterative_refiner',
                        help='Checkpoint directory to load from')
    parser.add_argument('--n_mols',        type=int, default=None)
    parser.add_argument('--device',        default='cuda')
    parser.add_argument('--output',        default='results/iterative_refiner_eval')
    parser.add_argument('--no_elem_cond',  action='store_true', help='Disable soft element conditioning')
    parser.add_argument('--no_dist_bias',  action='store_true', help='Disable distance bias (W_b)')
    parser.add_argument('--v2',            action='store_true', help='Use IterativeRefinerV2 (tmQM fine-tuned)')
    parser.add_argument('--v3',            action='store_true', help='Use IterativeRefinerV3 (center conditioned atten)')
    args = parser.parse_args()

    if args.v3:
        refiner_ckpt = 'ase_checkpoints_refiner_v3_k5/checkpoint_best.pt'
    elif args.v2:
        refiner_ckpt = 'ase_checkpoints_refiner_v2/checkpoint_best.pt'
    else:
        refiner_ckpt = f'{args.ckpt_dir}/checkpoint_best.pt'

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {'V3' if args.v3 else 'V2' if args.v2 else 'V1'} from {refiner_ckpt}...")
    ckpt  = torch.load(refiner_ckpt, map_location='cpu')
    K     = ckpt['config'].get('K', 3)
    if args.v3:
        model = IterativeRefinerV3(METALLIC_CKPT, K=K).to(device)
    elif args.v2:
        model = IterativeRefinerV2(METALLIC_CKPT, K=K).to(device)
    else:
        print(f"  use_elem_cond={not args.no_elem_cond}  use_dist_bias={not args.no_dist_bias}")
        model = IterativeRefiner(METALLIC_CKPT, K=K).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Loading TopoVerifier...")
    topo_verifier = TopoVerifier(checkpoint_path=METALLIC_CKPT,
                                 ssl_checkpoint_path='ase_checkpoints_tuned/checkpoint_best.pt',
                                 bond_config_path='results/graph_calibrate/bond_config.json',
                                 device=device)

    splits = ['single_type_a', 'single_type_b', 'multi_site'] \
             if args.split == 'all' else [args.split]

    all_results = []
    for split in splits:
        r = evaluate_split(split, model, topo_verifier, device, args.n_mols, use_v2=args.v2 or args.v3)
        all_results.append(r)

    # Save
    torch.save(all_results, f"{args.output}/results.pt")

    # Print table
    print("\n" + "="*72)
    print("COMBINED RESULTS")
    print("="*72)
    print(f"{'Split':<22} {'SiteAcc':>9} {'ΔS_topo':>9} {'BondF1':>9} {'Success':>9}")
    print("-"*72)
    for r in all_results:
        print(f"{r['split']:<22} {r['site_acc']*100:>8.1f}%"
              f" {r['delta_s']:>+9.4f} {r['bond_f1']*100:>8.1f}%"
              f" {r['success']*100:>8.1f}%")
    print("="*72)
