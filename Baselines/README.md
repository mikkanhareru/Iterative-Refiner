# Baselines

This directory contains baseline methods used to compare against the Iterative Refiner on the CSD benchmark.

The included baseline workflows are:

- **Simulated annealing (SA)**: train and evaluate the simulated annealing baseline with `ase_sa_update` and `ase_sa_eval`.
- **Pruned exhaustive search**: run the joint-search baseline with `joint_search`.

## How to examine the baselines

Run the baseline scripts in the following order.

### 1. Train the simulated annealing baseline

First, run `ase_sa_update` to train or update the simulated annealing baseline. This step can be run either with CEP mode enabled or without CEP mode, depending on the comparison setting you want to examine.

```bash
python ase_sa_update.py
```

If you also like to examine the performance of simulated annealing without using CEP, just set `--beta_cep=0`. For example:

```bash
python ase_sa_update.py --beta_cep=0
```

### 2. Evaluate simulated annealing on the benchmark

After training or updating the SA baseline, run `ase_sa_eval` to evaluate simulated annealing performance on the CSD benchmark.

```bash
python ase_sa_eval.py
```

### 3. Run the pruned exhaustive-search baseline

Lastly, run `joint_search` to evaluate the pruned exhaustive-search baseline.

```bash
python joint_search.py
```

## Notes

- Use the same CSD benchmark split and evaluation settings when comparing these baselines with the Iterative Refiner.
- Run SA experiments both with and without CEP mode if you want to compare the effect of CEP directly.
- Save the outputs from `ase_sa_eval` and `joint_search` so they can be compared with the Iterative Refiner benchmark results.
