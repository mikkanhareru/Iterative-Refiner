# Baselines

This directory contains baseline methods used to compare against the Iterative Refiner on the CSD benchmark.

The included baseline workflows are:

- **Simulated annealing (SA)**: train and evaluate the simulated annealing baseline with `ase_sa_update.py` and `ase_sa_eval.py`.
- **Pruned exhaustive search**: run the joint-search baseline with `joint_search.py`.
- **Iterative refiner with static attention**: train and evaluate the iterative refiner with static attention with `ase_train_refiner_v2.py` and `ase_eval_iterative_refiner.py` located in the `main` directory.

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

Run `joint_search` to evaluate the pruned exhaustive-search baseline.

```bash
python joint_search.py
```

### 4. If you want to examine the ablation study of the Iterative Refiner
First, train the Iterative Refiner with static attention by running `ase_train_refiner_v2.py`. 
```bash
python ase_train_itertaive_refiner_v2.py 
```
Next, evaluate the model by running 
```bash
python ase_eval_iterative_refiner.py --split all --v2
```

## Notes

- Use the same CSD benchmark split and evaluation settings when comparing these baselines with the Iterative Refiner.
- Run SA experiments both with and without CEP mode if you want to compare the effect of CEP directly.
- Save the outputs from `ase_sa_eval` and `joint_search` so they can be compared with the Iterative Refiner benchmark results.
