# Iterative Refiner

**Coordination-Aware Attention for 3D Electron Diffraction Structural Repair**

## Repository Overview

This repository contains the implementation of **Iterative Refiner**, a coordination-aware molecular structure repair framework for 3D Electron Diffraction (3D ED) structural analysis.

The repository is organized into the following main components.

### `TopoCLR/`

`TopoCLR` is the upstream representation learning component of this research and serves as the preliminary stage for the Iterative Refiner. It is designed to learn topology-aware molecular graph representations, which are then used to support downstream molecular structure refinement.

Please execute the TopoCLR workflow first by following the instructions in `TopoCLR/README.md`.

### `Baselines/`

This directory contains baseline methods used to compare against the Iterative Refiner on the CSD benchmark.

## Execution Order

### 1. Pretrain the TopoCLR model

Follow the instructions provided in the `README.md` file inside the `TopoCLR/` directory.

### 2. Process the pretraining data for the Iterative Refiner

Run the following script to preprocess the tmQM dataset containing 75,680 metallic molecules:

```bash
python ase_prepare_tmqm_data.py
```

### 3. Prepare the benchmark dataset

Run the following script to preprocess the CSD benchmark dataset containing 12,000 metallic molecules:

```bash
python ase_prepare_benchmark.py
```

### 4. Pretrain the Iterative Refiner

Execute the latest Iterative Refiner model:

```bash
python ase_train_refiner_v3.py
```

This version uses dynamic attention and achieves the best performance among the current implementations on the CSD benchmark.

### 5. Evaluate the model

Run the following script to evaluate the performance of the Iterative Refiner:

```bash
python ase_eval_iterative_refiner.py --split all --v3
```

The `--v3` flag activates the current best model version.
