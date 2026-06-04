TopoCLR serves as the preliminary stage of the research. It learns molecular topology representations and provides the foundation for the subsequent structure refinement process. Therefore, users should first run the scripts inside this directory before executing the Iterative Refiner scripts in the main directory.

This repository provides scripts for preprocessing molecular structures, pretraining the TopoCLR, and evaluating the pretrained model on downstream molecular-topology or performance tasks.

## Data availability and licensing

Raw molecular structure files are not included when redistribution is restricted by the original data license.
Users should obtain the original datasets from their official sources and run the provided preprocessing scripts.

## Supported input structures

The preprocessing workflow supports molecular structure files in `.pdb` format. Place the `.pdb` files for your dataset in an input directory before running the preprocessing step.

## Execution workflow

Run the scripts in the following order.

### 1. Preprocess your data

First, convert the raw `.pdb` molecular structures into graph tensors with the ASE/DGL preprocessing script:

```bash
python TopoCLR/Data_Preprocessing/ase_batch_process.py \
  --input /path/to/pdb_files \
  --output /path/to/processed_graphs \
  --batch_size 10000 \
  --verify
```

Arguments:

- `--input`: directory containing raw `.pdb` structure files.
- `--output`: directory where processed `.pt` graph files will be written.
- `--batch_size`: number of molecules saved per output `.pt` file.
- `--n_workers`: number of parallel workers; defaults to `0`.
- `--verify`: optionally inspect the processed output after preprocessing.
- `--verbose`: print detailed processing messages.

> Note: The preprocessing script in this repository is named `ase_batch_process.py`. Use this script for the preprocessing step before training.

### 2. Pretrain the model

Second, pretrain the TopoCLR model with the processed graph data:

```bash
python TopoCLR/model/ase_train.py \
  --data_path /path/to/split_directory \
  --output_dir /path/to/pretrain_checkpoints \
  --batch_size 128 \
  --epochs 100 \
  --use_different_graphs
```

The `--data_path` directory is expected to contain the training split files used by the training script, such as `train.pt`, `val.pt`, and `test.pt`. The script writes checkpoints and logs to `--output_dir`.

Common options include:

- `--hidden_dim`, `--num_layers`, `--proj_dim`: model architecture settings.
- `--lr`, `--weight_decay`, `--optimizer`, `--scheduler`: optimization settings.
- `--node_drop_rate`, `--edge_drop_rate`, `--feature_mask_rate`, `--subgraph_sample_rate`: graph augmentation settings.
- `--early_stopping --patience <N>`: enable early stopping.
- `--resume /path/to/checkpoint.pt`: resume from a previous checkpoint.

### 3. Test downstream model performance

Lastly, evaluate model performance on labeled downstream data:

```bash
python TopoCLR/model/ase_downstream.py \
  --ssl_checkpoint /path/to/pretrain_checkpoints/checkpoint_best.pt \
  --data_path /path/to/downstream/all_data.pt \
  --output_dir /path/to/downstream_results \
  --n_splits 5 \
  --epochs 50 \
  --device cuda
```

Arguments:

- `--ssl_checkpoint`: pretrained checkpoint produced by `ase_train.py`.
- `--data_path`: labeled downstream `.pt` file containing all samples.
- `--output_dir`: directory where downstream metrics, plots, and results will be saved.
- `--n_splits`: number of stratified cross-validation folds.
- `--loss_type`: downstream classifier loss; choose from `focal`, `weighted_ce`, or `ce`.
- `--device`: use `cuda` when a GPU is available, otherwise use `cpu`.

Make sure `--hidden_dim`, `--num_layers`, and `--proj_dim` match the architecture used during pretraining.

## Minimal command summary

```bash
# 1. Preprocess raw .pdb structures
python TopoCLR/Data_Preprocessing/ase_batch_process.py --input data/pdb --output data/processed --verify

# 2. Pretrain the self-supervised model
python TopoCLR/model/ase_train.py --data_path data/splits/split_01 --output_dir checkpoints/pretrain --use_different_graphs

# 3. Evaluate downstream performance
python TopoCLR/model/ase_downstream.py --ssl_checkpoint checkpoints/pretrain/checkpoint_best.pt --data_path data/downstream/all_data.pt --output_dir results/downstream
```
