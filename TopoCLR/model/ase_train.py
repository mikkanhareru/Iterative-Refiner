'''
Molecular Topology SSL training 
'''
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import dgl
from tqdm import tqdm

from ase_model import TopoSSL, LocalGlobalDiscriminator
from ase_losses import compute_ssl_loss
from ase_augmentation import MolecularAugmentation, ssl_collate_fn

warnings.filterwarnings('ignore')

class MolecularDataset(Dataset):
    def __init__(self, data_path: str, split:str='train'):
        self.data_path = Path(data_path)
        self.split = split

        # Load data
        split_file = self.data_path / f'{split}.pt'
        if not split_file.exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        
        data = torch.load(split_file)
        self.graphs = data['graphs']

        print(f"Loaded {len(self.graphs)} molecules from {split} split")

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, discriminator, config):
    params = list(model.parameters()) + list(discriminator.parameters())

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params,
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params, 
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"unknown optimizer: {config['optimizer']}")
    
    return optimizer

def get_scheduler(optimizer, config):
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['lr_min']
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )
    elif config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None

    return scheduler

def train_epoch(
    model: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary of averaged losses
    """
    model.train()
    discriminator.train()
    
    total_losses = {
        'total': 0.0,
        'contrastive': 0.0,
        'mi': 0.0,
        'topology': 0.0,
        'ring': 0.0,
        'degree': 0.0,
        'distance': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (view1, view2, batch_indices) in enumerate(pbar):
        # Move to device
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model.forward_contrast(view1, view2)
        
        # Compute loss
        loss, loss_dict = compute_ssl_loss(
            outputs,
            discriminator,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            temperature=config['temperature']
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(discriminator.parameters()),
                config['grad_clip']
            )
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cont': f"{loss_dict['contrastive']:.4f}",
            'mi': f"{loss_dict['mi']:.4f}",
            'topo': f"{loss_dict['topology']:.4f}"
        })
    
    # Average losses
    num_batches = len(train_loader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    discriminator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict
) -> Dict[str, float]:
    """
    Validate the model
    
    Returns:
        Dictionary of averaged losses
    """
    model.eval()
    discriminator.eval()
    
    total_losses = {
        'total': 0.0,
        'contrastive': 0.0,
        'mi': 0.0,
        'topology': 0.0,
        'ring': 0.0,
        'degree': 0.0,
        'distance': 0.0
    }
    
    for view1, view2, batch_indices in tqdm(val_loader, desc="Validating"):
        # Move to device
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        outputs = model.forward_contrast(view1, view2)
        
        # Compute loss
        loss, loss_dict = compute_ssl_loss(
            outputs,
            discriminator,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            temperature=config['temperature']
        )
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
    
    # Average losses
    num_batches = len(val_loader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    discriminator: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_loss: float,
    config: Dict,
    save_path: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, save_path / 'checkpoint_latest.pt')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, save_path / 'checkpoint_best.pt')
    
    # Save periodic checkpoint
    if epoch % config['save_freq'] == 0:
        torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pt')


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    discriminator: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, float]:
    """
    Load model checkpoint
    
    Returns:
        start_epoch, best_val_loss
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    return start_epoch, best_val_loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(config: Dict):
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(config['seed'])
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_dataset = MolecularDataset(config['data_path'], split='train')
    val_dataset = MolecularDataset(config['data_path'], split='val')
    
    # Create augmentor
    augmentor = MolecularAugmentation(
        node_drop_rate=config['node_drop_rate'],
        edge_drop_rate=config['edge_drop_rate'],
        feature_mask_rate=config['feature_mask_rate'],
        subgraph_sample_rate=config['subgraph_sample_rate'],
        apply_noise_coords=config['apply_noise_coords'],
        noise_scale=config['noise_scale'],
        seed=config['seed']
    )
    
    # Create data loaders
    from functools import partial
    train_collate = partial(
        ssl_collate_fn,
        augmentor=augmentor,
        n_views=2,
        use_different_graphs=config['use_different_graphs']
    )
    val_collate = partial(
        ssl_collate_fn,
        augmentor=augmentor,
        n_views=2,
        use_different_graphs=config['use_different_graphs']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=train_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=val_collate,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = TopoSSL(
        num_features=config['num_features'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        pool=config['pool']
    )
    
    discriminator = LocalGlobalDiscriminator(
        hidden_dim=config['hidden_dim'] * config['num_layers']
    )

    model.apply(init_weights)
    discriminator.apply(init_weights)

    model = model.to(device)
    discriminator = discriminator.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Discriminator parameters: {num_disc_params:,}")
    print(f"Total parameters: {num_params + num_disc_params:,}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, discriminator, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if config['resume']:
        checkpoint_path = Path(config['resume'])
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            # Skip optimizer/scheduler loading — Phase 1 has different param groups
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, discriminator, None, None
            )
            start_epoch = 1          # restart epoch count for Phase 2
            best_val_loss = float('inf')
            print(f"Loaded model weights from Phase 1 (fresh optimizer for Phase 2)")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_losses = train_epoch(
            model, discriminator, train_loader,
            optimizer, device, config, epoch
        )
        
        # Validate
        val_losses = validate(
            model, discriminator, val_loader,
            device, config
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(Cont: {train_losses['contrastive']:.4f}, "
              f"MI: {train_losses['mi']:.4f}, "
              f"Topo: {train_losses['topology']:.4f})")
        print(f"  Val Loss:   {val_losses['total']:.4f} "
              f"(Cont: {val_losses['contrastive']:.4f}, "
              f"MI: {val_losses['mi']:.4f}, "
              f"Topo: {val_losses['topology']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['lr'].append(current_lr)
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Check if best model
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            patience_counter = 0
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Save checkpoint
        save_checkpoint(
            model, discriminator, optimizer, scheduler,
            epoch, best_val_loss, config, output_dir, is_best
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses['total'])
            else:
                scheduler.step()
        
        # Early stopping
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Molecular Topology SSL Model"
    )
    
    # Data paths
    parser.add_argument('--data_path', type=str, default="ase_split/split_01",
                        help='Path to directory containing train.pt, val.pt, test.pt')
    parser.add_argument('--output_dir', type=str, default='./ase_checkpoints_tuned_benchmark',
                        help='Output directory for checkpoints and logs')
    
    # Model architecture
    parser.add_argument('--num_features', type=int, default=53,
                        help='Number of input node features')
    parser.add_argument('--hidden_dim', type=int, default=512, #256->512
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GNN layers')
    parser.add_argument('--proj_dim', type=int, default=256, #128->256
                        help='Projection dimension for contrastive learning')
    parser.add_argument('--pool', type=str, default='mean',
                        choices=['mean', 'add'],
                        help='Graph pooling method')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help='Gradient clipping (0 to disable)')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Step size for step scheduler')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='Decay rate for step scheduler')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for MI loss')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Weight for topology loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive learning')
    
    # Augmentation
    parser.add_argument('--node_drop_rate', type=float, default=0.25,
                        help='Node dropping rate')
    parser.add_argument('--edge_drop_rate', type=float, default=0.25,
                        help='Edge dropping rate')
    parser.add_argument('--feature_mask_rate', type=float, default=0.25,
                        help='Feature masking rate')
    parser.add_argument('--subgraph_sample_rate', type=float, default=0.7,
                        help='Subgraph sampling rate')
    parser.add_argument('--apply_noise_coords', action='store_true',
                        help='Apply Gaussian noise to RBF features')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='Scale of Gaussian noise')
    parser.add_argument('--use_different_graphs', action='store_true',
                        help='Use outline + geometry as two views')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Save checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Start training
    train(config)


if __name__ == '__main__':
    main()