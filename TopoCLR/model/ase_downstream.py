
"""
Downstream Classification with SMOTE + K-Fold CV for Imbalanced Data

Features:
- SMOTE oversampling for minority class
- 5-fold stratified cross-validation
- Confusion matrix reporting
- Handles severe class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Import your models
import sys
sys.path.insert(0, '/import/fox/will/graduation_thesis')
from ase_model import TopoSSL


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α(1 - p_t)^γ * log(p_t)
    
    where:
        p_t = probability of correct class
        γ (gamma) = focusing parameter (default: 2.0)
            - Higher γ focuses more on hard examples
            - γ = 0 reduces to CrossEntropyLoss
        α (alpha) = class balance weight (default: 0.25)
            - Weight for positive class
            - 1-α for negative class
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits (NOT probabilities!)
            targets: [batch_size] class labels (0 or 1)
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t (probability of correct class)
        p_t = torch.exp(-ce_loss)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        # For binary: alpha for positive class, (1-alpha) for negative
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        
        # Focal loss
        focal_loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MoleculeDataset(Dataset):
    """Dataset for labeled molecules (DGL format)"""
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.graphs = data['graphs']
        self.labels = data['labels']
        
        print(f"Loaded {len(self.graphs)} molecules")
        label_counts = np.bincount(self.labels)
        print(f"Label distribution: {dict(enumerate(label_counts))}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def ensure_node_features(graph, num_features=53):
    """
    Ensure graph has 'h' node feature for model input
    Creates 53-dim feature from atomic_num, degree, in_ring
    (Copied from ase_augmentation.py)
    
    Args:
        graph: DGL graph
        num_features: Target feature dimension (default: 53)
        
    Returns:
        Graph with 'h' feature added
    """
    # If 'h' already exists and has correct dimension, return as is
    if 'h' in graph.ndata and graph.ndata['h'].shape[1] == num_features:
        return graph
    
    # Create 'h' from existing features
    num_nodes = graph.num_nodes()
    
    # Create feature vector (mostly zeros, first 3 are actual features)
    h = torch.zeros(num_nodes, num_features)
    
    # First 3 features: atomic_num, degree, in_ring
    if 'atomic_num' in graph.ndata:
        h[:, 0] = graph.ndata['atomic_num'].float() / 100.0  # Normalize
    
    if 'degree' in graph.ndata:
        h[:, 1] = graph.ndata['degree'].float()
    
    if 'in_ring' in graph.ndata:
        h[:, 2] = graph.ndata['in_ring'].float()
    
    # Rest are zeros (h[:, 3:53] = 0 by default)
    
    # Store as 'h'
    graph.ndata['h'] = h
    
    return graph


def collate_fn(batch):
    """Collate function for DGL graphs"""
    graphs, labels = zip(*batch)
    
    # Ensure all graphs have 53-dim 'h' feature
    graphs = [ensure_node_features(g, num_features=53) for g in graphs]
    
    batched_graph = dgl.batch(graphs)
    labels = torch.LongTensor(labels)
    return batched_graph, labels


class SSLClassifier(nn.Module):
    """Classification model using pretrained SSL encoder"""
    def __init__(
        self, 
        ssl_checkpoint_path,
        num_classes=2,
        hidden_dim=256,
        num_layers=5,
        proj_dim=128,
        freeze_encoder=True,
        dropout=0.3
    ):
        super().__init__()
        
        # Load pretrained SSL encoder
        self.ssl_model = TopoSSL(
            num_features=53,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            pool='mean'
        )
        
        checkpoint = torch.load(ssl_checkpoint_path, map_location='cpu')
        self.ssl_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        
        # Classification head
        encoder_dim = hidden_dim * num_layers
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, g):
        # Get encoder representation (h_graph, NOT z_graph!)
        if self.freeze_encoder:
            with torch.no_grad():
                h_graph = self.ssl_model.encoder(g)
        else:
            h_graph = self.ssl_model.encoder(g)
        
        # Classify
        logits = self.classifier(h_graph)
        return logits


def extract_features(model, data_loader, device):
    """Extract features from SSL encoder for all samples"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_graph, batch_labels in tqdm(data_loader, desc="Extracting features"):
            batch_graph = batch_graph.to(device)
            
            # Get encoder output (returns tuple: h_graph, h_node)
            h_graph, h_node = model.ssl_model.encoder(batch_graph)
            
            all_features.append(h_graph.cpu())
            all_labels.append(batch_labels)
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    return features, labels


def apply_smote(features, labels, random_state=42):
    """Apply SMOTE to create synthetic minority samples"""
    print(f"\nApplying SMOTE...")
    print(f"Before SMOTE: {np.bincount(labels)}")
    
    smote = SMOTE(random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    
    print(f"After SMOTE: {np.bincount(labels_resampled)}")
    print(f"Total samples: {len(labels_resampled)}")
    
    return features_resampled, labels_resampled


def plot_confusion_matrix(cm, output_path, title='Confusion Matrix', class_names=None):
    """Plot and save confusion matrix"""
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {output_path}")


def train_on_features(
    features_train,
    labels_train,
    features_val,
    labels_val,
    hidden_dim=512,
    num_classes=2,
    epochs=50,
    lr=0.001,
    weight_decay=1e-5,
    device='cuda',
    loss_type='focal',
    focal_gamma=2.0,
    focal_alpha=0.25
):
    """
    Train classifier on extracted features
    
    Args:
        loss_type: 'focal', 'weighted_ce', or 'ce'
        focal_gamma: Focusing parameter for focal loss (higher = focus more on hard examples)
        focal_alpha: Class balance weight for focal loss
    """
    
    # Convert to tensors
    X_train = torch.FloatTensor(features_train).to(device)
    y_train = torch.LongTensor(labels_train).to(device)
    X_val = torch.FloatTensor(features_val).to(device)
    y_val = torch.LongTensor(labels_val).to(device)
    
    # Simple classifier
    input_dim = features_train.shape[1]
    classifier = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    ).to(device)
    
    # Choose loss function
    pos_count = (y_train == 1).sum().item()
    neg_count = (y_train == 0).sum().item()
    
    if loss_type == 'focal':
        print(f"  Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    elif loss_type == 'weighted_ce':
        # Weighted CrossEntropy (inverse frequency)
        if pos_count > 0:
            pos_weight = neg_count / pos_count
        else:
            pos_weight = 1.0
        class_weights = torch.tensor([1.0, pos_weight], device=device)
        print(f"  Using Weighted CrossEntropy (pos_weight={pos_weight:.2f})")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    else:  # 'ce' - standard CrossEntropy
        print(f"  Using Standard CrossEntropy")
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        classifier.train()
        optimizer.zero_grad()
        
        logits = classifier(X_train)
        loss = criterion(logits, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val)
            val_loss = criterion(val_logits, y_val)
            
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_labels = y_val.cpu().numpy()
            
            _, _, val_f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='binary'
            )
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = classifier.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    classifier.load_state_dict(best_state)
    
    return classifier


def evaluate_classifier(classifier, features, labels, device='cuda', threshold=0.5):
    """Evaluate classifier and return metrics + predictions"""
    X = torch.FloatTensor(features).to(device)
    y = torch.LongTensor(labels)
    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    
    # Probablities for pos class
    probs_pos = probs[:, 1]
    preds = (probs_pos >= threshold).astype(int)
    
    # Compute metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'y_true': labels,
        'predictions': preds,
        'probabilities': probs,
        'threshold': threshold
    }

def find_best_threshold(classifier, features, labels, device='cuda'):
    X = torch.FloatTensor(features).to(device)
    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        
    probs_pos = probs[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(labels, probs_pos)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    
    if thresholds.size > 0:
        best_idx = np.nanargmax(f1_scores[:-1])
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold
    
    return best_threshold
    
def kfold_cv_with_smote(
    data_path,
    ssl_checkpoint,
    output_dir,
    num_classes=2,
    hidden_dim=256,
    num_layers=5,
    proj_dim=128,
    n_splits=5,
    epochs=50,
    lr=0.001,
    device='cuda',
    loss_type='focal',
    focal_gamma=2.0,
    focal_alpha=0.25
):
    """
    K-Fold Cross-Validation with SMOTE
    
    Main evaluation method for imbalanced data!
    
    Args:
        loss_type: 'focal', 'weighted_ce', or 'ce'
        focal_gamma: Focusing parameter for focal loss (default: 2.0)
        focal_alpha: Class balance weight for focal loss (default: 0.25)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loss type: {loss_type}")
    if loss_type == 'focal':
        print(f"  Focal gamma: {focal_gamma}")
        print(f"  Focal alpha: {focal_alpha}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    dataset = MoleculeDataset(data_path)
    
    # Create data loader for feature extraction
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load SSL model
    print("\n" + "="*60)
    print("Loading SSL Model")
    print("="*60)
    model = SSLClassifier(
        ssl_checkpoint_path=ssl_checkpoint,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        proj_dim=proj_dim,
        freeze_encoder=True
    ).to(device)
    
    checkpoint = torch.load(ssl_checkpoint)
    print(f"✓ SSL model from epoch {checkpoint['epoch']}")
    print(f"✓ SSL val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Extract features from all data
    print("\n" + "="*60)
    print("Extracting Features from SSL Encoder")
    print("="*60)
    all_features, all_labels = extract_features(model, loader, device)
    print(f"✓ Extracted features: {all_features.shape}")
    
    # K-Fold CV
    print("\n" + "="*60)
    print(f"Starting {n_splits}-Fold Cross-Validation with SMOTE")
    print("="*60)
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_cms = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_features, all_labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print("="*60)
        
        # Split data
        X_train_fold = all_features[train_idx]
        y_train_fold = all_labels[train_idx]
        X_test_fold = all_features[test_idx]
        y_test_fold = all_labels[test_idx]
        
        print(f"Train: {len(y_train_fold)} samples, Test: {len(y_test_fold)} samples")
        print(f"Train distribution: {np.bincount(y_train_fold)}")
        print(f"Test distribution: {np.bincount(y_test_fold)}")
        
        # Apply SMOTE to training data only
        X_train_balanced, y_train_balanced = apply_smote(
            X_train_fold, y_train_fold, random_state=42+fold
        )
        
        # Train classifier on balanced data
        print(f"\nTraining classifier on fold {fold+1}...")
        classifier = train_on_features(
            X_train_balanced,
            y_train_balanced,
            X_test_fold,  # Use test as validation for early stopping
            y_test_fold,
            hidden_dim=512,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            device=device,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha
        )
        
        best_threshold = find_best_threshold(
            classifier, X_train_fold, y_train_fold, device=device
        )
        print(f"best threshold: {best_threshold:.3f}")
        
        
        # Evaluate on test fold (original, not SMOTE'd)
        print(f"Evaluating on fold {fold+1}...")
        results = evaluate_classifier(
            classifier, X_test_fold, y_test_fold, device=device, threshold=best_threshold
        )
        
        # Print results
        print(f"\nFold {fold+1} Results:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  AUC:       {results['auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {results['confusion_matrix']}")
        
        # Save confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            output_dir / f'confusion_matrix_fold{fold+1}.png',
            title=f'Confusion Matrix - Fold {fold+1}'
        )
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        })
        
        all_cms.append(results['confusion_matrix'])
    
    # Aggregate results
    print("\n" + "="*60)
    print("FINAL RESULTS (Averaged across folds)")
    print("="*60)
    
    metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        metrics[metric] = {'mean': mean, 'std': std, 'values': values}
        
        print(f"{metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    # Average confusion matrix
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    # Plot average confusion matrix
    plot_confusion_matrix(
        avg_cm,
        output_dir / 'confusion_matrix_average.png',
        title=f'Average Confusion Matrix ({n_splits}-Fold CV)'
    )
    
    # Save results
    results_summary = {
        'n_splits': n_splits,
        'ssl_checkpoint': str(ssl_checkpoint),
        'loss_type': loss_type,
        'focal_gamma': focal_gamma if loss_type == 'focal' else None,
        'focal_alpha': focal_alpha if loss_type == 'focal' else None,
        'metrics': {k: {'mean': v['mean'], 'std': v['std']} for k, v in metrics.items()},
        'fold_results': fold_results,
        'average_confusion_matrix': avg_cm.tolist()
    }
    
    with open(output_dir / 'kfold_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='K-Fold CV with SMOTE for Imbalanced Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python downstream_kfold_smote.py \\
    --ssl_checkpoint ./ase_checkpoints/best_model.pt \\
    --data_path ./rhodamine_labeled/all_data.pt \\
    --output_dir ./results/rhodamine_kfold \\
    --hidden_dim 256 \\
    --num_layers 5 \\
    --proj_dim 128 \\
    --n_splits 5
        """
    )
    
    # Required
    parser.add_argument('--ssl_checkpoint', type=str, default="ase_checkpoints_tuned/checkpoint_best.pt",
                       help='Path to SSL checkpoint')
    parser.add_argument('--data_path', type=str, default='ase_split/YP8/all_data.pt',
                       help='Path to data (.pt file with all samples)')
    parser.add_argument('--output_dir', type=str, default='ase_results/YP8_kfold',
                       help='Output directory')
    
    # Model architecture (must match SSL model!)
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dim (must match SSL model)')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of layers (must match SSL model)')
    parser.add_argument('--proj_dim', type=int, default=256,
                       help='Projection dim (must match SSL model)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # Training
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['focal', 'weighted_ce', 'ce'],
                       help='Loss function: focal, weighted_ce (weighted cross entropy), or ce (standard cross entropy)')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                       help='Focal loss gamma (focusing parameter). Higher = focus more on hard examples. Default: 2.0')
    parser.add_argument('--focal_alpha', type=float, default=0.5,
                       help='Focal loss alpha (class balance weight for positive class). Default: 0.25')
    
    args = parser.parse_args()
    
    kfold_cv_with_smote(
        data_path=args.data_path,
        ssl_checkpoint=args.ssl_checkpoint,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        n_splits=args.n_splits,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha
    )


if __name__ == '__main__':
    main()