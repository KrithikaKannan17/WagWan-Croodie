"""
Training script for self-play generated data with policy and value targets.
Handles new format: (boards, policy_targets, value_targets)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

from src.utils.model import ChessModel


class SelfPlayDataset(Dataset):
    """Dataset for self-play training data with policy and value targets."""
    
    def __init__(self, boards, policy_targets, value_targets):
        """
        Args:
            boards: (N, 12, 8, 8) board tensors
            policy_targets: (N, 256) policy distributions
            value_targets: (N,) game outcomes
        """
        self.boards = torch.from_numpy(boards).float()
        self.policy_targets = torch.from_numpy(policy_targets).float()
        self.value_targets = torch.from_numpy(value_targets).float()
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        return self.boards[idx], self.policy_targets[idx], self.value_targets[idx]


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    for boards, policy_targets, value_targets in train_loader:
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        
        optimizer.zero_grad()
        
        policy_logits, value_pred = model(boards)
        
        # Policy loss: cross-entropy with soft targets
        policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        
        # Value loss: MSE
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)
        
        # Combined loss
        loss = policy_loss + value_loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
    
    num_batches = len(train_loader)
    return {
        'total': total_loss / num_batches,
        'policy': policy_loss_sum / num_batches,
        'value': value_loss_sum / num_batches
    }


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    with torch.no_grad():
        for boards, policy_targets, value_targets in val_loader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            policy_logits, value_pred = model(boards)
            
            policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_targets)
            
            loss = policy_loss + value_loss
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
    
    num_batches = len(val_loader)
    return {
        'total': total_loss / num_batches,
        'policy': policy_loss_sum / num_batches,
        'value': value_loss_sum / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train on self-play data')
    parser.add_argument('--data', type=str, default='training_data_merged.npz',
                       help='Training data file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--num-residual-blocks', type=int, default=6,
                       help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=64,
                       help='Number of channels')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--output', type=str, default='chess_model_selfplay.pth',
                       help='Output model file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 SELF-PLAY TRAINING - AlphaZero Style")
    print("="*70)
    print(f"✓ Dropout: {args.dropout}")
    print(f"✓ Learning Rate Scheduling: Enabled")
    print(f"✓ Early Stopping: {args.patience} epochs")
    print(f"✓ Gradient Clipping: Enabled")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nLoading training data from {args.data}...")
    data = np.load(args.data)
    boards = data['boards']
    policy_targets = data['policy_targets']
    value_targets = data['value_targets']
    print(f"Loaded {len(boards)} positions")

    # Create dataset
    dataset = SelfPlayDataset(boards, policy_targets, value_targets)

    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create or load model
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model = ChessModel(
            num_residual_blocks=checkpoint.get('num_residual_blocks', args.num_residual_blocks),
            channels=checkpoint.get('channels', args.channels),
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
    else:
        model = ChessModel(
            num_residual_blocks=args.num_residual_blocks,
            channels=args.channels,
            dropout=args.dropout
        ).to(device)
        start_epoch = 0
        best_val_loss = float('inf')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")
    print(f"Architecture: {args.num_residual_blocks} blocks, {args.channels} channels")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training for up to {args.epochs} epochs...")
    print(f"{'='*70}\n")

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update LR
        scheduler.step(val_metrics['total'])

        epoch_time = time.time() - epoch_start

        # Print
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Policy: {train_metrics['policy']:.4f}, "
              f"Value: {train_metrics['value']:.4f}, Total: {train_metrics['total']:.4f}")
        print(f"  Val   - Policy: {val_metrics['policy']:.4f}, "
              f"Value: {val_metrics['value']:.4f}, Total: {val_metrics['total']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_metrics['total'],
                'model_type': 'ChessModel',
                'num_residual_blocks': args.num_residual_blocks,
                'channels': args.channels,
                'dropout': args.dropout
            }, args.output)

            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n{'='*70}")
                print(f"⚠ Early stopping after {epoch+1} epochs")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"{'='*70}")
                break

        print()

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"  Best model: {args.output}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


