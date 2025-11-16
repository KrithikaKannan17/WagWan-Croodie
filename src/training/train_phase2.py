"""
Local training script for self-play data (boards, policies, values format).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import argparse

from src.utils.model import ChessModel


class ChessDataset(Dataset):
    """Dataset for chess training data."""
    
    def __init__(self, boards, policies, values):
        self.boards = torch.FloatTensor(boards)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx]


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    for boards, policies, values in dataloader:
        boards = boards.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        optimizer.zero_grad()
        
        policy_pred, value_pred = model(boards)
        
        policy_loss = policy_criterion(policy_pred, policies)
        value_loss = value_criterion(value_pred.squeeze(), values)
        
        loss = policy_loss + value_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / len(dataloader), total_value_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_policy_loss = 0
    total_value_loss = 0
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for boards, policies, values in dataloader:
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            policy_pred, value_pred = model(boards)
            
            policy_loss = policy_criterion(policy_pred, policies)
            value_loss = value_criterion(value_pred.squeeze(), values)
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    return total_policy_loss / len(dataloader), total_value_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train chess model locally")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz data file")
    parser.add_argument("--output", type=str, required=True, help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-input", type=str, default=None, help="Input model to resume from")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 Local Chess Model Training")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data)
    boards = data['boards']
    policies = data['policies']
    values = data['values']
    
    print(f"✓ Loaded {len(boards)} positions")
    print(f"  Boards: {boards.shape}")
    print(f"  Policies: {policies.shape}")
    print(f"  Values: {values.shape}")
    
    # Create dataset
    dataset = ChessDataset(boards, policies, values)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"\n✓ Train: {train_size}, Val: {val_size}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = ChessModel(num_residual_blocks=6, channels=64, dropout=0.0)
    
    # Load existing model if specified
    if args.model_input and os.path.exists(args.model_input):
        checkpoint = torch.load(args.model_input, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {args.model_input}")

    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n" + "="*70)
    print("Training...")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_p_loss, train_v_loss = train_epoch(model, train_loader, optimizer, device)
        val_p_loss, val_v_loss = validate(model, val_loader, device)

        val_loss = val_p_loss + val_v_loss

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train P: {train_p_loss:.4f} V: {train_v_loss:.4f} | "
              f"Val P: {val_p_loss:.4f} V: {val_v_loss:.4f} Total: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()

