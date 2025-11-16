"""
Train on merged tactical + selfplay data on Modal GPU
Merges the data ON Modal to avoid download/upload cycles
"""

import modal

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.2.1",
        "python-chess==1.999",
    )
)

app = modal.App("train-merged-tactical-selfplay", image=image)

# Volume for persistent storage
volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)

# Add local code
image = image.add_local_dir("src", remote_path="/root/src")
image = image.add_local_file("chess_model_sp_v2.pth", remote_path="/root/chess_model_sp_v2.pth")


@app.function(
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,
    image=image,
)
def merge_and_train(
    selfplay_file1: str = "/data/data/selfplay_v1.npz",
    selfplay_file2: str = "/data/data/selfplay_v2.npz",
    tactical_file: str = "/data/data/tactical_data_all_merged.npz",
    tactical_ratio: float = 0.3,
    model_input: str = "/root/chess_model_sp_v2.pth",
    model_output: str = "chess_model_merged_v1.pth",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
):
    """Merge data and train on Modal GPU."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import numpy as np
    import sys
    import os
    sys.path.insert(0, '/root')
    from src.utils.model import ChessModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # ============= MERGE DATA =============
    print("\n" + "="*70)
    print("🔗 MERGING DATA ON MODAL GPU")
    print("="*70)
    
    # Check volume contents
    print("\n📂 Volume contents:")
    try:
        for file in os.listdir("/data"):
            print(f"  - /data/{file}")
    except Exception as e:
        print(f"  Error listing /data: {e}")
    
    # Load data
    print("\nLoading data...")
    sp1 = np.load(selfplay_file1)
    tactical = np.load(tactical_file)
    
    print(f"  Self-play v1: {len(sp1['boards'])} positions")
    print(f"  Tactical: {len(tactical['boards'])} positions")
    
    # Use only selfplay v1 (correct format)
    sp_boards = sp1['boards']
    sp_policy = sp1['policy_targets']
    sp_values = sp1['value_targets']
    
    total_selfplay = len(sp_boards)
    total_tactical = len(tactical['boards'])
    
    # Calculate sampling
    sampled_selfplay_count = int(total_tactical * (1 - tactical_ratio) / tactical_ratio)
    if sampled_selfplay_count > total_selfplay:
        sampled_selfplay_count = total_selfplay
        sampled_tactical_count = int(total_selfplay * tactical_ratio / (1 - tactical_ratio))
    else:
        sampled_tactical_count = total_tactical
    
    print(f"\n📊 Using {sampled_tactical_count} tactical + {sampled_selfplay_count} self-play")
    print(f"   Ratio: {sampled_tactical_count / (sampled_tactical_count + sampled_selfplay_count) * 100:.1f}% tactical")
    
    # Sample
    tactical_indices = np.random.choice(total_tactical, sampled_tactical_count, replace=False)
    selfplay_indices = np.random.choice(total_selfplay, sampled_selfplay_count, replace=False)
    
    # Combine and shuffle
    merged_boards = np.concatenate([
        tactical['boards'][tactical_indices],
        sp_boards[selfplay_indices]
    ])
    merged_policy = np.concatenate([
        tactical['policy_targets'][tactical_indices],
        sp_policy[selfplay_indices]
    ])
    merged_values = np.concatenate([
        tactical['value_targets'][tactical_indices],
        sp_values[selfplay_indices]
    ])
    
    indices = np.random.permutation(len(merged_boards))
    merged_boards = merged_boards[indices]
    merged_policy = merged_policy[indices]
    merged_values = merged_values[indices]
    
    print(f"✅ Merged dataset: {len(merged_boards)} positions")
    
    # ============= TRAINING =============
    print("\n" + "="*70)
    print("🎓 TRAINING ON MERGED DATASET")
    print("="*70)
    
    # Dataset
    class MergedDataset(Dataset):
        def __init__(self, boards, policy_targets, value_targets):
            self.boards = torch.FloatTensor(boards)
            self.policy_targets = torch.FloatTensor(policy_targets)
            self.value_targets = torch.FloatTensor(value_targets).unsqueeze(1)
        
        def __len__(self):
            return len(self.boards)
        
        def __getitem__(self, idx):
            return self.boards[idx], self.policy_targets[idx], self.value_targets[idx]
    
    dataset = MergedDataset(merged_boards, merged_policy, merged_values)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} positions")
    print(f"  Val: {val_size} positions")
    
    # Load model
    print(f"\nLoading model from {model_input}...")
    checkpoint = torch.load(model_input, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect architecture
    if 'initial_conv.weight' in state_dict:
        channels = state_dict['initial_conv.weight'].shape[0]
        num_residual_blocks = sum(1 for key in state_dict.keys() if key.startswith('residual_blocks.') and key.endswith('.conv1.weight'))
    else:
        num_residual_blocks = 6
        channels = 64
    
    model = ChessModel(num_residual_blocks=num_residual_blocks, channels=channels).to(device)
    model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Model loaded: {num_residual_blocks} blocks, {channels} channels")
    
    # Training setup
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for boards, policy_targets, value_targets in train_loader:
            boards, policy_targets, value_targets = boards.to(device), policy_targets.to(device), value_targets.to(device)
            
            optimizer.zero_grad()
            policy_logits, value_pred = model(boards)
            
            loss_policy = criterion_policy(policy_logits, policy_targets)
            loss_value = criterion_value(value_pred, value_targets)
            loss = loss_policy + loss_value
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for boards, policy_targets, value_targets in val_loader:
                boards, policy_targets, value_targets = boards.to(device), policy_targets.to(device), value_targets.to(device)
                policy_logits, value_pred = model(boards)
                loss_policy = criterion_policy(policy_logits, policy_targets)
                loss_value = criterion_value(value_pred, value_targets)
                val_loss += (loss_policy + loss_value).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model to volume
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_residual_blocks': num_residual_blocks,
                'channels': channels,
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, f"/data/data/{model_output}")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: /data/{model_output}")
    print("="*70)
    
    volume.commit()
    print("\n✓ Volume committed")


@app.local_entrypoint()
def main(
    tactical_ratio: float = 0.3,
    epochs: int = 10,
    download: bool = False,
    model_output: str = "chess_model_merged_v1.pth",
):
    """Local entry point."""
    if download:
        print(f"\n📥 Downloading {model_output} from Modal volume...")
        with volume.batch_download() as batch:
            batch.download(f"/data/data/{model_output}", f"./{model_output}")
        print(f"✅ Downloaded to ./{model_output}")
    else:
        print("\n🚀 Starting merged training on Modal GPU...")
        print(f"   Tactical ratio: {tactical_ratio*100:.0f}%")
        print(f"   Epochs: {epochs}")
        print(f"   Output: {model_output}")
        merge_and_train.remote(
            tactical_ratio=tactical_ratio,
            model_output=model_output,
            epochs=epochs
        )
        print(f"\n✅ Training complete! Model saved to Modal volume: /data/{model_output}")
        print(f"\n📥 To download:")
        print(f"   modal run train_merged_modal.py --download --model-output {model_output}")
        print(f"\n🎮 To evaluate:")
        print(f"   python -m src.evaluation.test_model_comparison \\")
        print(f"     --model1 chess_model_sp_v2.pth \\")
        print(f"     --model2 {model_output} \\")
        print(f"     --games 20 --skip-policy --skip-value")

