"""
Quick script to inspect the trained model checkpoint.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def inspect_checkpoint(model_path=None):
    """Inspect the model checkpoint file."""
    if model_path is None:
        # Look for model in parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, "chess_model.pth")
    print(f"Inspecting: {model_path}")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("Checkpoint keys:", list(checkpoint.keys()))
    print()
    
    if 'epoch' in checkpoint:
        print(f"Training epoch: {checkpoint['epoch']}")
    
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel layers ({len(state_dict)} parameters):")
        for name, param in list(state_dict.items())[:10]:  # Show first 10
            print(f"  {name}: shape {tuple(param.shape)}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more layers")
    
    if 'move_mapper' in checkpoint:
        mapper = checkpoint['move_mapper']
        if hasattr(mapper, 'next_index'):
            print(f"\nMove mapper: {mapper.next_index} unique moves mapped")
        else:
            print("\nMove mapper: present (custom object)")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "chess_model.pth"
    inspect_checkpoint(model_path)

