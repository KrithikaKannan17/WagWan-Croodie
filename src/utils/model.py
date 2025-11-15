"""
Neural network model for chess move prediction and position evaluation.
Uses a simple MLP architecture for initial implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessModel(nn.Module):
    """
    Simple MLP model for chess move prediction.
    
    Input: 8x8x12 board representation (flattened to 768)
    Output: 
    - Move probabilities (policy head)
    - Position evaluation (value head)
    """
    
    def __init__(self, hidden_size=256, num_hidden_layers=3):
        """
        Initialize the chess model.
        
        Args:
            hidden_size: Size of hidden layers
            num_hidden_layers: Number of hidden layers
        """
        super(ChessModel, self).__init__()
        
        # Input: 12 channels * 8 * 8 = 768
        input_size = 12 * 8 * 8
        
        # Build hidden layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head: outputs probabilities for moves
        # We'll use a variable output size based on legal moves
        # For now, use a large fixed size (max possible moves ~218)
        self.policy_head = nn.Linear(hidden_size, 256)  # Max ~218 legal moves
        
        # Value head: outputs position evaluation (win probability)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            policy_logits: Raw logits for move probabilities, shape (batch, 256)
            value: Position evaluation, shape (batch, 1)
        """
        # Flatten: (batch, 12, 8, 8) -> (batch, 768)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        # Shared layers
        shared = self.shared_layers(x)
        
        # Policy head
        policy_logits = self.policy_head(shared)
        
        # Value head
        value = self.value_head(shared)
        
        return policy_logits, value
    
    def predict(self, x):
        """
        Predict move probabilities and position value.
        
        Args:
            x: Input tensor of shape (batch, 12, 8, 8)
            
        Returns:
            move_probs: Softmax probabilities for moves, shape (batch, 256)
            value: Position evaluation, shape (batch, 1)
        """
        policy_logits, value = self.forward(x)
        move_probs = F.softmax(policy_logits, dim=1)
        return move_probs, value

