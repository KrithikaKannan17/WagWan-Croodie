"""
Move mapping utilities to convert between chess moves and model output indices.
Since the model outputs 256 logits, we need to map legal moves to these indices.
"""

from chess import Move, Square
from typing import Dict, List, Optional
import numpy as np


class MoveMapper:
    """
    Maps chess moves to model output indices.
    Uses a simple encoding: from_square * 64 + to_square (max 4096, but we use 256)
    For simplicity, we'll use a hash-based approach.
    """
    
    def __init__(self):
        self.move_to_index: Dict[str, int] = {}
        self.index_to_move: Dict[int, str] = {}
        self.next_index = 0
        self.max_indices = 256
    
    def get_move_index(self, move: Move) -> int:
        """
        Get the index for a move. Creates new index if move hasn't been seen.
        
        Args:
            move: Chess move (can be Move object or UCI string)
            
        Returns:
            Index in range [0, 255]
        """
        # Convert to UCI string
        if isinstance(move, Move):
            move_str = move.uci()
        else:
            move_str = str(move)
        
        # If we've seen this move, return its index
        if move_str in self.move_to_index:
            return self.move_to_index[move_str]
        
        # If we haven't seen it and have space, assign new index
        if self.next_index < self.max_indices:
            index = self.next_index
            self.move_to_index[move_str] = index
            self.index_to_move[index] = move_str
            self.next_index += 1
            return index
        
        # If we're out of space, use hash-based mapping
        # This ensures we always return a valid index
        index = hash(move_str) % self.max_indices
        if index not in self.index_to_move:
            self.index_to_move[index] = move_str
        if move_str not in self.move_to_index:
            self.move_to_index[move_str] = index
        return index
    
    def get_move_from_index(self, index: int) -> Optional[str]:
        """
        Get the move UCI string from an index.
        
        Args:
            index: Model output index
            
        Returns:
            Move UCI string or None if index not found
        """
        return self.index_to_move.get(index)
    
    def create_policy_target(self, legal_moves: List[Move], target_move: Move) -> np.ndarray:
        """
        Create a policy target vector (one-hot for the target move).
        
        Args:
            legal_moves: List of legal moves
            target_move: The move that was actually played
            
        Returns:
            Array of shape (256,) with 1.0 at target move index, 0.0 elsewhere
        """
        target = np.zeros(self.max_indices, dtype=np.float32)
        target_index = self.get_move_index(target_move)
        target[target_index] = 1.0
        return target
    
    def get_move_probabilities(self, policy_logits: np.ndarray, legal_moves: List[Move]) -> Dict[Move, float]:
        """
        Convert model policy output to move probabilities for legal moves only.
        
        Args:
            policy_logits: Raw logits from model, shape (256,)
            legal_moves: List of legal moves in current position
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        # Get indices for legal moves
        legal_indices = [self.get_move_index(move) for move in legal_moves]
        
        # Extract logits for legal moves
        legal_logits = policy_logits[legal_indices]
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(legal_logits - np.max(legal_logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Create dictionary
        move_probs = {move: float(prob) for move, prob in zip(legal_moves, probabilities)}
        return move_probs

