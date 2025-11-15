"""
Monte Carlo Tree Search (MCTS) implementation with neural network guidance.
Uses neural network for policy (move probabilities) and value (position evaluation).
"""

import math
import numpy as np
from typing import List, Optional, Dict, Tuple
from chess import Board, Move
import torch


class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        """
        Initialize a MCTS node.
        
        Args:
            board: Current board position
            parent: Parent node (None for root)
            move: Move that led to this position
        """
        self.board = board
        self.parent = parent
        self.move = move
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.value_estimate = 0.0  # From neural network
        
        # Children
        self.children: Dict[Move, 'MCTSNode'] = {}
        self.legal_moves = list(board.generate_legal_moves())
        
        # Policy prior from neural network (probability for each legal move)
        self.policy_prior: Dict[Move, float] = {}
        
        # Whether this node has been expanded
        self.expanded = False
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been explored."""
        return len(self.children) == len(self.legal_moves) and self.expanded
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over()
    
    def get_value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return self.value_estimate
        return self.value_sum / self.visit_count
    
    def ucb_score(self, exploration_constant: float = 1.5) -> float:
        """
        Calculate UCB1 score for selection.
        Only valid for non-root nodes.
        """
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.get_value()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def select_child(self, exploration_constant: float = 1.5) -> 'MCTSNode':
        """
        Select child using UCB formula with policy prior.
        Uses PUCT (Policy + UCT) algorithm.
        """
        best_score = float('-inf')
        best_child = None
        
        for move in self.legal_moves:
            if move not in self.children:
                # Unexplored move - use policy prior
                prior = self.policy_prior.get(move, 1.0 / len(self.legal_moves))
                # Encourage exploration of unexplored moves
                score = prior * math.sqrt(self.visit_count) / (1 + 0)  # No visits yet
            else:
                child = self.children[move]
                prior = self.policy_prior.get(move, 1.0 / len(self.legal_moves))
                
                # PUCT formula: Q + c_puct * P * sqrt(N) / (1 + n)
                q_value = child.get_value()
                puct = exploration_constant * prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
                score = q_value + puct
            
            if score > best_score:
                best_score = score
                best_child = move
        
        if best_child is None:
            # Fallback: return first legal move
            best_child = self.legal_moves[0]
        
        return self.children.get(best_child, None)
    
    def expand(self, policy_prior: Dict[Move, float], value_estimate: float):
        """
        Expand this node by adding policy prior and value estimate.
        
        Args:
            policy_prior: Dictionary mapping moves to probabilities
            value_estimate: Value estimate from neural network
        """
        self.policy_prior = policy_prior
        self.value_estimate = value_estimate
        self.expanded = True
    
    def backpropagate(self, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            value: Value to propagate (from perspective of node's player)
        """
        self.visit_count += 1
        self.value_sum += value
        
        # Propagate to parent (negate value since it's opponent's perspective)
        if self.parent is not None:
            self.parent.backpropagate(-value)
    
    def get_best_move(self) -> Move:
        """Get the move with highest visit count (most explored)."""
        if not self.children:
            # No children explored, return first legal move
            return self.legal_moves[0] if self.legal_moves else None
        
        best_move = max(self.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_move


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """
    
    def __init__(self, model, move_mapper, num_simulations: int = 100, exploration_constant: float = 1.5):
        """
        Initialize MCTS.
        
        Args:
            model: Trained neural network model
            move_mapper: MoveMapper instance
            num_simulations: Number of MCTS simulations per move
            exploration_constant: Exploration constant for UCB
        """
        self.model = model
        self.move_mapper = move_mapper
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.model.eval()
    
    def search(self, board: Board) -> Tuple[Move, Dict[Move, float]]:
        """
        Perform MCTS search from given position.
        
        Args:
            board: Current board position
            
        Returns:
            Best move and move probabilities (from visit counts)
        """
        root = MCTSNode(board)
        
        # Get initial policy and value from neural network
        policy_prior, value_estimate = self._evaluate_position(root.board)
        root.expand(policy_prior, value_estimate)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Selection: traverse from root to leaf
            node = self._select(root)
            
            # Evaluation: get policy and value from neural network
            if not node.is_terminal():
                policy_prior, value_estimate = self._evaluate_position(node.board)
                node.expand(policy_prior, value_estimate)
                
                # Expand: add children for all legal moves
                self._expand(node)
            
            # Backpropagation: update statistics
            if node.is_terminal():
                # Terminal node: get actual game result
                value = self._get_terminal_value(node.board)
            else:
                # Use neural network value estimate
                value = value_estimate
            
            node.backpropagate(value)
        
        # Get best move and probabilities
        best_move = root.get_best_move()
        move_probs = self._get_move_probabilities(root)
        
        return best_move, move_probs
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select a leaf node by traversing the tree.
        
        Args:
            root: Root node
            
        Returns:
            Leaf node to evaluate
        """
        node = root
        
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child(self.exploration_constant)
            if node is None:
                break
        
        return node
    
    def _expand(self, node: MCTSNode):
        """
        Expand a node by creating children for all legal moves.
        
        Args:
            node: Node to expand
        """
        for move in node.legal_moves:
            if move not in node.children:
                # Create child node
                child_board = node.board.copy()
                child_board.push(move)
                child_node = MCTSNode(child_board, parent=node, move=move)
                node.children[move] = child_node
    
    def _evaluate_position(self, board: Board) -> Tuple[Dict[Move, float], float]:
        """
        Evaluate position using neural network.
        
        Args:
            board: Board position to evaluate
            
        Returns:
            Tuple of (policy_prior, value_estimate)
        """
        from .board_encoder import board_to_tensor_torch
        
        board_tensor = board_to_tensor_torch(board)
        legal_moves = list(board.generate_legal_moves())
        
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
        
        # Convert to move probabilities
        policy_prior = self.move_mapper.get_move_probabilities(
            policy_logits[0].numpy(), legal_moves
        )
        
        # Get value estimate
        value_estimate = value.item()
        
        # Adjust value based on whose turn it is
        # Model outputs from white's perspective
        if not board.turn:  # Black's turn
            value_estimate = -value_estimate
        
        return policy_prior, value_estimate
    
    def _get_terminal_value(self, board: Board) -> float:
        """
        Get value for terminal position.
        
        Args:
            board: Terminal board position
            
        Returns:
            Value from perspective of player to move
        """
        if board.is_checkmate():
            return -1.0  # Lost (opponent checkmated us)
        else:
            return 0.0  # Draw
    
    def _get_move_probabilities(self, root: MCTSNode) -> Dict[Move, float]:
        """
        Get move probabilities from visit counts.
        
        Args:
            root: Root node
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        if not root.children:
            # No children explored, return uniform
            return {move: 1.0 / len(root.legal_moves) for move in root.legal_moves}
        
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits == 0:
            return {move: 1.0 / len(root.legal_moves) for move in root.legal_moves}
        
        move_probs = {}
        for move, child in root.children.items():
            move_probs[move] = child.visit_count / total_visits
        
        # Normalize to ensure all legal moves have probabilities
        for move in root.legal_moves:
            if move not in move_probs:
                move_probs[move] = 0.0
        
        # Renormalize
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        
        return move_probs

