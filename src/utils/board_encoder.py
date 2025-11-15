"""
Board representation encoder for chess positions.
Converts python-chess Board to 8x8x12 tensor representation.

Encoding: 12 channels (one for each piece type and color)
- Channel 0: White Pawns
- Channel 1: White Rooks
- Channel 2: White Knights
- Channel 3: White Bishops
- Channel 4: White Queens
- Channel 5: White King
- Channel 6: Black Pawns
- Channel 7: Black Rooks
- Channel 8: Black Knights
- Channel 9: Black Bishops
- Channel 10: Black Queens
- Channel 11: Black King
"""

import numpy as np
import torch
from chess import Board, Piece, PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING


def board_to_tensor(board: Board) -> np.ndarray:
    """
    Convert a python-chess Board to an 8x8x12 numpy array.
    
    Args:
        board: python-chess Board object
        
    Returns:
        numpy array of shape (8, 8, 12) with binary values
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Map piece types to channel indices
    piece_to_channel = {
        (PAWN, True): 0,    # White Pawn
        (ROOK, True): 1,    # White Rook
        (KNIGHT, True): 2,  # White Knight
        (BISHOP, True): 3,  # White Bishop
        (QUEEN, True): 4,   # White Queen
        (KING, True): 5,    # White King
        (PAWN, False): 6,   # Black Pawn
        (ROOK, False): 7,   # Black Rook
        (KNIGHT, False): 8, # Black Knight
        (BISHOP, False): 9, # Black Bishop
        (QUEEN, False): 10, # Black Queen
        (KING, False): 11,  # Black King
    }
    
    # Fill tensor based on board state
    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            color = piece.color
            channel = piece_to_channel[(piece_type, color)]
            
            # Convert square index to (rank, file)
            rank = 7 - (square // 8)  # 0-7, where 0 is rank 8 (top from White's perspective)
            file = square % 8         # 0-7, where 0 is file a
            
            tensor[rank, file, channel] = 1.0
    
    return tensor


def board_to_tensor_torch(board: Board) -> torch.Tensor:
    """
    Convert a python-chess Board to a PyTorch tensor.
    
    Args:
        board: python-chess Board object
        
    Returns:
        torch.Tensor of shape (1, 12, 8, 8) - batch dimension added for model input
    """
    numpy_tensor = board_to_tensor(board)
    # Convert to (1, 12, 8, 8) format: (batch, channels, height, width)
    torch_tensor = torch.from_numpy(numpy_tensor).permute(2, 0, 1).unsqueeze(0)
    return torch_tensor.float()

