# 🤖 Chess Neural Network Bot

A sophisticated chess bot powered by a custom-trained neural network using deep reinforcement learning, self-play, and tactical pattern recognition. Built for the ChessHacks hackathon with comprehensive training pipelines and evaluation tools.

## 🎯 Features

- **Deep Neural Network Architecture**: CNN/ResNet-based model with residual blocks for advanced position evaluation
- **Self-Play Training**: Iterative improvement through playing against itself
- **Tactical Pattern Recognition**: Specialized training on forks, pins, checkmates, and hanging pieces
- **Real Grandmaster Games**: Trained on high-quality games from Lichess database
- **MCTS Integration**: Optional Monte Carlo Tree Search for enhanced decision-making
- **Live Testing Interface**: Next.js-based devtools for real-time bot evaluation

## 📁 Project Structure

```
my-chesshacks-bot/
├── src/                          # Core source code
│   ├── main.py                  # Bot entry point
│   ├── data/                    # Data generation modules
│   ├── training/                # Training pipelines
│   ├── evaluation/              # Model testing & comparison
│   └── utils/                   # Core utilities (model, MCTS, encoders)
│
├── models/                       # Trained model checkpoints (.pth files)
│   ├── chess_model_ULTIMATE_V2.pth
│   ├── chess_model_DETERMINISTIC.pth
│   └── ...                      # Various model iterations
│
├── training_data/               # Organized training datasets
│   ├── checkmates/             # Checkmate patterns
│   ├── tactical/               # Tactical motifs (forks, pins, etc.)
│   ├── selfplay/               # Self-play generated games
│   ├── strategic/              # Strategic position training
│   ├── real_games/             # GM/Lichess games
│   ├── synthetic/              # Synthetically generated positions
│   └── tests/                  # Test datasets
│
├── datasets/                     # Base training datasets
│   └── *.npz                   # Random, stockfish, and base data
│
├── scripts/                      # Utility and execution scripts
│   ├── generation/             # Data generation scripts
│   ├── training/               # Training execution scripts
│   ├── testing/                # Bot quality tests
│   ├── comparison/             # Model comparison tools
│   └── utilities/              # Helper scripts
│
├── devtools/                     # Next.js UI for testing (gitignored)
│   └── ...                     # Frontend application
│
├── logs/                         # Training and evaluation logs
├── docs/                         # Phase guides and documentation
├── serve.py                      # FastAPI backend for devtools
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for devtools)
- Optional: Modal account (for GPU training)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd my-chesshacks-bot
```

2. **Set up Python environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install devtools dependencies**
```bash
cd devtools
npm install
```

4. **Configure environment**
```bash
# Copy environment template
cp .env.template .env.local
# Edit .env.local with your Python path and ports
```

### Running the Bot

**Development Mode with UI:**
```bash
cd devtools
npm run dev
```

This starts both the Next.js frontend and Python backend (`serve.py`) on port 5058.

**Command Line Testing:**
```bash
# Test bot quality
python scripts/testing/test_bot_quality_FIXED.py

# Play against Stockfish
python scripts/testing/test_vs_stockfish.py

# Compare two models
python scripts/comparison/compare_models_modal.py
```

## 🎓 Training Pipeline

### Phase 1: Bootstrap Training
Generate initial random game data:
```bash
python -m src.data.generate_random_games --games 5000 --output datasets/random_5k.npz
python -m src.training.train_phase2 --data datasets/random_5k.npz --epochs 10
```

### Phase 2: Self-Play Training
Use the model to generate better training data:
```bash
# Local self-play
python scripts/generation/selfplay_generator.py --games 100 --mcts-sims 50

# GPU-accelerated (Modal)
modal run scripts/utilities/selfplay_modal.py --games 500 --mcts-sims 50
```

### Phase 3: Tactical Training
Train on specific tactical patterns:
```bash
python scripts/generation/generate_checkmates.py
python scripts/generation/tactical_generator.py
python scripts/training/train_ultimate_v2.py
```

### Phase 4: Real Games Integration
Download and train on grandmaster games:
```bash
python scripts/generation/download_lichess_games.py
python scripts/training/merge_and_train.py
```

## 📊 Model Architecture

**ChessModel (CNN/ResNet)**
- Input: 12-channel board representation (6 piece types × 2 colors)
- Architecture: 6 residual blocks, 64 channels per layer
- Outputs:
  - Policy head: 4672 possible moves
  - Value head: Position evaluation (-1 to +1)
- Optimizer: AdamW with learning rate scheduling
- Total parameters: ~500K+

**Board Encoding:**
- 8×8×12 tensor (piece-centric representation)
- Turn indicator and castling rights as additional features
- Fixed move mapper for consistent encoding (4672 UCI moves)

## 🔧 Key Scripts

### Data Generation
- `generate_checkmates.py` - Generate checkmate puzzles
- `generate_tactical_training.py` - Create tactical positions
- `download_lichess_games.py` - Fetch real GM games
- `selfplay_generator.py` - Self-play data generation

### Training
- `train_ultimate_v2.py` - Main training script
- `train_modal_selfplay.py` - GPU-accelerated training
- `aggressive_training.py` - High-intensity training loop

### Evaluation
- `test_bot_quality_FIXED.py` - Evaluate bot strength
- `test_vs_stockfish.py` - Benchmark against Stockfish
- `compare_models_modal.py` - Compare model versions
- `test_response_time.py` - Performance benchmarking

## 📈 Training Data

The bot has been trained on a diverse dataset including:

- **50,000+** positions from random/bootstrap games
- **20,000+** self-play positions (MCTS-enhanced)
- **10,000+** tactical patterns (forks, pins, discovered attacks)
- **5,000+** checkmate positions
- **15,000+** positions from GM games (2200+ ELO)
- **3,000+** strategic endgame positions

All data is stored in NumPy `.npz` format with:
- `states`: Board representations (N × 8 × 8 × 12)
- `policies`: Move probabilities (N × 4672)
- `values`: Position evaluations (N × 1)

## 🎮 Usage Examples

### Basic Bot Usage
```python
from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor_torch
import chess
import torch

# Load model
model = ChessModel()
checkpoint = torch.load('models/chess_model_ULTIMATE_V2.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make a move
board = chess.Board()
board_tensor = board_to_tensor_torch(board)
with torch.no_grad():
    policy, value = model(board_tensor)

print(f"Position evaluation: {value.item()}")
```

### Custom Training
```python
from src.training.train_phase2 import train

# Train with custom parameters
train(
    data_path='training_data/custom_data.npz',
    model_path='models/custom_model.pth',
    epochs=20,
    batch_size=256,
    learning_rate=0.001
)
```

## 🔍 Model Versions

- **chess_model_ULTIMATE_V2.pth** - Latest production model
- **chess_model_DETERMINISTIC.pth** - Deterministic policy model
- **chess_model_GM_TRAINED.pth** - Trained on GM games
- **chess_model_TACTICAL_BLITZ.pth** - Specialized for tactics

See `models/` directory for all available checkpoints.

## 🧪 Testing & Evaluation

```bash
# Quick quality test (10 games)
python scripts/testing/test_final_bot.py

# Comprehensive evaluation
python scripts/testing/test_bot_quality_FIXED.py --games 50

# Compare against Stockfish (depth 5)
python scripts/testing/test_vs_stockfish.py --depth 5 --games 20

# Test response time
python scripts/testing/test_response_time.py
```

## 📝 Development Notes

### Hot Module Reloading (HMR)
The devtools automatically reload when you modify code in `/src`. Press the manual reload button in the UI or let it detect changes automatically.

### Common Issues

**Import Error (`attempted relative import`)**
- Don't run `main.py` directly
- Use `npm run dev` in the devtools folder
- The subprocess architecture requires this setup

**Model Not Loading**
- Ensure model file exists at the expected path
- Check for architecture mismatches (old MLP vs new CNN)
- Verify checkpoint format (should contain `model_state_dict`)

### Performance Tips
- Use GPU training with Modal for 10-50× speedup
- Batch size of 256-512 works well for most systems
- MCTS with 50-100 simulations balances speed/quality
- Disable MCTS for faster inference (raw policy is still strong)

## 📚 Additional Resources

- **Documentation**: See `docs/` for phase-specific guides
- **ChessHacks Docs**: [docs.chesshacks.dev](https://docs.chesshacks.dev/)
- **Discord**: Join for support and discussions

## 🤝 Contributing

This is a hackathon project, but feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
# WagWan-Croodie
