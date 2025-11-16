# ChessHacks Starter Bot

This is a starter bot for ChessHacks. It includes a basic bot and devtools. This is designed to help you get used to what the interface for building a bot feels like, as well as how to scaffold your own bot.

## Directory Structure

### Core Directories

- **`/src`** - Source code for your bot
  - `/src/utils/` - Core utilities (model, MCTS, encoders)
  - `/src/data/` - Data generation scripts
  - `/src/training/` - Training scripts by phase
  - `/src/evaluation/` - Model testing & comparison
  - `/src/main.py` - Bot entry point

- **`/datasets`** - All training data (`.npz` files)

- **`/devtools`** - Next.js UI for testing your bot (gitignored)

- **`/tests`** - Phase-specific tests

### Root-Level Scripts

- `serve.py` - FastAPI backend for devtools UI
- `selfplay_generator.py` - Local self-play generation
- `selfplay_modal.py` - Modal GPU self-play
- `train_modal_selfplay.py` - Modal GPU training

### How It Works

`serve.py` is the backend that interacts with the Next.js app and your bot (`/src/main.py`). It handles hot reloading when you make changes. The Next.js app runs `serve.py` as a subprocess when you run `npm run dev`.

The backend deploys on port `5058` by default.

This architecture is similar to how your bot will run once deployed. See [the docs](https://docs.chesshacks.dev/) for deployment info.

## Setup

Start by creating a Python virtual environment and installing the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or however you want to set up your Python.

Then, install the dependencies for the Next.js app:

```bash
cd devtools
npm install
```

Afterwards, make a copy of `.env.template` and name it `.env.local` (NOT `.env`). Then fill out the values with the path to your Python environment, and the ports you want to use.

> Copy the `.env.local` file to the `devtools` directory as well.

## Installing `devtools` (if you did not run `npx chesshacks create`)

If you started from your own project and only want to add the devtools UI, you can install it with the CLI:

```bash
npx chesshacks install
```

This will add a `devtools` folder to your current directory and ensure it is gitignored. If you want to install into a subdirectory, you can pass a path:

```bash
npx chesshacks install my-existing-bot
```

In both cases, you can then follow the instructions in [Setup](#setup) and [Running the app](#running-the-app) from inside the `devtools` folder.

## Running the app

Lastly, simply run the Nextjs app inside of the devtools folder.

```bash
cd devtools
npm run dev
```

## Troubleshooting

First, make sure that you aren't running any `python` commands! These devtools are designed to help you play against your bot and see how its predictions are working. You can see [Setup](#setup) and [Running the app](#running-the-app) above for information on how to run the app. You should be running the Next.js app, not the Python files directly!

If you get an error like this:

```python
Traceback (most recent call last):
  File "/Users/obama/dev/chesshacks//src/main.py", line 1, in <module>
    from .utils import chess_manager, GameContext
ImportError: attempted relative import with no known parent package
```

you might think that you should remove the period before `utils` and that will fix the issue. But in reality, this will just cause more problems in the future! You aren't supposed to run `main.py ` on your own—it's designed for `serve.py` to run it for you within the subprocess. Removing the period would cause it to break during that step.

### Logs

Once you run the app, you should see logs from both the Next.js app and the Python subprocess, which includes both `serve.py` and `main.py`. `stdout`s and `stderr`s from both Python files will show in your Next.js terminal. They are designed to be fairly verbose by default.

## HMR (Hot Module Reloading)

By default, the Next.js app will automatically reload (dismount and remount the subprocess) when you make changes to the code in `/src` OR press the manual reload button on the frontend. This is called HMR (Hot Module Reloading). This means that you don't need to restart the app every time you make a change to the Python code. You can see how it's happening in real-time in the Next.js terminal.

## 🚀 Quick Command Reference

### **Data Generation**
```powershell
# Generate random games for bootstrapping
python -m src.data.generate_random_games --games 5000 --output datasets/random_5k.npz

# Generate Stockfish games (requires Stockfish installed)
python -m src.data.generate_stockfish_data --num-games 500 --depth 10

# Merge multiple datasets
python -m src.data.merge_datasets --datasets datasets/data1.npz datasets/data2.npz --output datasets/merged.npz
```

### **Training**
```powershell
# Local training (Phase 2)
python -m src.training.train_phase2 --data datasets/random_5k.npz --epochs 10

# Self-play training with soft targets
python -m src.training.train_selfplay --data datasets/selfplay_data.npz --epochs 20

# Modal GPU training (fast!)
modal run train_modal_selfplay.py --data-file datasets/random_5k.npz --epochs 10
```

### **Evaluation**
```powershell
# Compare two models
python -m src.evaluation.test_models --model1 chess_model_sp_v1.pth --model2 chess_model_sp_v2.pth --games 10

# Watch a model play
python -m src.evaluation.watch_model --model chess_model.pth --games 5
```

### **Self-Play Generation**
```powershell
# Local self-play
python selfplay_generator.py --games 100 --mcts-sims 50

# Modal GPU self-play (fast!)
modal run selfplay_modal.py --games 500 --mcts-sims 50 --output-name selfplay_500g.npz
```

---

## Parting Words

Keep in mind that you fully own all of this code! This entire devtool system runs locally, so feel free to modify it however you want. This is just designed as scaffolding to help you get started.

If you need further help, please first check out the [docs](https://docs.chesshacks.dev/). If you still need help, please join our [Discord](https://docs.chesshacks.dev/resources/discord) and ask for help.
