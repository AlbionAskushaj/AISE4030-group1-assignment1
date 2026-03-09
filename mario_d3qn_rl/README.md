# Mario D3QN RL

Starter scaffold for a university reinforcement learning assignment using Mario on `SuperMarioBros-1-1-v3`.

The codebase currently includes:

- Task 1: D3QN (Dueling Double DQN) without experience replay.
- Task 2: D3QN with uniform experience replay.

The project is structured so Task 3 can be added without rewriting common environment, network, logging, and plotting code.

## Features

- Dueling DQN network architecture
- Double DQN target computation
- Task 1 online step-wise updates (no replay)
- Task 2 uniform replay buffer with mini-batch learning
- Shared utilities for metric logging and plotting

## Setup

Use Python 3.10 and create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The Mario environment is created with wrappers in this order:

1. `SkipFrame(4)`
2. `GrayScaleObservation`
3. `ResizeObservation(84)`
4. `FrameStack(4)`

Action space is restricted to:

- `0`: move right
- `1`: move right and jump

## Run Training

From the `mario_d3qn_rl/` directory:

```bash
python training_script.py
```

Task 1 artifacts are saved in `results/task1_d3qn/`.

For Task 2:

```bash
python training_task2.py
```

Task 2 artifacts are saved in `results/task2_d3qn_er/`.
