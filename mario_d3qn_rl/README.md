# Mario D3QN RL

Starter scaffold for a university reinforcement learning assignment using Mario on `SuperMarioBros-1-1-v3`.

The codebase currently includes:

- Task 1: D3QN (Dueling Double DQN) without experience replay.
- Task 2: D3QN with uniform experience replay.

The project uses a single configuration file and a single training script so each assignment variant can be selected without changing code.

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

Before running, set `agent_type` in `config.yaml`:

- `d3qn` for Task 1
- `d3qn_er` for Task 2

Task 1 artifacts are saved in `results/task1_d3qn/`.

Task 2 artifacts are saved in `results/task2_d3qn_er/`.

The default `training.episodes` value in `config.yaml` is `5000`.

## Device Selection

`config.yaml` supports these device values:

- `auto`: use CUDA if available, otherwise fall back to CPU
- `cuda`: request GPU training and fall back to CPU if CUDA is unavailable
- `cpu`: force CPU training
