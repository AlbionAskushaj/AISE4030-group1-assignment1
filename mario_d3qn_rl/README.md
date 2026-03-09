# Mario D3QN RL

Starter scaffold for a university reinforcement learning assignment using Task 1: D3QN (Dueling Double DQN) without experience replay on `SuperMarioBros-1-1-v3`.

The project is structured so later tasks can extend the current setup without rewriting the training loop, logging utilities, or environment factory. Task 1 performs online updates only, meaning the agent learns from the current transition at every environment step.

## Features

- Dueling DQN network architecture
- Double DQN target computation
- No experience replay
- Online step-wise learning updates
- Modular environment, agent, network, logging, and plotting code
- Training reward and loss tracking

## Setup

Use Python 3.10 and create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The project uses:

- `gym-super-mario-bros`
- `nes-py`
- `PyTorch`

The Mario environment is created with the following wrappers in order:

1. `SkipFrame(4)`
2. `GrayScaleObservation`
3. `ResizeObservation(84)`
4. `FrameStack(4)`

The action space is restricted to:

- `0`: move right
- `1`: move right and jump

## Configuration

Training hyperparameters live in `config.yaml`. Adjust them there before running experiments.

## Run Training

From the `mario_d3qn_rl/` directory:

```bash
python training_script.py
```

Training artifacts are saved in `results/task1_d3qn/`.

## Extension Notes

The code is intentionally separated into `agents/`, `networks/`, `environment/`, and `utils/` so Task 2 and Task 3 features such as replay buffers or prioritized replay can be added with minimal changes to the rest of the project.
