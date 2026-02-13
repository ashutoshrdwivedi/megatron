# Nanotron
Jax implementation of a nano GPT model

<img src="https://github.com/user-attachments/assets/bde3710e-1f54-4a18-83a7-3207b8cb4f2d" alt="jransformers" width="200"/>

## Installation

This project uses UV for package management. To get started:

## Usage

Run training:
```bash
uv run --env-file .env python -m nanotron.train
```
Generate samples
```bash
uv run --env-file .env python -m nanotron.sample
```
Debugging: To run a debugger you'll have to disable jit, which you can do using an env variable like this: 
```
JAX_DISABLE_JIT=1 uv run --env-file .env python -m nanotron.train
```

Alternatively, you can use the `train.sh` and `sample.sh` within the environment.


## Config

