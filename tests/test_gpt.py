"""
Tests for the top-level GPT model (nanotron/model.py).

Focuses on output shapes across three calling modes:
  - training forward pass  (all token logits)
  - inference forward pass (last token only)
  - autoregressive decode  (token-by-token generation)
"""

import jax
import jax.numpy as jnp

from nanotron import model
from nanotron.config import GPTConfig


def _small_config() -> GPTConfig:
    """
    Minimal model config for fast shape-checking tests.
    Small enough to construct and run in milliseconds on CPU.

    block_size=8  — max sequence length the model supports
    n_layers=1    — single transformer block (shape tests don't need depth)
    vocab_size=10 — tiny vocabulary (characters 0–9)
    n_head=2      — two attention heads
    n_embed=8     — embedding dimension (must be divisible by n_head)
    dropout=0.0   — disabled so tests are deterministic
    """
    return GPTConfig(
        block_size=8, n_layers=1, vocab_size=10, n_head=2, n_embed=8, dropout=0.0
    )


def test_gpt_forward_shape():
    """
    In training mode (inference=False), the model returns one logit vector per
    input token: shape (S, V).

    This is what the training loop expects — it computes cross-entropy loss
    against the next-token targets for every position simultaneously.
    """
    key = jax.random.PRNGKey(0)
    gpt = model.GPT(key, _small_config())
    tokens_S = jnp.array([1, 2, 3, 4])  # (S=4,)
    logits_SxV = gpt(key, tokens_S)
    assert logits_SxV.shape == (4, 10)  # (S, V)


def test_gpt_inference_shape():
    """
    In inference mode (inference=True), the model only returns the logit vector
    for the *last* token, wrapped in a leading dimension: shape (1, V).

    Why: during autoregressive generation we only need to sample the next token,
    so computing logits for every past position would be wasted work.
    """
    key = jax.random.PRNGKey(0)
    gpt = model.GPT(key, _small_config())
    tokens_S = jnp.array([1, 2, 3])  # (S=3,)
    logits_SxV = gpt(key, tokens_S, inference=True)
    assert logits_SxV.shape == (1, 10)  # (1, V) — always 1 token in inference mode


def test_gpt_decode_shapes():
    """
    decode() grows the token sequence by exactly max_new_tokens.
    The returned array contains the original prompt concatenated with the
    newly generated tokens.
    """
    key = jax.random.PRNGKey(0)
    gpt = model.GPT(key, _small_config())
    prompt_S = jnp.array([1, 2, 3])  # (S=3,) prompt tokens
    key, subkey = jax.random.split(key)
    out_S = gpt.decode(subkey, prompt_S, max_new_tokens=2)
    assert out_S.shape[0] == len(prompt_S) + 2  # 3 prompt + 2 generated = 5
