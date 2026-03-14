"""
Training health tests for the GPT model (nanotron/model.py + train.py).

These tests do not check shapes — they check that the model can actually *learn*.
Three complementary checks:

  1. test_loss_is_finite      — forward pass doesn't produce NaN/Inf
  2. test_no_nan_gradients    — backward pass doesn't produce NaN/Inf
  3. test_overfit_single_batch — model can reduce loss on a repeated batch

Plus a golden-file regression test to catch silent numerical changes.

Note: run with JAX_PLATFORM_NAME=cpu on this shared GPU machine to avoid
XLA device conflicts with other users.
"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nanotron import model
from nanotron.config import GPTConfig


# Golden file stores reference logits produced by a fixed model + seed.
# Delete this file whenever an intentional architectural change alters numerics.
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
GOLDEN_FILE = os.path.join(GOLDEN_DIR, "logits.npy")


def _small_config() -> GPTConfig:
    """
    Slightly larger than test_gpt.py's config so training tests have enough
    capacity to actually overfit a tiny batch.

    n_layers=2, n_embed=16 gives ~10k parameters — small enough to train in
    seconds on CPU but expressive enough that loss can drop significantly.
    dropout=0.0 keeps forward passes deterministic (no stochastic masking).
    """
    return GPTConfig(block_size=8, n_layers=2, vocab_size=10, n_head=2, n_embed=16, dropout=0.0)


def _loss_fn(
    gpt: model.GPT,
    key: jax.Array,
    x_BxS: jax.Array,   # (B, S)  integer token ids
    y_BxS: jax.Array,   # (B, S)  next-token targets — x shifted left by one
    vocab_size: int,
) -> jax.Array:
    """
    Compute mean cross-entropy loss over a batch.

    Uses jax.vmap to run the model independently on each example in the batch,
    giving each its own PRNG key (required by equinox dropout layers).
    Flattens (B, S) → (B*S,) before computing the loss so that
    optax's cross-entropy function sees a 2-D logit matrix.

    This mirrors what the real training loop does in nanotron/train.py.
    """
    keys_B = jax.random.split(key, x_BxS.shape[0])         # one key per batch element
    logits_BxSxV = jax.vmap(gpt, in_axes=(0, 0))(keys_B, x_BxS)
    logits_BSxV = logits_BxSxV.reshape(-1, vocab_size)      # (B*S, V)
    y_BS = y_BxS.reshape(-1)                                 # (B*S,)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_BSxV, y_BS))


def test_loss_is_finite():
    """
    A freshly initialised model must produce a finite loss on the first forward
    pass.  NaN here usually means exploding activations from bad weight init.
    """
    key = jax.random.PRNGKey(0)
    cfg = _small_config()
    gpt = model.GPT(key, cfg)
    x_BxS = jnp.array([[1, 2, 3, 4, 5, 6, 7, 0]])   # (B=1, S=8)
    y_BxS = jnp.array([[2, 3, 4, 5, 6, 7, 0, 1]])
    loss = _loss_fn(gpt, key, x_BxS, y_BxS, cfg.vocab_size)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"


def test_no_nan_gradients():
    """
    Every parameter gradient must be finite after the first backward pass.
    NaN gradients are the most common symptom of:
      - exploding activations (fix: lower lr, add gradient clipping)
      - log(0) in softmax (fix: add epsilon or use stable cross-entropy)
      - incorrect masking producing -inf logits everywhere

    eqx.filter(..., eqx.is_inexact_array) strips non-differentiable leaves
    (integers, booleans, static fields) so we only inspect float arrays.
    """
    key = jax.random.PRNGKey(0)
    cfg = _small_config()
    gpt = model.GPT(key, cfg)
    x_BxS = jnp.array([[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1]])  # (B=2, S=8)
    y_BxS = jnp.array([[2, 3, 4, 5, 6, 7, 0, 1], [3, 4, 5, 6, 7, 0, 1, 2]])

    grad_fn = eqx.filter_value_and_grad(_loss_fn)
    _, grads = grad_fn(gpt, key, x_BxS, y_BxS, cfg.vocab_size)

    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    for g in grad_leaves:  # g shape varies per parameter tensor
        assert jnp.all(jnp.isfinite(g)), f"NaN/Inf in gradients, shape={g.shape}"


def test_overfit_single_batch():
    """
    A model with sufficient capacity must be able to memorise a tiny fixed batch.
    We train for 100 steps on the same two examples and require the loss to drop
    by at least 50 % from its initial value.

    Why this matters: if loss doesn't drop, the model has a gradient flow problem
    (dead activations, wrong learning rate sign, disconnected graph, etc.).

    Hyperparameter choices:
      lr=1e-2   — aggressive but safe for a 10k-param model on a tiny batch
      steps=100 — enough to guarantee significant loss reduction without being slow
    """
    key = jax.random.PRNGKey(0)
    cfg = _small_config()
    gpt = model.GPT(key, cfg)

    # Two short sequences where y is x shifted left by one — the standard
    # next-token prediction objective.
    x_BxS = jnp.array([[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1]])  # (B=2, S=8)
    y_BxS = jnp.array([[2, 3, 4, 5, 6, 7, 0, 1], [3, 4, 5, 6, 7, 0, 1, 2]])

    optimizer = optax.adam(1e-2)
    # eqx.filter strips non-array leaves (e.g. Python ints stored as static fields)
    # before handing the parameter tree to optax.
    model_params = eqx.filter(gpt, eqx.is_inexact_array)
    opt_state = optimizer.init(model_params)
    grad_fn = eqx.filter_value_and_grad(_loss_fn)

    initial_loss = None
    for i in range(100):
        # jax.random.fold_in produces a unique key per step without consuming
        # the base key — the standard JAX pattern for stateless iteration.
        step_key = jax.random.fold_in(key, i)
        loss, grads = grad_fn(gpt, step_key, x_BxS, y_BxS, cfg.vocab_size)
        if i == 0:
            initial_loss = float(loss)
        model_params = eqx.filter(gpt, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, model_params)
        gpt = eqx.apply_updates(gpt, updates)

    final_loss = float(loss)
    assert final_loss < initial_loss * 0.5, (
        f"Model failed to overfit: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )


def test_golden_logits():
    """
    Regression test: the model's output logits must not change unexpectedly.

    How it works:
      - First run: the golden file doesn't exist yet, so we compute logits,
        save them to tests/golden/logits.npy, and skip with a message.
      - Subsequent runs: we load the saved logits and compare against the
        current model output with a tight tolerance (atol=1e-5).

    When to regenerate:
      Delete tests/golden/logits.npy whenever you make an intentional change
      to model architecture or weight initialisation.  The next test run will
      create a new golden file from the updated model.

    What this catches:
      Silent numerical regressions — e.g. a refactor that changes output values
      without breaking any shape assertions.
    """
    key = jax.random.PRNGKey(42)   # fixed seed so the model is reproducible
    cfg = _small_config()
    gpt = model.GPT(key, cfg)
    tokens_S = jnp.array([1, 2, 3, 4])          # (S=4,)
    logits_SxV = gpt(key, tokens_S)              # (S, V)

    if not os.path.exists(GOLDEN_FILE):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        np.save(GOLDEN_FILE, np.array(logits_SxV))
        pytest.skip("Golden file created — run tests again to validate")

    expected_SxV = jnp.array(np.load(GOLDEN_FILE))
    max_diff = float(jnp.max(jnp.abs(logits_SxV - expected_SxV)))
    assert jnp.allclose(logits_SxV, expected_SxV, atol=1e-5), (
        f"Logits differ from golden file. Max diff: {max_diff:.6f}. "
        "If this change is intentional, delete tests/golden/logits.npy to regenerate."
    )
