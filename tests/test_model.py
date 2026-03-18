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

from typing import Generator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from nanotron import model
from nanotron.config import GPTConfig
from nanotron.train import get_optimizers, step


def _small_config() -> GPTConfig:
    """
    Slightly larger than test_gpt.py's config so training tests have enough
    capacity to actually overfit a tiny batch.

    n_layers=2, n_embed=16 gives ~10k parameters — small enough to train in
    seconds on CPU but expressive enough that loss can drop significantly.
    dropout=0.0 keeps forward passes deterministic (no stochastic masking).
    """
    return GPTConfig(
        block_size=8, n_layers=2, vocab_size=10, n_head=2, n_embed=16, dropout=0.0
    )


def _loss_fn(
    gpt: model.GPT,
    key: jax.Array,
    x_BxS: jax.Array,  # (B, S)  integer token ids
    y_BxS: jax.Array,  # (B, S)  next-token targets — x shifted left by one
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
    keys_B = jax.random.split(key, x_BxS.shape[0])  # one key per batch element
    logits_BxSxV = jax.vmap(gpt, in_axes=(0, 0))(keys_B, x_BxS)
    logits_BSxV = logits_BxSxV.reshape(-1, vocab_size)  # (B*S, V)
    y_BS = y_BxS.reshape(-1)  # (B*S,)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_BSxV, y_BS))


def test_loss_is_finite():
    """
    A freshly initialised model must produce a finite loss on the first forward
    pass.  NaN here usually means exploding activations from bad weight init.
    """
    key = jax.random.PRNGKey(0)
    cfg = _small_config()
    gpt = model.GPT(key, cfg)
    x_BxS = jnp.array([[1, 2, 3, 4, 5, 6, 7, 0]])  # (B=1, S=8)
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
    x_BxS = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1]]
    )  # (B=2, S=8)
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
    x_BxS = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1]]
    )  # (B=2, S=8)
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

    Uses three scalar checksums (sum, min, max) of the logit matrix computed
    from a fixed seed.  Scalars are hardcoded directly in the test — no files,
    no binary blobs, diffs are human-readable.

    When to update:
      If you make an intentional architectural change, re-run the snippet below
      to get the new values and update the three expected constants here:

        JAX_PLATFORM_NAME=cpu python -c "
        import os; os.environ['OUT_DIR'] = '/tmp/test'
        import jax, jax.numpy as jnp
        from nanotron import model
        from nanotron.config import GPTConfig
        cfg = GPTConfig(block_size=8, n_layers=2, vocab_size=10, n_head=2, n_embed=16, dropout=0.0)
        key = jax.random.PRNGKey(42)
        logits = model.GPT(key, cfg)(key, jnp.array([1, 2, 3, 4]))
        print('sum =', round(float(jnp.sum(logits)), 4))
        print('min =', round(float(jnp.min(logits)), 4))
        print('max =', round(float(jnp.max(logits)), 4))
        "

    What this catches:
      Silent numerical regressions — e.g. a refactor that changes output values
      without breaking any shape assertions.
    """
    key = jax.random.PRNGKey(42)  # fixed seed — must match the snippet above
    cfg = _small_config()
    gpt = model.GPT(key, cfg)
    tokens_S = jnp.array([1, 2, 3, 4])  # (S=4,)
    logits_SxV = gpt(key, tokens_S)  # (S, V)

    # Three independent checksums reduce the chance that cancelling shifts go undetected.
    actual_sum = round(float(jnp.sum(logits_SxV)), 4)
    actual_min = round(float(jnp.min(logits_SxV)), 4)
    actual_max = round(float(jnp.max(logits_SxV)), 4)

    assert actual_sum == 2.0373, f"logits sum changed: {actual_sum} (expected 2.0373)"
    assert actual_min == -1.5516, f"logits min changed: {actual_min} (expected -1.5516)"
    assert actual_max == 1.0738, f"logits max changed: {actual_max} (expected  1.0738)"


def _synthetic_dataloader(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
    """
    Yield random (x, y) token batches without touching any real dataset.

    Generates a fixed corpus of random tokens once, then samples overlapping
    windows from it — the same strategy as the real dataloader in data.py,
    but fully in-memory and dependency-free.

    x_BxS: token ids at positions [i .. i+seq_len)
    y_BxS: token ids at positions [i+1 .. i+seq_len+1)  (next-token targets)
    """
    corpus_len = batch_size * seq_len * 4  # large enough to sample from
    key, corpus_key = jax.random.split(key)
    corpus = jax.random.randint(
        corpus_key, (corpus_len,), 0, vocab_size, dtype=jnp.int32
    )

    while True:
        key, idx_key = jax.random.split(key)
        start_indices = jax.random.randint(
            idx_key, (batch_size,), 0, corpus_len - seq_len - 1
        )
        arange_S = jnp.arange(seq_len)
        idx_BxS = start_indices[:, None] + arange_S[None, :]  # (B, S)
        x_BxS = jnp.take(corpus, idx_BxS)
        y_BxS = jnp.take(corpus, idx_BxS + 1)
        yield x_BxS, y_BxS


def test_training_loop():
    """
    End-to-end smoke test for the training loop (train.py's `step` function).

    Runs 20 steps on a tiny synthetic in-memory dataset — no real dataset
    download required.  Verifies two properties:

      1. Every step produces a finite loss (NaN/Inf would indicate a broken
         forward or backward pass).
      2. The final loss is lower than the initial loss (the model actually
         learns something, ruling out disconnected gradients or a broken
         optimizer).

    Uses `get_optimizers` from train.py so the weight-decay partitioning and
    gradient clipping paths are exercised, not just the optax primitives.

    Design choices:
      steps=20  — enough to see a clear loss drop without being slow on CPU
      lr=1e-2   — aggressive but safe for a 10k-param model
      B=4, S=8  — minimal batch to keep compile + runtime fast
    """
    NUM_STEPS = 20
    BATCH_SIZE = 4
    key = jax.random.PRNGKey(7)
    cfg = _small_config()

    key, model_key = jax.random.split(key)
    gpt = model.GPT(model_key, cfg)

    model_params = eqx.filter(gpt, eqx.is_inexact_array)
    optimizer = get_optimizers(
        model_params,
        weight_decay=1e-1,
        learning_rate=1e-2,
        betas=(0.9, 0.99),
    )
    opt_state = optimizer.init(model_params)

    key, data_key = jax.random.split(key)
    dataloader = _synthetic_dataloader(
        data_key, BATCH_SIZE, cfg.block_size, cfg.vocab_size
    )

    losses = []
    for i in range(NUM_STEPS):
        key, step_key = jax.random.split(key)
        batch = next(dataloader)
        gpt, opt_state, loss = step(step_key, gpt, optimizer, opt_state, batch)
        loss_val = float(loss)
        assert jnp.isfinite(loss), f"Non-finite loss at step {i}: {loss_val}"
        losses.append(loss_val)

    assert losses[-1] < losses[0], (
        f"Loss did not decrease over {NUM_STEPS} steps: "
        f"initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )
