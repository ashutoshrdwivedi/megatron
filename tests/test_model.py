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

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from nanotron import model
from nanotron.config import GPTConfig


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

    assert actual_sum == -9.9176, f"logits sum changed: {actual_sum} (expected -9.9176)"
    assert actual_min == -1.4068, f"logits min changed: {actual_min} (expected -1.4068)"
    assert actual_max == 0.6481, f"logits max changed: {actual_max} (expected  0.6481)"
