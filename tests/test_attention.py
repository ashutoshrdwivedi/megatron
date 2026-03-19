"""
Tests for the attention mechanism (nanotron/attention.py).

Covers three layers of correctness:
  1. scaled_dot_product — the raw mathematical operation
  2. expand_mask         — utility that broadcasts masks to 4D
  3. MultiHeadAttention  — the full module (projection → split heads → attend → merge)
"""

import jax
import jax.numpy as jnp

from nanotron import attention


def test_scaled_dot_product_shape():
    """Output shapes must match (H, S, D) for values and (H, Sq, Sk) for attention weights."""
    key = jax.random.PRNGKey(0)
    # Split into three independent keys so q, k, v are uncorrelated random matrices.
    # JAX requires explicit key management — reusing the same key gives identical arrays.
    key_q, key_k, key_v = jax.random.split(key, 3)
    q_SxD = jax.random.normal(key_q, (2, 4))  # (S=2, D=4)
    k_SxD = jax.random.normal(key_k, (2, 4))
    v_SxD = jax.random.normal(key_v, (2, 4))
    # Add a leading head dimension (H=1) so the function sees a 3-D input.
    values_HxSxD, attn_HxSxS = attention.scaled_dot_product(
        q_SxD[None, :], k_SxD[None, :], v_SxD[None, :]
    )
    assert values_HxSxD.shape == (1, 2, 4)  # (H, S, D)
    assert attn_HxSxS.shape == (1, 2, 2)  # (H, Sq, Sk)


def test_attention_rows_sum_to_one():
    """
    Attention weights are produced by softmax, so each query's weights over all
    keys must sum to exactly 1.0.  Failure here means the softmax normalisation
    is broken or the mask is zeroing everything out.
    """
    key = jax.random.PRNGKey(0)
    q_HxSxD = jax.random.normal(key, (1, 4, 8))  # (H=1, S=4, D=8)
    k_HxSxD = jax.random.normal(key, (1, 4, 8))
    v_HxSxD = jax.random.normal(key, (1, 4, 8))
    _, attn_HxSxS = attention.scaled_dot_product(q_HxSxD, k_HxSxD, v_HxSxD)
    row_sums_HxS = attn_HxSxS.sum(axis=-1)  # sum over key dimension → (H, S)
    assert jnp.allclose(row_sums_HxS, jnp.ones_like(row_sums_HxS), atol=1e-5)


def test_causal_mask_strict():
    """
    A causal (autoregressive) model must never let token i attend to token j > i.
    We pass a lower-triangular mask and assert that every entry in the upper
    triangle of the attention weight matrix is exactly 0.

    Why this matters: if future tokens leak into the attention weights, the model
    sees information it shouldn't have during training, making loss artificially
    low and causing garbage at inference time.
    """
    key = jax.random.PRNGKey(0)
    seq_len, n_embed, n_heads = 6, 8, 2
    mha = attention.MultiHeadAttention(key, n_embed=n_embed, n_heads=n_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))  # (S=6, E=8)

    # Lower triangular → token i can only attend to positions 0..i (inclusive).
    causal_mask_SxS = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    _, attn_HxSxS = mha(x_SxE, mask=causal_mask_SxS)  # attn: (H, S, S)

    # The upper triangle (j > i) is the complement of the causal mask.
    upper_SxS = (
        ~causal_mask_SxS
    )  # equivalent to triu(..., k=1), but derived from the same mask
    assert jnp.all(attn_HxSxS[:, upper_SxS] == 0.0), (
        "Future tokens have non-zero attention weight"
    )


def test_attention_deterministic():
    """
    MultiHeadAttention has no dropout or stochastic ops by default, so calling
    it twice with the same input must return bit-identical results.

    Flaky attention (e.g. due to unintentional randomness in a weight init path)
    would make losses non-reproducible and complicate debugging.
    """
    key = jax.random.PRNGKey(1)
    mha = attention.MultiHeadAttention(key, n_embed=8, n_heads=2)
    x_SxE = jax.random.normal(key, (4, 8))  # (S=4, E=8)
    out1_SxE, attn1_HxSxS = mha(x_SxE)
    out2_SxE, attn2_HxSxS = mha(x_SxE)
    assert jnp.allclose(out1_SxE, out2_SxE)
    assert jnp.allclose(attn1_HxSxS, attn2_HxSxS)


def test_expand_mask():
    """
    expand_mask must broadcast a 2-D mask up to 4-D (B, H, Sq, Sk)
    so it can be applied uniformly across all heads in MultiHeadAttention.
    """
    mask_SqxSk = jnp.ones((2, 3))  # (Sq=2, Sk=3)
    out_BxHxSqxSk = attention.expand_mask(mask_SqxSk)
    assert out_BxHxSqxSk.ndim == 4
    assert out_BxHxSqxSk.shape[-2:] == (2, 3)  # spatial dims must be preserved


def test_multi_head_attention_output_shape():
    """
    The module must return:
      - output embeddings of shape (S, E)  — same shape as input
      - attention weights of shape (H, S, S)
    """
    key = jax.random.PRNGKey(0)
    seq_len, n_embed, n_heads = 3, 8, 2
    mha = attention.MultiHeadAttention(key, n_embed=n_embed, n_heads=n_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))  # (S=3, E=8)
    values_SxE, attn_HxSxS = mha(x_SxE)
    assert values_SxE.shape == (seq_len, n_embed)  # (S, E)
    assert attn_HxSxS.shape == (n_heads, seq_len, seq_len)  # (H, S, S)


# ── Grouped Query Attention (GQA) ─────────────────────────────────────────────


def test_gqa_output_shape():
    """
    GQA must return the same shapes as MHA:
      - output embeddings: (S, E)  — identical to input shape
      - attention weights: (Hq, S, S)  — one weight matrix per query head

    Using 4 query heads and 2 KV heads (group size = 2).
    """
    key = jax.random.PRNGKey(0)
    seq_len, n_embed, n_heads, n_kv_heads = 5, 8, 4, 2
    gqa = attention.GroupedQueryAttention(
        key, n_embed=n_embed, n_heads=n_heads, n_kv_heads=n_kv_heads
    )
    x_SxE = jax.random.normal(key, (seq_len, n_embed))
    out_SxE, attn_HxSxS = gqa(x_SxE)
    assert out_SxE.shape == (seq_len, n_embed)
    assert attn_HxSxS.shape == (n_heads, seq_len, seq_len)


def test_gqa_attention_rows_sum_to_one():
    """
    Each query's attention weights over all keys must sum to 1.0 (softmax normalisation).
    Checked across all query heads.
    """
    key = jax.random.PRNGKey(1)
    gqa = attention.GroupedQueryAttention(key, n_embed=8, n_heads=4, n_kv_heads=2)
    x_SxE = jax.random.normal(key, (6, 8))
    _, attn_HxSxS = gqa(x_SxE)
    row_sums = attn_HxSxS.sum(axis=-1)  # (Hq, S)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)


def test_gqa_causal_mask():
    """
    With a lower-triangular causal mask, every entry in the upper triangle of
    each query head's attention matrix must be exactly 0.
    """
    key = jax.random.PRNGKey(2)
    seq_len, n_embed, n_heads, n_kv_heads = 6, 8, 4, 2
    gqa = attention.GroupedQueryAttention(
        key, n_embed=n_embed, n_heads=n_heads, n_kv_heads=n_kv_heads
    )
    x_SxE = jax.random.normal(key, (seq_len, n_embed))
    causal_mask_SxS = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    _, attn_HxSxS = gqa(x_SxE, mask=causal_mask_SxS)
    upper_SxS = ~causal_mask_SxS
    assert jnp.all(attn_HxSxS[:, upper_SxS] == 0.0), (
        "Future tokens have non-zero attention weight in GQA"
    )


def test_gqa_deterministic():
    """
    GQA has no stochastic ops, so identical inputs must produce bit-identical outputs.
    """
    key = jax.random.PRNGKey(3)
    gqa = attention.GroupedQueryAttention(key, n_embed=8, n_heads=4, n_kv_heads=2)
    x_SxE = jax.random.normal(key, (4, 8))
    out1, attn1 = gqa(x_SxE)
    out2, attn2 = gqa(x_SxE)
    assert jnp.allclose(out1, out2)
    assert jnp.allclose(attn1, attn2)


def test_gqa_mqa_is_special_case():
    """
    Multi-Query Attention (MQA) is GQA with n_kv_heads=1. All query heads share
    a single K and V head. Output shape must still be (S, E) with attention (Hq, S, S).
    """
    key = jax.random.PRNGKey(4)
    seq_len, n_embed, n_heads = 5, 8, 4
    mqa = attention.GroupedQueryAttention(
        key, n_embed=n_embed, n_heads=n_heads, n_kv_heads=1
    )
    x_SxE = jax.random.normal(key, (seq_len, n_embed))
    out_SxE, attn_HxSxS = mqa(x_SxE)
    assert out_SxE.shape == (seq_len, n_embed)
    assert attn_HxSxS.shape == (n_heads, seq_len, seq_len)
