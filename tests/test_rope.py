"""
Tests for Rotary Position Embedding (RoPE) — nanotron/attention.py.

All tests in this file are automatically skipped until RoPE is implemented.
Once you add `compute_rope_freqs` and `apply_rope` to nanotron/attention.py,
these tests will activate and verify the key mathematical properties of RoPE.

Background — what RoPE does:
  Instead of adding a position vector to token embeddings (like learned positional
  embeddings), RoPE rotates the query and key vectors inside each attention head
  by a position-dependent angle.  The rotation is defined as a block-diagonal
  matrix of 2×2 rotation matrices, one per pair of dimensions in the head.

  For position m and dimension pair i, the rotation angle is:
      θ_{m,i} = m / (theta_base ^ (2i / head_dim))

  Key property: because rotation is linear, the dot product q_m · k_n depends
  only on the *relative* position (m - n), not on the absolute positions.  This
  makes RoPE a relative position encoding, which generalises better to sequence
  lengths not seen during training.

Expected API (to be implemented in nanotron/attention.py):
    compute_rope_freqs(seq_len, head_dim, theta=10000.0)
        → (cos_SxDh: Array[S, Dh],   # Dh = head_dim // 2
           sin_SxDh: Array[S, Dh])

    apply_rope(x_HxSxD: Array[H, S, D], cos_SxDh, sin_SxDh)
        → Array[H, S, D]
"""

import pytest
import jax
import jax.numpy as jnp


def _import_rope():
    """
    Try to import the RoPE functions.  If they don't exist yet, skip the calling
    test with an informative message instead of raising an ImportError.

    This lets the test file live in the repo before the implementation is done,
    so the tests act as a specification of the expected API.
    """
    try:
        from nanotron.attention import compute_rope_freqs, apply_rope

        return compute_rope_freqs, apply_rope
    except ImportError:
        pytest.skip(
            "RoPE not yet implemented (compute_rope_freqs / apply_rope missing from nanotron/attention.py)"
        )


def test_rope_output_shape():
    """
    apply_rope must return a tensor with the exact same shape as its input (H, S, D).
    RoPE only rotates values — it must not change the number of heads,
    the sequence length, or the head dimension.
    """
    compute_rope_freqs, apply_rope = _import_rope()
    seq_len, n_heads, head_dim = 8, 4, 16
    key = jax.random.PRNGKey(0)
    x_HxSxD = jax.random.normal(key, (n_heads, seq_len, head_dim))  # (H=4, S=8, D=16)
    cos_SxDh, sin_SxDh = compute_rope_freqs(seq_len, head_dim)  # Dh = head_dim // 2 = 8
    out_HxSxD = apply_rope(x_HxSxD, cos_SxDh, sin_SxDh)
    assert out_HxSxD.shape == x_HxSxD.shape


def test_rope_norm_preservation():
    """
    Rotation is an orthogonal transformation, so it must preserve the L2 norm
    of every vector.  Concretely: ||apply_rope(x)[h, t, :]||₂ == ||x[h, t, :]||₂
    for all heads h and positions t.

    Why this matters: if norms change, attention logits (q·k) will be scaled
    differently at different positions, introducing a spurious position-dependent
    bias that is *not* relative-position-invariant.
    """
    compute_rope_freqs, apply_rope = _import_rope()
    seq_len, n_heads, head_dim = 6, 2, 16
    key = jax.random.PRNGKey(1)
    x_HxSxD = jax.random.normal(key, (n_heads, seq_len, head_dim))  # (H=2, S=6, D=16)
    cos_SxDh, sin_SxDh = compute_rope_freqs(seq_len, head_dim)
    out_HxSxD = apply_rope(x_HxSxD, cos_SxDh, sin_SxDh)
    original_norms_HxS = jnp.linalg.norm(x_HxSxD, axis=-1)  # (H, S)
    rotated_norms_HxS = jnp.linalg.norm(out_HxSxD, axis=-1)
    assert jnp.allclose(original_norms_HxS, rotated_norms_HxS, atol=1e-5), (
        "RoPE changed vector norms — rotation must be norm-preserving"
    )


def test_rope_invertible():
    """
    A rotation matrix R satisfies R⁻¹ = Rᵀ (transpose = inverse).
    For the 2×2 rotation by angle θ:
        R(θ)  = [[cos θ,  -sin θ],
                 [sin θ,   cos θ]]
        R(-θ) = [[cos θ,   sin θ],
                 [-sin θ,  cos θ]]   ← same as Rᵀ

    So applying RoPE then passing -sin to apply_rope should recover x exactly.
    This test catches sign errors or wrong pairing of dimensions.
    """
    compute_rope_freqs, apply_rope = _import_rope()
    seq_len, n_heads, head_dim = 6, 2, 16
    key = jax.random.PRNGKey(2)
    x_HxSxD = jax.random.normal(key, (n_heads, seq_len, head_dim))  # (H=2, S=6, D=16)
    cos_SxDh, sin_SxDh = compute_rope_freqs(seq_len, head_dim)
    rotated_HxSxD = apply_rope(x_HxSxD, cos_SxDh, sin_SxDh)
    recovered_HxSxD = apply_rope(
        rotated_HxSxD, cos_SxDh, -sin_SxDh
    )  # -sin ↔ inverse rotation
    assert jnp.allclose(x_HxSxD, recovered_HxSxD, atol=1e-5), (
        "RoPE is not invertible — applying rotation then inverse should return original"
    )


def test_rope_relative_position():
    """
    The core guarantee of RoPE: the dot product q_m · k_n depends only on the
    relative offset (m - n), not on the absolute positions m and n.

    We verify this by checking that the dot product is the same for pairs with
    the same offset but different absolute positions:
        rotate(q, pos=0) · rotate(k, pos=1)   should equal
        rotate(q, pos=1) · rotate(k, pos=2)   and
        rotate(q, pos=3) · rotate(k, pos=4)

    All three pairs have offset = 1, so their dot products must agree.

    Why relative position matters: it allows the model to generalise to sequence
    lengths longer than those seen during training, because attention scores
    depend on how far apart tokens are, not where they appear in the absolute
    sequence.
    """
    compute_rope_freqs, apply_rope = _import_rope()
    head_dim = 16
    q_D = jax.random.normal(jax.random.PRNGKey(3), (head_dim,))  # (D=16,)
    k_D = jax.random.normal(jax.random.PRNGKey(4), (head_dim,))

    def dot_at_positions(pos_q: int, pos_k: int) -> float:
        """Rotate a single q and k vector at the given positions and return their dot product."""
        max_pos = max(pos_q, pos_k) + 1
        cos_SxDh, sin_SxDh = compute_rope_freqs(max_pos, head_dim)
        # Reshape to (H=1, S=1, D) so apply_rope accepts the expected 3-D input.
        q_rot_HxSxD = apply_rope(
            q_D[None, None, :], cos_SxDh[pos_q : pos_q + 1], sin_SxDh[pos_q : pos_q + 1]
        )
        k_rot_HxSxD = apply_rope(
            k_D[None, None, :], cos_SxDh[pos_k : pos_k + 1], sin_SxDh[pos_k : pos_k + 1]
        )
        return float(jnp.dot(q_rot_HxSxD[0, 0], k_rot_HxSxD[0, 0]))

    # All three pairs have relative offset = 1.
    dot_01 = dot_at_positions(0, 1)
    dot_12 = dot_at_positions(1, 2)
    dot_34 = dot_at_positions(3, 4)

    assert abs(dot_01 - dot_12) < 1e-4, (
        f"dot(q_0, k_1)={dot_01:.6f} != dot(q_1, k_2)={dot_12:.6f} — relative position not preserved"
    )
    assert abs(dot_01 - dot_34) < 1e-4, (
        f"dot(q_0, k_1)={dot_01:.6f} != dot(q_3, k_4)={dot_34:.6f} — relative position not preserved"
    )
