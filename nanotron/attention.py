from __future__ import annotations

import einops
import equinox as eqx
import jax
import math

from typing import Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def compute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[
    Float[Array, "seq_len head_dim_half"], Float[Array, "seq_len head_dim_half"]
]:
    """
    Precomputes the cosine and sine rotation frequencies for Rotary Position Embeddings (RoPE).

    Conceptually, this function determines the rotation speed for every 2D "dial" (subspace)
    in the model's attention mechanism. It generates a geometric sequence of frequencies:
        * Early dimension pairs (i near 0) have high frequencies, acting as fast-spinning
          dials that track immediate, local grammar.
        * Later dimension pairs (i near head_dim // 2) have exponentially lower frequencies,
          acting as slow-spinning dials that track long-distance context.

    The absolute baseline limit of the model's context window is fundamentally tied to when
    the slowest dial completes a full 360-degree rotation.

    For each sequence position 'm' and dimension pair 'i', the rotation angle is calculated as:
        angle[m, i] = m * (1 / (theta ^ (2i / head_dim)))

    Args:
        seq_len: The maximum sequence length (number of text positions) to precompute angles for.
        head_dim: The total number of dimensions in each attention head. Must be an even number
            so it can be perfectly split into 2D subspaces.
        theta: The base frequency constant. The original RoPE paper defaults to 10000.0.
            Modifying this value (e.g., increasing to 500000.0) is a common RoPE scaling
            technique to stretch the context window for longer documents.

    Returns:
        A tuple containing two arrays:
        - cos_SxDh: Precomputed cosine values for the rotation angles. Shape: (seq_len, head_dim // 2).
        - sin_SxDh: Precomputed sine values for the rotation angles. Shape: (seq_len, head_dim // 2).
    """
    n_subspaces = head_dim // 2
    # Dimension indices 0, 1, ..., n_subspaces - 1
    i = jnp.arange(n_subspaces)
    # freqs[i] = 1 / theta^(2i / head_dim) — one frequency per dimension pair
    freqs = 1.0 / (theta ** (2 * i / head_dim))  # (n_subspaces,)
    positions = jnp.arange(seq_len)  # (seq_len,)
    angles = jnp.outer(positions, freqs)  # (seq_len, n_subspaces)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x_HxSxD: Float[Array, "n_heads seq_len head_dim"],
    cos_SxDh: Float[Array, "seq_len head_dim_half"],
    sin_SxDh: Float[Array, "seq_len head_dim_half"],
) -> Float[Array, "n_heads seq_len head_dim"]:
    """
    Rotates query or key vectors using precomputed RoPE frequencies.

    Conceptually, RoPE isolates pairs of dimensions into isolated 2D subspaces and
    rotates them to encode relative sequence position. Because the rotation only
    changes direction and preserves the vector's magnitude, the word's underlying
    semantic meaning remains intact.

    Implementation Details:
    For hardware memory efficiency, this implementation does not pair adjacent
    dimensions (e.g., x[0] with x[1]). Instead, it splits the head dimension in
    half, pairing x[i] with x[i + head_dim_half]. The neural network naturally
    learns to encode its 2D features across this split structure during training.

    Treating each pair as a complex number (x1 + i*x2), the rotation by angle θ
    is applied via complex multiplication (where i^2 = -1 results in the minus sign):
        New x1 (Real):      x1·cos(θ) - x2·sin(θ)
        New x2 (Imaginary): x1·sin(θ) + x2·cos(θ)

    Broadcasting:
    The cos and sin tensors of shape (seq_len, head_dim_half) are automatically
    broadcast against the full attention tensor.

    Args:
        x_HxSxD: The input query (Q) or key (K) tensor to be rotated.
            Expected shape: (..., n_heads, seq_len, head_dim).
        cos_SxDh: The precomputed cosine values for the rotation angles.
            Expected shape: (..., seq_len, head_dim_half).
        sin_SxDh: The precomputed sine values for the rotation angles.
            Expected shape: (..., seq_len, head_dim_half).

    Returns:
        The newly rotated query or key tensor. It maintains the exact same shape
        and magnitude as the input tensor `x_HxSxD`, now embedded with positional
        context. Shape: (..., n_heads, seq_len, head_dim).
    """
    head_dim = x_HxSxD.shape[-1]

    # Split the dimensions directly in half for hardware contiguous memory
    x1_HxSxDh = x_HxSxD[..., : head_dim // 2]  # (H, S, Dh)
    x2_HxSxDh = x_HxSxD[..., head_dim // 2 :]  # (H, S, Dh)

    rotated_x1_HxSxDh = x1_HxSxDh * cos_SxDh - x2_HxSxDh * sin_SxDh  # broadcasting
    rotated_x2_HxSxDh = x1_HxSxDh * sin_SxDh + x2_HxSxDh * cos_SxDh  # broadcasting

    # Stitch the halves back together
    return jnp.concatenate([rotated_x1_HxSxDh, rotated_x2_HxSxDh], axis=-1)  # (H, S, D)


def scaled_dot_product(
    q: Float[Array, "... dims seq_len"],
    k: Float[Array, "... dims seq_len"],
    v: Float[Array, "... dims seq_len"],
    mask: Optional[Array] = None,
) -> Tuple[Float[Array, "... seq_q dims"], Float[Array, "... seq_q seq_k"]]:
    d_k = q.shape[-1]
    logits = einops.einsum(q, k, "... seq_q dims, ... seq_k dims -> ... seq_q seq_k")
    attn_logits = logits / math.sqrt(d_k)
    if mask is not None:
        logits = jnp.where(mask == 0, jnp.finfo(logits.dtype).min, attn_logits)
    else:
        logits = attn_logits
    attention = jax.nn.softmax(logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask: Array) -> Array:
    assert mask.ndim >= 2, "mask must be atleast 2 dim"
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, axis=0)
    return mask


class MultiHeadAttention(eqx.Module):
    """Given initial embeddings, get q k v, apply attention, and output projection"""

    n_embed: int
    n_heads: int
    rope_theta: float
    qkv_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(
        self, key: PRNGKeyArray, n_embed: int, n_heads: int, rope_theta: float = 10000.0
    ) -> None:
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.rope_theta = rope_theta

        key_qkv, key_proj = jax.random.split(key, 2)

        # TODO: is bias initialization and kernel init with xavier required?
        qkv_out_size = 3 * self.n_embed
        self.qkv_proj = eqx.nn.Linear(
            in_features=self.n_embed,
            out_features=qkv_out_size,
            key=key_qkv,
            use_bias=True,
        )
        self.output_proj = eqx.nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_embed,
            use_bias=True,
            key=key_proj,
        )

    def __call__(
        self, x: Float[Array, "seq_len n_embed"], mask: Optional[Array] = None
    ) -> Tuple[
        Float[Array, "seq_len n_embed"], Float[Array, "n_heads seq_len seq_len"]
    ]:
        seq_len, n_embed = x.shape
        head_dim = n_embed // self.n_heads

        # a single projection layer, given the input produces Q, K, V matrices
        qkv = jax.vmap(self.qkv_proj)(x)

        # The scaled dot product attention allows a network to attend over a sequence.
        # However, often there are multiple different aspects a sequence element
        # wants to attend to, and a single weighted average is not a good option for it.
        # This is why we extend the attention mechanisms to multiple heads,
        # i.e. multiple different query-key-value triplets on the same features.
        # Specifically, given a query, key, and value matrix, we transform those into sub-queries,
        # sub-keys, and sub-values, which we pass through the scaled dot product attention independently.
        # Afterward, we concatenate the heads and combine them with a final weight matrix

        # split the embeding_dim into multiple heads
        # dim here is different from embed_dim, it's 3 * embed_dims
        reshaped_qkv = einops.rearrange(
            qkv,
            "seq_len (n_heads d) -> n_heads seq_len d",
            seq_len=seq_len,
            n_heads=self.n_heads,
        )
        # embedding dims contains all of qkv, so split
        q_HxSxD, k_HxSxD, v_HxSxD = jnp.array_split(reshaped_qkv, 3, axis=-1)

        # Apply RoPE to queries and keys — not values.
        # RoPE encodes relative position by rotating q and k so that q_m · k_n
        # depends only on (m - n).  Values carry content, not position, so they
        # are left unrotated.
        cos_SxDh, sin_SxDh = compute_rope_freqs(seq_len, head_dim, self.rope_theta)
        q_HxSxD = apply_rope(q_HxSxD, cos_SxDh, sin_SxDh)
        k_HxSxD = apply_rope(k_HxSxD, cos_SxDh, sin_SxDh)

        values, attention = scaled_dot_product(q_HxSxD, k_HxSxD, v_HxSxD, mask=mask)
        values = einops.rearrange(
            values,
            "n_heads seq_len d -> seq_len (n_heads d)",
            n_heads=self.n_heads,
            seq_len=seq_len,
        )
        output_embeddings = jax.vmap(self.output_proj)(values)
        return output_embeddings, attention
