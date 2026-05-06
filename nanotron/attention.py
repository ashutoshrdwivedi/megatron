from __future__ import annotations

import einops
import equinox as eqx
import jax
import math

from typing import NamedTuple, Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class KVCache(NamedTuple):
    """
    Pre-allocated cache of Key and Value tensors for a single attention layer.

    The fundamental bottleneck in autoregressive generation is that every new
    token requires re-computing K and V for *all* past tokens.  KV caching
    solves this by projecting each token's K and V once and storing them; on
    every subsequent step we just look them up.

    Complexity impact for generating T new tokens after a prompt of length P:
        Without cache:  O((P + T)²)  — full attention is recomputed at each step
        With    cache:  O(P² + T·P)  — quadratic only in the prompt (prefill),
                                       then linear in new tokens (decode)

    Layout
    ------
    Both arrays are *pre-allocated* to `max_seq_len` and zero-initialised.
    Only positions [0 .. cur_pos) are ever populated; the rest stay zero.
    A position-aware causal mask prevents attention from bleeding into those
    uninitialised zero slots.

        k_KVHxSxD : (n_kv_heads, max_seq_len, head_dim)  — cached key vectors
        v_KVHxSxD : (n_kv_heads, max_seq_len, head_dim)  — cached value vectors

    n_kv_heads equals n_heads for MHA, but can be much smaller for GQA/MQA —
    which is the whole memory-bandwidth motivation for grouped query attention.
    The cache always stores the *un-expanded* KV heads (before the GQA repeat).

    NamedTuple is used (over a plain dataclass) because JAX automatically
    treats NamedTuples as PyTrees, making the cache compatible with jit and
    lax.scan without any extra registration boilerplate.
    """

    k_KVHxSxD: Float[Array, "n_kv_heads max_seq_len head_dim"]
    v_KVHxSxD: Float[Array, "n_kv_heads max_seq_len head_dim"]


def compute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    start_pos: int = 0,
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

    KV-cache note: during autoregressive decoding, new tokens arrive at absolute positions
    [start_pos, start_pos+1, ..., start_pos+seq_len-1] rather than always starting at 0.
    Passing start_pos ensures the Q and K vectors for those tokens are rotated by the correct
    absolute angles, so that Q·K dot products still encode the right *relative* distance.

    Args:
        seq_len:   Number of positions to generate angles for.
        head_dim:  Total dimensions per attention head (must be even).
        theta:     Base frequency constant (default 10000.0 from the original paper).
        start_pos: Absolute position of the first token in this window (default 0).
                   During normal training this is always 0.  During KV-cache decoding
                   it equals the number of tokens already in the cache.

    Returns:
        A tuple containing two arrays:
        - cos_SxDh: Precomputed cosine values. Shape: (seq_len, head_dim // 2).
        - sin_SxDh: Precomputed sine values.  Shape: (seq_len, head_dim // 2).
    """
    n_subspaces = head_dim // 2
    # Dimension indices 0, 1, ..., n_subspaces - 1
    i = jnp.arange(n_subspaces)
    # freqs[i] = 1 / theta^(2i / head_dim) — one frequency per dimension pair
    freqs = 1.0 / (theta ** (2 * i / head_dim))  # (n_subspaces,)
    positions = jnp.arange(start_pos, start_pos + seq_len)  # (seq_len,)
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


class GroupedQueryAttention(eqx.Module):
    """
    Grouped Query Attention (GQA) from "GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023).
    https://arxiv.org/abs/2305.13245

    In standard MHA every query head owns its own K and V head, leading to
    memory bandwidth that scales with n_heads. GQA reduces that cost by letting
    a *group* of query heads share a single K/V head, so bandwidth scales with
    the much smaller n_kv_heads instead.

    Special cases that collapse to known variants:
        n_kv_heads == n_heads  →  standard Multi-Head Attention (MHA)
        n_kv_heads == 1        →  Multi-Query Attention (MQA)

    Projection layout:
        Q  projection:  n_embed → n_heads    × head_dim
        KV projection:  n_embed → n_kv_heads × head_dim  (×2 for K and V)

    After projection, each KV head is *repeated* (not re-projected) n_groups
    times so that every query head has a matching K and V to attend against.
    n_groups = n_heads // n_kv_heads.
    """

    n_embed: int
    n_heads: int
    n_kv_heads: int
    rope_theta: float
    q_proj: eqx.nn.Linear
    kv_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        n_embed: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float = 10000.0,
    ) -> None:
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta

        head_dim = n_embed // n_heads
        key_q, key_kv, key_proj = jax.random.split(key, 3)

        # Queries use all n_heads; keys and values share n_kv_heads
        self.q_proj = eqx.nn.Linear(
            in_features=n_embed,
            out_features=n_heads * head_dim,
            use_bias=True,
            key=key_q,
        )
        self.kv_proj = eqx.nn.Linear(
            in_features=n_embed,
            out_features=2 * n_kv_heads * head_dim,
            use_bias=True,
            key=key_kv,
        )
        self.output_proj = eqx.nn.Linear(
            in_features=n_embed,
            out_features=n_embed,
            use_bias=True,
            key=key_proj,
        )

    def __call__(
        self,
        x_SxE: Float[Array, "seq_len n_embed"],
        mask: Optional[Array] = None,
    ) -> Tuple[
        Float[Array, "seq_len n_embed"],
        Float[Array, "n_heads seq_len seq_len"],
    ]:
        seq_len, _ = x_SxE.shape
        head_dim = self.n_embed // self.n_heads
        n_groups = self.n_heads // self.n_kv_heads

        # ── Project ──────────────────────────────────────────────────────────
        q_SxHD = jax.vmap(self.q_proj)(x_SxE)  # (S, n_heads * head_dim)
        kv_SxKD = jax.vmap(self.kv_proj)(x_SxE)  # (S, 2 * n_kv_heads * head_dim)

        # ── Reshape into per-head tensors ─────────────────────────────────────
        q_HxSxD = einops.rearrange(
            q_SxHD, "s (h d) -> h s d", h=self.n_heads, d=head_dim
        )  # (n_heads, S, D)

        # kv=2 splits the last axis evenly into K and V
        kv_2xKVHxSxD = einops.rearrange(
            kv_SxKD, "s (kv h d) -> kv h s d", kv=2, h=self.n_kv_heads, d=head_dim
        )  # (2, n_kv_heads, S, D)
        k_KVHxSxD, v_KVHxSxD = kv_2xKVHxSxD[0], kv_2xKVHxSxD[1]

        # ── Expand each KV head to cover its group of query heads ─────────────
        # (n_kv_heads, S, D) → (n_heads, S, D)  by repeating n_groups times
        k_HxSxD = jnp.repeat(k_KVHxSxD, n_groups, axis=0)
        v_HxSxD = jnp.repeat(v_KVHxSxD, n_groups, axis=0)

        # ── Apply RoPE to queries and keys (not values) ───────────────────────
        cos_SxDh, sin_SxDh = compute_rope_freqs(seq_len, head_dim, self.rope_theta)
        q_HxSxD = apply_rope(q_HxSxD, cos_SxDh, sin_SxDh)
        k_HxSxD = apply_rope(k_HxSxD, cos_SxDh, sin_SxDh)

        # ── Attention + merge heads ───────────────────────────────────────────
        values_HxSxD, attention_HxSxS = scaled_dot_product(
            q_HxSxD, k_HxSxD, v_HxSxD, mask=mask
        )
        values_SxE = einops.rearrange(values_HxSxD, "h s d -> s (h d)", h=self.n_heads)
        output_SxE = jax.vmap(self.output_proj)(values_SxE)
        return output_SxE, attention_HxSxS

    def forward_with_cache(
        self,
        x_SxE: Float[Array, "seq_len n_embed"],
        cache: KVCache,
        start_pos: int,
    ) -> Tuple[Float[Array, "seq_len n_embed"], KVCache]:
        """
        Incremental attention forward pass using a KV cache.

        The two phases of cached generation
        ------------------------------------
        Prefill  (start_pos=0, seq_len=prompt_length):
            Process all prompt tokens at once — identical to a normal forward
            pass with a causal mask, but also populates the cache as a side
            effect.  Token i attends to tokens 0..i (causality preserved).

        Decode  (start_pos=prompt_length+t, seq_len=1):
            Process a single newly-generated token at step t.  The query is
            just that one token's projected embedding; the keys and values span
            the entire context (prompt + all t previously generated tokens)
            and are read directly from the cache.
            Cost per step: O(context_len) instead of O(context_len²).

        Cache write
        -----------
        New K and V are inserted at positions [start_pos .. start_pos+S) using
        jax.lax.dynamic_update_slice.  Unlike Python/NumPy slice assignment,
        dynamic_update_slice accepts *traced* (runtime-computed) indices and is
        safe inside jit and lax.scan.

        Causal mask
        -----------
        The mask has shape (S, max_seq_len) — asymmetric because the query
        window S is short (1 during decode) while keys span the full cache.
            mask[i, j] = True  iff  (start_pos + i) >= j

        Two properties fall out automatically:
          1. Causality — future tokens are blocked (j > start_pos+i → False).
          2. Uninitialised slots — positions j >= start_pos+S can never satisfy
             j <= start_pos+(S-1), so they are always masked out.  Their zero
             K/V values never contribute to the output.

        Args:
            x_SxE:     Input embeddings for the current window.
                       Shape (seq_len, n_embed).  seq_len=1 during decode.
            cache:     KVCache holding pre-allocated k/v buffers for this layer.
            start_pos: Absolute position of x_SxE[0] in the full sequence.
                       0 during prefill; increments by seq_len each decode step.

        Returns:
            output_SxE: Updated token embeddings, shape (seq_len, n_embed).
            new_cache:  KVCache with current window's k/v written in.
        """
        seq_len, _ = x_SxE.shape
        head_dim = self.n_embed // self.n_heads
        n_groups = self.n_heads // self.n_kv_heads
        max_seq_len = cache.k_KVHxSxD.shape[1]

        # ── Project ──────────────────────────────────────────────────────────
        q_SxHD = jax.vmap(self.q_proj)(x_SxE)  # (S, n_heads * head_dim)
        kv_SxKD = jax.vmap(self.kv_proj)(x_SxE)  # (S, 2 * n_kv_heads * head_dim)

        # ── Reshape into per-head tensors ─────────────────────────────────────
        q_HxSxD = einops.rearrange(
            q_SxHD, "s (h d) -> h s d", h=self.n_heads, d=head_dim
        )  # (n_heads, S, D)

        kv_2xKVHxSxD = einops.rearrange(
            kv_SxKD, "s (kv h d) -> kv h s d", kv=2, h=self.n_kv_heads, d=head_dim
        )  # (2, n_kv_heads, S, D)
        k_KVHxSxD, v_KVHxSxD = kv_2xKVHxSxD[0], kv_2xKVHxSxD[1]

        # ── Apply RoPE at the correct absolute positions ──────────────────────
        # During training, positions are always 0..S-1 (start_pos=0).
        # During KV-cache decoding, new tokens arrive at [start_pos .. start_pos+S-1].
        # Using the wrong positions here would corrupt the Q·K relative distances
        # and silently degrade model quality without any shape errors.
        cos_SxDh, sin_SxDh = compute_rope_freqs(
            seq_len, head_dim, self.rope_theta, start_pos=start_pos
        )
        q_HxSxD = apply_rope(q_HxSxD, cos_SxDh, sin_SxDh)
        k_KVHxSxD = apply_rope(k_KVHxSxD, cos_SxDh, sin_SxDh)

        # ── Write new K and V into the cache ──────────────────────────────────
        # Start indices (0, start_pos, 0) mean: write at all KV-head rows,
        # starting at column start_pos, all head-dim positions.
        #
        #   Before write:  cache.k[:, :start_pos,          :] holds history
        #   After  write:  new_k  [:, :start_pos+seq_len,  :] holds history + window
        new_k = jax.lax.dynamic_update_slice(
            cache.k_KVHxSxD, k_KVHxSxD, (0, start_pos, 0)
        )  # (n_kv_heads, max_seq_len, D)
        new_v = jax.lax.dynamic_update_slice(
            cache.v_KVHxSxD, v_KVHxSxD, (0, start_pos, 0)
        )
        new_cache = KVCache(k_KVHxSxD=new_k, v_KVHxSxD=new_v)

        # ── Expand KV heads to match query heads (GQA repeat) ─────────────────
        # We repeat over the *full* cache, not just the newly-written window.
        # This is the GQA head-expansion step applied to the entire history.
        k_HxMaxSxD = jnp.repeat(new_k, n_groups, axis=0)  # (n_heads, max_seq_len, D)
        v_HxMaxSxD = jnp.repeat(new_v, n_groups, axis=0)

        # ── Build the asymmetric causal mask ──────────────────────────────────
        # Q shape: (n_heads, S,           D) — only the current window
        # K shape: (n_heads, max_seq_len, D) — full cache
        # mask[i, j] = True iff query at absolute position (start_pos+i) may
        #              attend to key at absolute position j.
        q_pos_S = jnp.arange(start_pos, start_pos + seq_len)  # (S,)
        k_pos_MaxS = jnp.arange(max_seq_len)  # (max_seq_len,)
        causal_mask_SxMaxS = q_pos_S[:, None] >= k_pos_MaxS[None, :]  # (S, max_seq_len)

        # ── Attend over the full cache + merge heads ──────────────────────────
        # scaled_dot_product handles asymmetric (Sq, Sk) shapes automatically.
        # Uninitialised cache slots (positions >= start_pos+seq_len) are always
        # masked out, so their zero K/V values never affect the output.
        values_HxSxD, _ = scaled_dot_product(
            q_HxSxD, k_HxMaxSxD, v_HxMaxSxD, mask=causal_mask_SxMaxS
        )
        values_SxE = einops.rearrange(values_HxSxD, "h s d -> s (h d)", h=self.n_heads)
        output_SxE = jax.vmap(self.output_proj)(values_SxE)

        return output_SxE, new_cache


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

    def forward_with_cache(
        self,
        x_SxE: Float[Array, "seq_len n_embed"],
        cache: KVCache,
        start_pos: int,
    ) -> Tuple[Float[Array, "seq_len n_embed"], KVCache]:
        """
        Incremental MHA forward pass using a KV cache.

        MHA is the special case of GQA where n_kv_heads == n_heads, so the
        K/V cache has shape (n_heads, max_seq_len, head_dim) and no head
        expansion is needed.  See GroupedQueryAttention.forward_with_cache for
        a detailed explanation of the prefill/decode flow and masking strategy.

        Args:
            x_SxE:     Input embeddings, shape (seq_len, n_embed).
            cache:     KVCache with (n_heads, max_seq_len, head_dim) buffers.
            start_pos: Absolute position of x_SxE[0] in the full sequence.

        Returns:
            output_SxE: Updated embeddings, shape (seq_len, n_embed).
            new_cache:  KVCache with the current window written in.
        """
        seq_len, n_embed = x_SxE.shape
        head_dim = n_embed // self.n_heads
        max_seq_len = cache.k_KVHxSxD.shape[1]

        # Project to QKV — single matrix, then split
        qkv_SxD = jax.vmap(self.qkv_proj)(x_SxE)  # (S, 3 * n_embed)

        # Reshape: pack all 3*head_dim dims per head, then array_split into Q/K/V.
        # This mirrors the layout used in __call__ so the same trained weights apply.
        reshaped_qkv = einops.rearrange(
            qkv_SxD,
            "s (n_heads d) -> n_heads s d",
            s=seq_len,
            n_heads=self.n_heads,
        )  # (n_heads, S, 3 * head_dim)
        q_HxSxD, k_HxSxD, v_HxSxD = jnp.array_split(reshaped_qkv, 3, axis=-1)
        # Each: (n_heads, S, head_dim)

        # ── Apply RoPE at the correct absolute positions ──────────────────────
        cos_SxDh, sin_SxDh = compute_rope_freqs(
            seq_len, head_dim, self.rope_theta, start_pos=start_pos
        )
        q_HxSxD = apply_rope(q_HxSxD, cos_SxDh, sin_SxDh)
        k_HxSxD = apply_rope(k_HxSxD, cos_SxDh, sin_SxDh)

        # ── Write K and V into the cache ──────────────────────────────────────
        # For MHA, n_kv_heads == n_heads, so the cache first dim is n_heads.
        new_k = jax.lax.dynamic_update_slice(
            cache.k_KVHxSxD, k_HxSxD, (0, start_pos, 0)
        )  # (n_heads, max_seq_len, head_dim)
        new_v = jax.lax.dynamic_update_slice(
            cache.v_KVHxSxD, v_HxSxD, (0, start_pos, 0)
        )
        new_cache = KVCache(k_KVHxSxD=new_k, v_KVHxSxD=new_v)

        # ── Build asymmetric causal mask ──────────────────────────────────────
        q_pos_S = jnp.arange(start_pos, start_pos + seq_len)  # (S,)
        k_pos_MaxS = jnp.arange(max_seq_len)  # (max_seq_len,)
        causal_mask_SxMaxS = q_pos_S[:, None] >= k_pos_MaxS[None, :]  # (S, max_seq_len)

        # ── Attend over the full cache ────────────────────────────────────────
        # new_k / new_v shape: (n_heads, max_seq_len, head_dim)
        values_HxSxD, _ = scaled_dot_product(
            q_HxSxD, new_k, new_v, mask=causal_mask_SxMaxS
        )
        values_SxE = einops.rearrange(
            values_HxSxD,
            "n_heads s d -> s (n_heads d)",
            n_heads=self.n_heads,
            s=seq_len,
        )
        output_SxE = jax.vmap(self.output_proj)(values_SxE)

        return output_SxE, new_cache
