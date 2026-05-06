from __future__ import annotations
import equinox as eqx
import jax

from equinox import nn
from jax import numpy as jnp
from jaxtyping import Integer, Float, Array, PRNGKeyArray
from typing import List, Tuple, Optional

from . import attention
from .config import GPTConfig


class SwiGLU(eqx.Module):
    """
    SwiGLU activation unit from "GLU Variants Improve Transformer" (Shazeer, 2020).

    Implements: SwiGLU(x) = Swish(x @ W + b) ⊙ (x @ V + c)

    Both W and V project from in_features (n_embed) to out_features (4·n_embed),
    so this layer simultaneously expands and gates — replacing the plain c_fc linear.
    The key insight: gating should happen at the expansion step, on the raw embedding,
    not on an already-expanded intermediate.

    https://arxiv.org/abs/2002.05202 (section 5)
    https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: Float[Array, "in_features out_features"]
    V: Float[Array, "in_features out_features"]
    b: Float[Array, "out_features"]
    c: Float[Array, "out_features"]

    def __init__(self, key: PRNGKeyArray, in_features: int, out_features: int) -> None:
        k1, k2 = jax.random.split(key, 2)
        scale = jnp.sqrt(2.0 / in_features)
        self.W = jax.random.normal(k1, (in_features, out_features)) * scale
        self.V = jax.random.normal(k2, (in_features, out_features)) * scale
        self.b = jnp.zeros((out_features,))
        self.c = jnp.zeros((out_features,))

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "out_features"]:
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)


class MLP(eqx.Module):
    """
    FFN block using SwiGLU activation.

    Pipeline: SwiGLU(n_embed → 4·n_embed) → Linear(4·n_embed → n_embed) → Dropout

    SwiGLU replaces the traditional c_fc + activation pattern. Two parallel weight
    matrices W and V (each n_embed × 4·n_embed) project and gate in one step:
        h = Swish(x @ W) ⊙ (x @ V)
    Then c_proj contracts back to n_embed.

    Parameter count per layer: 2 × (n_embed × 4·n_embed) + (4·n_embed × n_embed)
                              = 3 × n_embed × 4·n_embed   (same as standard FFN)
    """

    swiglu: SwiGLU
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_swiglu, key_proj = jax.random.split(key, 2)

        # W and V are each (n_embed, 4·n_embed) — expands and gates simultaneously
        self.swiglu = SwiGLU(
            key=key_swiglu,
            in_features=model_config.n_embed,
            out_features=4 * model_config.n_embed,
        )

        self.c_proj = nn.Linear(
            key=key_proj,
            in_features=4 * model_config.n_embed,
            out_features=model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.dropout = nn.Dropout(model_config.dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "n_embed"],
        inference: bool = False,
    ) -> Float[Array, "n_embed"]:
        x = self.swiglu(x)  # n_embed → 4·n_embed
        x = self.c_proj(x)  # 4·n_embed → n_embed
        x = self.dropout(x, key=key, inference=inference)
        return x


class CasualSelfAttention(eqx.Module):
    # Holds either a MultiHeadAttention or GroupedQueryAttention instance.
    # Swap by setting n_kv_heads in GPTConfig:
    #   None (default) → MultiHeadAttention
    #   int            → GroupedQueryAttention  (n_kv_heads=1 gives MQA)
    mha: attention.MultiHeadAttention | attention.GroupedQueryAttention

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        if model_config.n_kv_heads is None:
            self.mha = attention.MultiHeadAttention(
                key=key,
                n_embed=model_config.n_embed,
                n_heads=model_config.n_head,
                rope_theta=model_config.rope_theta,
            )
        else:
            self.mha = attention.GroupedQueryAttention(
                key=key,
                n_embed=model_config.n_embed,
                n_heads=model_config.n_head,
                n_kv_heads=model_config.n_kv_heads,
                rope_theta=model_config.rope_theta,
            )

    def __call__(
        self,
        x: Float[Array, "n_tokens n_embed"],
        mask: Optional[Integer[Array, "n_tokens n_tokens"]] = None,
    ) -> Tuple[Float[Array, "n_tokens n_embed"], Float[Array, "n_tokens n_tokens"]]:
        """
        Args:
            x: Input embeddings of shape (n_tokens, n_embed)
        Returns:
            Tuple containing:
                - Output embeddings of shape (n_tokens, n_embed)
                - Attention weights of shape (n_tokens, n_tokens)
        """
        n_tokens = x.shape[0]
        causal_mask = jnp.tril(jnp.ones((n_tokens, n_tokens), dtype=bool))
        if mask is not None:
            final_mask = causal_mask & mask
        else:
            final_mask = causal_mask
        return self.mha(x, mask=final_mask)

    def forward_with_cache(
        self,
        x_SxE: Float[Array, "n_tokens n_embed"],
        cache: attention.KVCache,
        start_pos: int,
    ) -> Tuple[Float[Array, "n_tokens n_embed"], attention.KVCache]:
        """
        Delegates to the underlying MHA or GQA module's forward_with_cache.

        The causal mask is constructed *inside* forward_with_cache using absolute
        positions, so no mask-building is needed here — the position-awareness
        is handled entirely by start_pos.

        Args:
            x_SxE:     Input embeddings, shape (n_tokens, n_embed).
            cache:     KV cache for this attention layer.
            start_pos: Absolute position of x_SxE[0] in the full sequence.

        Returns:
            Tuple of (output embeddings, updated KV cache).
        """
        return self.mha.forward_with_cache(x_SxE, cache, start_pos)


class Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: CasualSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_attn, key_mlp = jax.random.split(key, 2)

        self.ln_1 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.attn = CasualSelfAttention(key=key_attn, model_config=model_config)
        self.ln_2 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.mlp = MLP(key=key_mlp, model_config=model_config)

    def __call__(self, key, x, mask=None):
        # 1. Attention Block
        # We normalize ONLY for the attention calculation
        normalized_x = jax.vmap(self.ln_1)(x)
        output_embeddings, attn = self.attn(normalized_x, mask=mask)

        # We add the result back to the ORIGINAL, un-normalized x
        x = x + output_embeddings

        # 2. MLP Block
        # We normalize the UPDATED x ONLY for the MLP calculation
        normalized_x2 = jax.vmap(self.ln_2)(x)
        mlp_keys = jax.random.split(key, x.shape[0])
        mlp_out = jax.vmap(self.mlp)(mlp_keys, normalized_x2)

        # Add the MLP result back to the residual highway
        x = x + mlp_out

        return x

    def forward_with_cache(
        self,
        key: PRNGKeyArray,
        x_SxE: Float[Array, "seq_len n_embed"],
        cache: attention.KVCache,
        start_pos: int,
    ) -> Tuple[Float[Array, "seq_len n_embed"], attention.KVCache]:
        """
        Single transformer block forward pass using a KV cache.

        Follows the same pre-norm + residual structure as __call__, with two
        differences that are appropriate for inference:
          - Attention uses forward_with_cache (attends over full cache history).
          - MLP is called with inference=True to disable dropout.

        Args:
            key:       PRNG key forwarded to MLP (unused when dropout=0).
            x_SxE:    Input embeddings, shape (seq_len, n_embed).
            cache:     KV cache for this block's attention layer.
            start_pos: Absolute position of x_SxE[0] in the sequence.

        Returns:
            (updated embeddings (seq_len, n_embed), updated KV cache)
        """
        # 1. Attention block — pre-norm, residual add
        normalized_x = jax.vmap(self.ln_1)(x_SxE)
        attn_out_SxE, new_cache = self.attn.forward_with_cache(
            normalized_x, cache, start_pos
        )
        x_SxE = x_SxE + attn_out_SxE

        # 2. MLP block — pre-norm, residual add, dropout disabled
        normalized_x2 = jax.vmap(self.ln_2)(x_SxE)
        mlp_keys = jax.random.split(key, x_SxE.shape[0])
        mlp_out_SxE = jax.vmap(lambda k, x: self.mlp(k, x, inference=True))(
            mlp_keys, normalized_x2
        )
        x_SxE = x_SxE + mlp_out_SxE

        return x_SxE, new_cache


class Transformer(eqx.Module):
    wte: nn.Embedding
    drop: nn.Dropout
    h: List[Block]
    ln_f: nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        te_key, h_key = jax.random.split(key, 2)

        # token embeddings — position information is injected by RoPE inside
        # each attention layer, so no separate positional embedding table is needed.
        self.wte = nn.Embedding(
            key=te_key,
            num_embeddings=model_config.vocab_size,
            embedding_size=model_config.n_embed,
        )
        self.drop = nn.Dropout(model_config.dropout)
        block_keys = jax.random.split(h_key, model_config.n_layers)
        self.h = [
            Block(key=block_keys[i], model_config=model_config)
            for i in range(model_config.n_layers)
        ]
        self.ln_f = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        mask: Optional[Integer[Array, "sequence_length sequence_length"]] = None,
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        t_embed = jax.vmap(self.wte)(tokens)  # (n_tokens, n_embed)
        x = self.drop(t_embed, inference=inference, key=key)
        for block in self.h:
            x = block(key, x, mask=mask)
        x = jax.vmap(self.ln_f)(x)
        return x

    def forward_with_cache(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        caches: List[attention.KVCache],
        start_pos: int,
    ) -> Tuple[Float[Array, "n_tokens n_embed"], List[attention.KVCache]]:
        """
        Full transformer stack forward pass using per-layer KV caches.

        Differences from __call__:
          - Embedding dropout is skipped (this method is inference-only).
          - Each block updates its own cache and returns the new version.
          - caches is a plain Python list; we rebuild it functionally each call
            rather than mutating in place (JAX's functional programming style).

        Args:
            key:      PRNG key forwarded to each block's MLP.
            tokens:   Token IDs, shape (n_tokens,).
            caches:   List of KVCache — one per transformer layer.
            start_pos: Absolute position of tokens[0] in the full sequence.

        Returns:
            (final hidden states (n_tokens, n_embed), list of updated KVCaches)
        """
        # Token embeddings — RoPE inside each attention layer provides position
        # information, so no separate positional embedding table is needed.
        x_SxE = jax.vmap(self.wte)(tokens)  # (n_tokens, n_embed)

        # No embedding dropout at inference time.

        new_caches = []
        for block, cache in zip(self.h, caches):
            x_SxE, new_cache = block.forward_with_cache(key, x_SxE, cache, start_pos)
            new_caches.append(new_cache)

        x_SxE = jax.vmap(self.ln_f)(x_SxE)
        return x_SxE, new_caches


class GPT(eqx.Module):
    transformer: Transformer
    lm_head: nn.Linear

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_transformer, key_lm_head = jax.random.split(key, 2)

        self.transformer = Transformer(key=key_transformer, model_config=model_config)
        self.lm_head = nn.Linear(
            key=key_lm_head,
            in_features=model_config.n_embed,
            out_features=model_config.vocab_size,
            use_bias=True,
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        mask: Optional[Integer[Array, "n_tokens n_tokens"]] = None,
        inference: bool = False,
    ) -> Float[Array, "n_tokens vocab_size"]:
        x = self.transformer(key, tokens, mask=mask, inference=inference)
        if not inference:
            logits = jax.vmap(self.lm_head)(x)  # (n_tokens, vocab_size)
        else:
            last_token_embedding = x[-1]
            # during inference we only care about the last token
            # vmap is not needed here, because it's only single token
            logits = self.lm_head(last_token_embedding)
            logits = jnp.expand_dims(logits, axis=0)
        return logits

    def decode(
        self,
        key: PRNGKeyArray,
        initial_tokens: Integer[Array, "n_tokens"],
        max_new_tokens: int,
        temperature=1.0,
        top_k=None,
    ) -> Integer[Array, "n_tokens + max_new_tokens"]:
        input_token_len = initial_tokens.shape[0]
        padding = jnp.zeros((max_new_tokens,), dtype=jnp.int32)
        tokens = jnp.concatenate([initial_tokens, padding], axis=-1)
        indexes = jnp.arange(input_token_len, input_token_len + max_new_tokens)

        def step(tokens, i):
            step_key = jax.random.fold_in(key, i)
            model_key, sample_key = jax.random.split(step_key)

            key_mask = jnp.arange(tokens.shape[0]) <= i  # (T,)
            mask = key_mask[None, :]  # shape (1, T)

            logits = self(
                model_key, tokens, mask=mask, inference=False
            )  # use inference=False to get all logits
            logits = logits[i - 1, :]  # get the logits for the next token
            logits = jnp.expand_dims(logits, axis=0)  # shape (1, vocab)

            # inference=True → logits shape (1, vocab)
            logits = logits[0] / temperature

            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(
                    logits, min(top_k, logits.shape[-1])
                )
                token_idx = jax.random.categorical(sample_key, top_logits)
                next_token = top_tokens[token_idx]
            else:
                next_token = jax.random.categorical(sample_key, logits)

            tokens = tokens.at[i].set(next_token)
            return tokens, None

        tokens, _ = jax.lax.scan(step, tokens, indexes)

        return tokens

    def decode_slow(
        self,
        key: PRNGKeyArray,
        initial_tokens: Integer[Array, "n_tokens"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[Integer] = None,
    ) -> Integer[Array, "n_tokens + max_new_tokens"]:
        """Generate text tokens given an initial sequence.

        Args:
            key: Random key for sampling
            initial_tokens: Initial sequence of tokens to continue from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more random)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            Array of generated tokens including the initial sequence
        """
        # Start with the initial tokens
        tokens = initial_tokens

        for i in range(max_new_tokens):
            # Get key for this iteration
            subkey = jax.random.fold_in(key, i)
            model_key, sample_key = jax.random.split(subkey)
            # during inference, we only get last token logits
            logits = self(model_key, tokens, inference=True)  # (1, vocab_size)
            logits = logits / temperature

            if top_k is not None:
                v, _ = jax.lax.top_k(logits, top_k)
                min_value = v[0, -1]
                logits = jnp.where(logits < min_value, -jnp.inf, logits)

            # jax.random.categorical expects log-probabilities. The logits are
            # already unnormalized log-probabilities, so we pass them directly
            # after applying temperature scaling and optional top-k filtering.
            next_token = jax.random.categorical(sample_key, logits[0])
            print(f"Generated token {i + 1}/{max_new_tokens}: {next_token}")
            tokens = jnp.append(tokens, next_token)

        return tokens

    def make_kv_cache(self, max_seq_len: int) -> List[attention.KVCache]:
        """
        Allocate a zero-filled KV cache for every transformer layer.

        Cache dimensions per layer:
            k_KVHxSxD : (n_kv_heads, max_seq_len, head_dim)
            v_KVHxSxD : (n_kv_heads, max_seq_len, head_dim)

        For MHA, n_kv_heads == n_heads.
        For GQA, n_kv_heads < n_heads — the reduced KV budget is the whole
        point: the cache is (n_heads / n_kv_heads)× smaller than MHA.

        Args:
            max_seq_len: Total position capacity (prompt + max_new_tokens).

        Returns:
            List of KVCache (one per layer), all zero-initialised.
        """
        caches = []
        for block in self.transformer.h:
            mha = block.attn.mha
            if isinstance(mha, attention.GroupedQueryAttention):
                n_kv = mha.n_kv_heads
            else:  # MultiHeadAttention — n_kv_heads == n_heads
                n_kv = mha.n_heads
            head_dim = mha.n_embed // mha.n_heads
            caches.append(
                attention.KVCache(
                    k_KVHxSxD=jnp.zeros((n_kv, max_seq_len, head_dim)),
                    v_KVHxSxD=jnp.zeros((n_kv, max_seq_len, head_dim)),
                )
            )
        return caches

    def decode_with_kv_cache(
        self,
        key: PRNGKeyArray,
        initial_tokens: Integer[Array, "n_tokens"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Integer[Array, "n_tokens + max_new_tokens"]:
        """
        Autoregressive text generation with KV caching.

        Why KV cache?
        -------------
        decode_slow re-runs the *full* forward pass at each of the T generation
        steps.  For a prompt of length P:

            decode_slow:           T full forward passes, each O((P+t)²) attention
                                   Total: O(T · (P+T)²)  — quadratic in T

            decode_with_kv_cache:  Two phases (explained below)
                                   Total: O(P² + T·P)     — linear in T

        Phase 1 — Prefill (one forward pass over the entire prompt)
        -----------------------------------------------------------
        All P prompt tokens are processed in *parallel*, exactly as in training.
        Each layer writes its K and V vectors for positions 0..P-1 into the cache.
        At the end we read the last position's embedding to predict the first new
        token.  Cost: O(P²).

        Phase 2 — Decode (T single-token steps)
        ----------------------------------------
        At each step we feed *one* token into the transformer.  That token's Q
        is dotted against all P+t cached K vectors — no re-projection of the
        prompt is ever needed again.  Cost per step: O(P+t) ≈ O(P).
        Total decode cost: O(T·P).

        Key/Value lifecycle
        -------------------
        After prefill the cache holds positions 0..P-1.
        Decode step 0 writes position P.
        Decode step 1 writes position P+1.  … and so on.

        Args:
            key:            PRNG key for sampling.
            initial_tokens: Prompt token IDs, shape (n_tokens,).
            max_new_tokens: Number of tokens to generate.
            temperature:    Softmax temperature (1.0 = unchanged, <1 = sharper,
                            >1 = more uniform / random).
            top_k:          If set, restrict sampling to the top-k most likely
                            next tokens at each step.

        Returns:
            Token array of shape (n_tokens + max_new_tokens,) containing the
            original prompt followed by the generated tokens.
        """
        prompt_len = initial_tokens.shape[0]
        max_seq_len = prompt_len + max_new_tokens

        # Allocate zero-filled caches (one KVCache per transformer layer)
        caches = self.make_kv_cache(max_seq_len)

        prefill_key, gen_key = jax.random.split(key)

        # ── Phase 1: Prefill ─────────────────────────────────────────────────
        # Process all prompt tokens in one shot, filling positions 0..P-1.
        # embeddings_PxE[i] holds the hidden state at position i, which the
        # model uses to predict the token at position i+1.
        embeddings_PxE, caches = self.transformer.forward_with_cache(
            prefill_key, initial_tokens, caches, start_pos=0
        )

        tokens = initial_tokens

        # ── Phase 2: Generate ─────────────────────────────────────────────────
        for i in range(max_new_tokens):
            step_key = jax.random.fold_in(gen_key, i)
            model_key, sample_key = jax.random.split(step_key)

            # embeddings_PxE[-1] is the hidden state at the *last processed*
            # position.  The language head converts it to next-token logits.
            logits_V = self.lm_head(embeddings_PxE[-1])  # (vocab_size,)
            logits_V = logits_V / temperature

            if top_k is not None:
                top_logits_K, top_tokens_K = jax.lax.top_k(
                    logits_V, min(top_k, logits_V.shape[-1])
                )
                token_idx = jax.random.categorical(sample_key, top_logits_K)
                next_token = top_tokens_K[token_idx]
            else:
                next_token = jax.random.categorical(sample_key, logits_V)

            tokens = jnp.append(tokens, next_token)

            # Feed the newly sampled token into the model at absolute position
            # (prompt_len + i).  This writes its K/V into the cache and
            # produces the embedding we will use to sample the *next* token.
            current_pos = prompt_len + i
            embeddings_PxE, caches = self.transformer.forward_with_cache(
                model_key, jnp.array([next_token]), caches, start_pos=current_pos
            )

        return tokens
