import equinox as eqx
import jax

from equinox import nn
from dataclasses import dataclass
from jax import numpy as jnp
from jaxtyping import Integer, Float, Array, PRNGKeyArray
from typing import List, Tuple, Optional, Union

from . import attention
from .config import GPTConfig


class SwiGLU(eqx.Module):
    """
    https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: Float[Array, "in_features out_features"]
    V: Float[Array, "in_features out_features"]
    b: Float[Array, "out_features"]
    c: Float[Array, "out_features"]

    def __init__(self, key: PRNGKeyArray, in_features: int, out_features: int) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # Use proper scaling for weight initialization (Xavier/Glorot uniform)
        # Scale by sqrt(2 / in_features) for better initialization
        scale = jnp.sqrt(2.0 / in_features)
        self.W = jax.random.normal(k1, (in_features, out_features)) * scale
        self.V = jax.random.normal(k2, (in_features, out_features)) * scale
        # Initialize biases to zero for better training stability
        self.b = jnp.zeros((out_features,))
        self.c = jnp.zeros((out_features,))

    def __call__(
        self, x: Float[Array, "... in_features"]
    ) -> Float[Array, "... out_features"]:
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)


class MLP(eqx.Module):
    c_fc: nn.Linear
    swiglu: SwiGLU
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_fc, key_swiglu, key_proj, key_dropout = jax.random.split(key, 4)

        self.c_fc = nn.Linear(
            key=key_fc,
            in_features=model_config.n_embed,
            out_features=4 * model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.swiglu = SwiGLU(
            key=key_swiglu,
            in_features=4 * model_config.n_embed,
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
        x: Float[Array, "n_tokens n_embed"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key, inference=inference)
        return x


class CasualSelfAttention(eqx.Module):
    mha: attention.MultiHeadAttention

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        self.mha = attention.MultiHeadAttention(
            key=key, n_embed=model_config.n_embed, n_heads=model_config.n_head
        )

    def __call__(
        self, x: Float[Array, "n_tokens n_embed"], mask:Optional[Integer[Array, "n_tokens n_tokens"]] = None
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

    def __call__(
        self, key: PRNGKeyArray, x: Float[Array, "n_tokens n_embed"], mask:Optional[Integer[Array, "n_tokens n_tokens"]] = None
    ) -> Float[Array, "n_tokens n_embed"]:
        mlp_keys = jax.random.split(key, x.shape[0])  # Create a key for each token
        x = jax.vmap(self.ln_1)(x)
        output_embeddings, attn = self.attn(x, mask=mask)
        x = x + output_embeddings
        x = jax.vmap(self.ln_2)(x)
        x = x + jax.vmap(self.mlp)(mlp_keys, x)
        return x


class Transformer(eqx.Module):
    wte: nn.Embedding
    wpe: nn.Embedding
    drop: nn.Dropout
    h: List[Block]
    ln_f: nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        te_key, pe_key, h_key = jax.random.split(key, 3)

        # token embeddings
        self.wte = nn.Embedding(
            key=te_key,
            num_embeddings=model_config.vocab_size,
            embedding_size=model_config.n_embed,
        )
        # positional embeddings
        self.wpe = nn.Embedding(
            key=pe_key,
            num_embeddings=model_config.block_size,
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
        mask:Optional[Integer[Array, "sequence_length sequence_length"]] = None,
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        pos = jnp.arange(0, len(tokens), dtype=jnp.int32)
        t_embed = jax.vmap(self.wte)(tokens)  # token embeddings
        p_embed = jax.vmap(self.wpe)(pos)  # positional embedidngs
        x = self.drop(
            t_embed + p_embed, inference=inference, key=key
        )  # TODO: confirm why is key optional in params
        for block in self.h:
            x = block(key, x, mask=mask)
        x = jax.vmap(self.ln_f)(x)
        return x


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
        mask:Optional[Integer[Array, "n_tokens n_tokens"]] = None,
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

            key_mask = jnp.arange(tokens.shape[0]) <= i   # (T,)
            mask = key_mask[None, :]  # shape (1, T)

            logits = self(model_key, tokens, mask=mask, inference=False) # use inference=False to get all logits
            logits = logits[i - 1, :] # get the logits for the next token
            logits = jnp.expand_dims(logits, axis=0) # shape (1, vocab)

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
