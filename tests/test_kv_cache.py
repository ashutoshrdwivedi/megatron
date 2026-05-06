"""
Tests for KV cache correctness (nanotron/attention.py, nanotron/model.py).

The central invariant tested throughout this file:

    forward_with_cache must produce bit-identical output to a standard __call__
    with a causal mask, at every sequence position.

Three layers of correctness are tested:

  1. Prefill correctness   — processing the full prompt at once with
                             forward_with_cache(start_pos=0) matches a
                             normal causal __call__ on the same input.

  2. Incremental correctness — feeding tokens one-by-one into the cache
                               (start_pos=0, 1, 2, …) matches each
                               position in the full causal forward pass.

  3. End-to-end generation  — decode_with_kv_cache logits at the first
                               new-token position match a full forward pass.

Why these tests matter
----------------------
The KV cache is a pure performance optimisation: it must not change what the
model *predicts*.  If any of these tests fail it means the cache is corrupting
attention outputs — incorrect RoPE positions, wrong causal masking, wrong
cache write offset, etc.
"""

import jax
import jax.numpy as jnp

from nanotron import attention
from nanotron.attention import KVCache
from nanotron.config import GPTConfig
from nanotron.model import GPT


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_gqa(key, n_embed=8, n_heads=4, n_kv_heads=2):
    return attention.GroupedQueryAttention(
        key, n_embed=n_embed, n_heads=n_heads, n_kv_heads=n_kv_heads
    )


def _make_mha(key, n_embed=8, n_heads=4):
    return attention.MultiHeadAttention(key, n_embed=n_embed, n_heads=n_heads)


def _empty_cache(n_kv_heads: int, max_seq_len: int, head_dim: int) -> KVCache:
    """Return a zero-initialised KVCache of the requested shape."""
    return KVCache(
        k_KVHxSxD=jnp.zeros((n_kv_heads, max_seq_len, head_dim)),
        v_KVHxSxD=jnp.zeros((n_kv_heads, max_seq_len, head_dim)),
    )


def _make_gpt(n_kv_heads=None, dropout=0.0, n_head=4, n_embed=16):
    """Construct a small GPT for testing with optional GQA and zeroed dropout."""
    cfg = GPTConfig(
        block_size=32,
        n_layers=2,
        vocab_size=16,
        n_head=n_head,
        n_embed=n_embed,
        dropout=dropout,
        n_kv_heads=n_kv_heads,
    )
    return GPT(key=jax.random.PRNGKey(0), model_config=cfg), cfg


# ── KVCache data structure ────────────────────────────────────────────────────


def test_kv_cache_is_jax_pytree():
    """
    KVCache is a NamedTuple, which JAX automatically registers as a PyTree.

    This matters because jit and lax.scan need to trace through the cache
    as a carry variable.  If KVCache were a plain Python object (not a PyTree)
    JAX would treat the whole thing as a static value and fail.

    We verify the property directly: tree_leaves should see exactly the two
    underlying arrays (k and v) and nothing else.
    """
    cache = _empty_cache(n_kv_heads=2, max_seq_len=8, head_dim=4)
    leaves = jax.tree_util.tree_leaves(cache)
    assert len(leaves) == 2, "KVCache should expose exactly 2 leaves (k and v)"
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_kv_cache_zero_initialised():
    """
    Freshly created cache must be all-zeros.

    Uninitialised positions must be zero so that the causal mask (which blocks
    them out anyway) doesn't accidentally admit garbage values if the mask ever
    has a bug.  Zero · anything = 0, providing a second line of defence.
    """
    cache = _empty_cache(n_kv_heads=2, max_seq_len=10, head_dim=4)
    assert jnp.all(cache.k_KVHxSxD == 0)
    assert jnp.all(cache.v_KVHxSxD == 0)


# ── GQA forward_with_cache ────────────────────────────────────────────────────


def test_gqa_prefill_matches_full_forward():
    """
    Processing a full prompt through forward_with_cache(start_pos=0) must
    produce the same output as the normal __call__ with a lower-triangular
    causal mask.

    This is the "prefill" phase: all S prompt tokens are processed in one shot,
    and their K/V vectors fill cache positions 0..S-1.  The output at every
    position should be indistinguishable from the non-cached forward pass
    because:
      - RoPE positions are [0..S-1] in both cases.
      - The causal mask allows the same token pairs in both cases.
    """
    key = jax.random.PRNGKey(0)
    seq_len, n_embed, n_heads, n_kv_heads = 5, 8, 4, 2
    gqa = _make_gqa(key, n_embed, n_heads, n_kv_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    # Reference: standard causal __call__
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    out_ref_SxE, _ = gqa(x_SxE, mask=causal_mask)

    # KV cache prefill: start_pos=0, full prompt
    head_dim = n_embed // n_heads
    cache = _empty_cache(n_kv_heads, seq_len, head_dim)
    out_cache_SxE, _ = gqa.forward_with_cache(x_SxE, cache, start_pos=0)

    assert jnp.allclose(out_ref_SxE, out_cache_SxE, atol=1e-5), (
        "GQA prefill output differs from full causal forward pass"
    )


def test_gqa_incremental_matches_full_forward():
    """
    Feeding tokens one-by-one into the cache must reproduce the output at each
    position from a full-sequence forward pass.

    This is the core correctness property of KV caching: whether you process
    all tokens at once or one at a time, the output at position i is identical
    — because in both cases position i only attends to positions 0..i.

    The test:
      1. Run a standard causal forward pass on S tokens → reference outputs.
      2. Run forward_with_cache S times, each time with one token and growing
         start_pos (0, 1, 2, …, S-1).
      3. Assert that step i's output matches out_ref[i].
    """
    key = jax.random.PRNGKey(1)
    seq_len, n_embed, n_heads, n_kv_heads = 6, 8, 4, 2
    gqa = _make_gqa(key, n_embed, n_heads, n_kv_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    # Reference: full causal pass
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    out_full_SxE, _ = gqa(x_SxE, mask=causal_mask)

    # Incremental: grow the cache one token at a time
    head_dim = n_embed // n_heads
    cache = _empty_cache(n_kv_heads, seq_len, head_dim)

    for i in range(seq_len):
        x_1xE = x_SxE[i : i + 1, :]  # (1, n_embed)
        out_1xE, cache = gqa.forward_with_cache(x_1xE, cache, start_pos=i)
        assert jnp.allclose(out_1xE[0], out_full_SxE[i], atol=1e-5), (
            f"GQA incremental output mismatch at position {i}"
        )


def test_gqa_cache_populated_correctly():
    """
    After a prefill of seq_len tokens into a max_seq_len > seq_len cache:
      - Positions [0 .. seq_len-1] must be non-zero (written by projection+RoPE).
      - Positions [seq_len .. max_seq_len-1] must remain zero (untouched).

    This verifies that dynamic_update_slice writes at the right offset and
    doesn't bleed into neighbouring positions.
    """
    key = jax.random.PRNGKey(2)
    seq_len, n_embed, n_heads, n_kv_heads = 4, 8, 4, 2
    gqa = _make_gqa(key, n_embed, n_heads, n_kv_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    max_seq_len = 10  # larger than seq_len to expose un-written slots
    head_dim = n_embed // n_heads
    cache = _empty_cache(n_kv_heads, max_seq_len, head_dim)
    _, new_cache = gqa.forward_with_cache(x_SxE, cache, start_pos=0)

    # Written region must be non-zero (projection produces non-zero K/V for
    # random inputs — the probability of an exact zero is negligible).
    assert jnp.any(new_cache.k_KVHxSxD[:, :seq_len, :] != 0), (
        "Cache should be populated at prompt positions"
    )
    # Unwritten region must remain exactly zero
    assert jnp.all(new_cache.k_KVHxSxD[:, seq_len:, :] == 0), (
        "Cache must not bleed into positions beyond the prompt"
    )


def test_gqa_cache_write_at_offset():
    """
    Writing at start_pos > 0 (a decode step) must only overwrite positions
    [start_pos .. start_pos+seq_len-1] and leave all earlier positions intact.

    This simulates a single decode step after a prefill: the previously cached
    history must be preserved unchanged.
    """
    key = jax.random.PRNGKey(3)
    n_embed, n_heads, n_kv_heads = 8, 4, 2
    gqa = _make_gqa(key, n_embed, n_heads, n_kv_heads)
    head_dim = n_embed // n_heads

    max_seq_len = 8
    # Pre-populate cache with sentinel value (1.0) everywhere
    sentinel = jnp.ones((n_kv_heads, max_seq_len, head_dim))
    prefilled_cache = KVCache(k_KVHxSxD=sentinel, v_KVHxSxD=sentinel)

    # Decode step: one token at start_pos=3
    x_1xE = jax.random.normal(key, (1, n_embed))
    _, updated_cache = gqa.forward_with_cache(x_1xE, prefilled_cache, start_pos=3)

    # Positions before start_pos must be untouched (still 1.0)
    assert jnp.all(updated_cache.k_KVHxSxD[:, :3, :] == 1.0), (
        "Positions before start_pos must not be overwritten"
    )
    # Positions after the written window must also be untouched
    assert jnp.all(updated_cache.k_KVHxSxD[:, 4:, :] == 1.0), (
        "Positions after the written window must not be overwritten"
    )


def test_gqa_mqa_special_case_with_cache():
    """
    Multi-Query Attention (MQA) is GQA with n_kv_heads=1.  forward_with_cache
    must still work: the cache has shape (1, max_seq_len, head_dim), and the
    single KV head is expanded to all n_heads query heads during attention.
    """
    key = jax.random.PRNGKey(4)
    seq_len, n_embed, n_heads, n_kv_heads = 4, 8, 4, 1
    mqa = attention.GroupedQueryAttention(
        key, n_embed=n_embed, n_heads=n_heads, n_kv_heads=n_kv_heads
    )
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    out_ref_SxE, _ = mqa(x_SxE, mask=causal_mask)

    head_dim = n_embed // n_heads
    cache = _empty_cache(n_kv_heads, seq_len, head_dim)
    out_cache_SxE, _ = mqa.forward_with_cache(x_SxE, cache, start_pos=0)

    assert jnp.allclose(out_ref_SxE, out_cache_SxE, atol=1e-5), (
        "MQA (n_kv_heads=1) prefill output differs from full forward"
    )


# ── MHA forward_with_cache ────────────────────────────────────────────────────


def test_mha_prefill_matches_full_forward():
    """
    Same invariant as test_gqa_prefill_matches_full_forward, but for MHA.

    MHA is GQA with n_kv_heads == n_heads, so the cache first dimension equals
    n_heads (not a reduced count).  No head expansion is needed at attention time.
    """
    key = jax.random.PRNGKey(5)
    seq_len, n_embed, n_heads = 5, 8, 4
    mha = _make_mha(key, n_embed, n_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    out_ref_SxE, _ = mha(x_SxE, mask=causal_mask)

    head_dim = n_embed // n_heads
    cache = _empty_cache(n_heads, seq_len, head_dim)  # n_kv_heads == n_heads for MHA
    out_cache_SxE, _ = mha.forward_with_cache(x_SxE, cache, start_pos=0)

    assert jnp.allclose(out_ref_SxE, out_cache_SxE, atol=1e-5), (
        "MHA prefill output differs from full causal forward pass"
    )


def test_mha_incremental_matches_full_forward():
    """
    One-by-one MHA decoding must reproduce every position of the full forward pass.
    """
    key = jax.random.PRNGKey(6)
    seq_len, n_embed, n_heads = 6, 8, 4
    mha = _make_mha(key, n_embed, n_heads)
    x_SxE = jax.random.normal(key, (seq_len, n_embed))

    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    out_full_SxE, _ = mha(x_SxE, mask=causal_mask)

    head_dim = n_embed // n_heads
    cache = _empty_cache(n_heads, seq_len, head_dim)

    for i in range(seq_len):
        x_1xE = x_SxE[i : i + 1, :]
        out_1xE, cache = mha.forward_with_cache(x_1xE, cache, start_pos=i)
        assert jnp.allclose(out_1xE[0], out_full_SxE[i], atol=1e-5), (
            f"MHA incremental output mismatch at position {i}"
        )


# ── GPT-level: make_kv_cache ──────────────────────────────────────────────────


def test_make_kv_cache_shapes_mha():
    """
    make_kv_cache must allocate caches with the correct shapes for MHA.
    For MHA (n_kv_heads=None), the cache stores n_head KV heads per layer.
    """
    gpt, cfg = _make_gpt(n_kv_heads=None)
    max_seq_len = 20
    head_dim = cfg.n_embed // cfg.n_head
    caches = gpt.make_kv_cache(max_seq_len)

    assert len(caches) == cfg.n_layers, "Need one KVCache per transformer layer"
    for layer_idx, cache in enumerate(caches):
        expected_shape = (cfg.n_head, max_seq_len, head_dim)
        assert cache.k_KVHxSxD.shape == expected_shape, (
            f"Layer {layer_idx} k shape: expected {expected_shape}, "
            f"got {cache.k_KVHxSxD.shape}"
        )
        assert cache.v_KVHxSxD.shape == expected_shape, (
            f"Layer {layer_idx} v shape: expected {expected_shape}, "
            f"got {cache.v_KVHxSxD.shape}"
        )


def test_make_kv_cache_shapes_gqa():
    """
    For GQA (n_kv_heads=2, n_head=4), the cache stores only n_kv_heads KV heads
    per layer — not n_head.  This is the memory saving GQA is designed for:
    the cache is n_head/n_kv_heads = 2× smaller than MHA would require.
    """
    n_kv_heads = 2
    gpt, cfg = _make_gpt(n_kv_heads=n_kv_heads)
    max_seq_len = 20
    head_dim = cfg.n_embed // cfg.n_head
    caches = gpt.make_kv_cache(max_seq_len)

    assert len(caches) == cfg.n_layers
    for layer_idx, cache in enumerate(caches):
        expected_shape = (n_kv_heads, max_seq_len, head_dim)
        assert cache.k_KVHxSxD.shape == expected_shape, (
            f"Layer {layer_idx}: expected n_kv_heads={n_kv_heads} in cache, "
            f"got first dim={cache.k_KVHxSxD.shape[0]}"
        )


def test_make_kv_cache_all_zeros():
    """Freshly allocated cache must be zero-filled at every layer."""
    gpt, _ = _make_gpt()
    caches = gpt.make_kv_cache(max_seq_len=16)
    for layer_idx, cache in enumerate(caches):
        assert jnp.all(cache.k_KVHxSxD == 0), f"Layer {layer_idx} k not zero"
        assert jnp.all(cache.v_KVHxSxD == 0), f"Layer {layer_idx} v not zero"


# ── GPT-level: decode_with_kv_cache ──────────────────────────────────────────


def test_decode_with_kv_cache_output_length():
    """
    decode_with_kv_cache must return exactly n_tokens + max_new_tokens tokens
    (prompt preserved, then generated tokens appended).
    """
    gpt, _ = _make_gpt(n_kv_heads=2)
    key = jax.random.PRNGKey(42)
    initial_tokens = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    max_new_tokens = 6

    tokens = gpt.decode_with_kv_cache(key, initial_tokens, max_new_tokens)

    expected_len = len(initial_tokens) + max_new_tokens
    assert tokens.shape == (expected_len,), (
        f"Expected {expected_len} tokens, got shape {tokens.shape}"
    )


def test_decode_with_kv_cache_preserves_prompt():
    """
    The first n_tokens elements of the output must be identical to initial_tokens.
    The KV cache should never modify the prompt.
    """
    gpt, _ = _make_gpt(n_kv_heads=2)
    key = jax.random.PRNGKey(42)
    initial_tokens = jnp.array([0, 3, 7, 1], dtype=jnp.int32)

    tokens = gpt.decode_with_kv_cache(key, initial_tokens, max_new_tokens=4)

    assert jnp.array_equal(tokens[: len(initial_tokens)], initial_tokens), (
        "Prompt tokens must be unchanged in the output"
    )


def test_decode_with_kv_cache_tokens_in_vocab():
    """
    Every generated token must be a valid vocabulary index (0 <= t < vocab_size).
    """
    gpt, cfg = _make_gpt(n_kv_heads=2)
    key = jax.random.PRNGKey(0)
    initial_tokens = jnp.array([0, 1], dtype=jnp.int32)
    max_new_tokens = 8

    tokens = gpt.decode_with_kv_cache(key, initial_tokens, max_new_tokens)
    generated = tokens[len(initial_tokens) :]

    assert jnp.all(generated >= 0), "Generated tokens must be non-negative"
    assert jnp.all(generated < cfg.vocab_size), (
        f"Generated tokens must be < vocab_size={cfg.vocab_size}"
    )


def test_decode_with_kv_cache_logits_match_full_forward():
    """
    The next-token logits after prefilling with the KV cache must match those
    from a standard full-sequence forward pass on the same prompt.

    This is the fundamental end-to-end correctness check at the GPT level:
    the KV cache is a pure performance optimisation — it must not change what
    the model predicts.

    We use dropout=0.0 so that both paths are fully deterministic and
    comparable.  We also use n_kv_heads=None (MHA) to cover that path.

    What we compare:
        Full forward:  GPT.__call__(tokens)[-1]  → logits at last prompt position
        KV cache:      transformer.forward_with_cache(tokens)[0][-1] → same spot
    """
    gpt, _ = _make_gpt(n_kv_heads=None, dropout=0.0)
    key = jax.random.PRNGKey(0)
    tokens = jnp.array([0, 2, 5, 1, 3], dtype=jnp.int32)  # prompt length 5

    # Reference: standard causal forward pass, all tokens
    # inference=False → full sequence logits shape (5, vocab_size)
    ref_logits = gpt(key, tokens, inference=False)
    ref_next_logits_V = ref_logits[-1]  # predict token at position 5

    # KV cache: prefill the same tokens, extract logits from last embedding
    max_seq_len = len(tokens) + 10
    caches = gpt.make_kv_cache(max_seq_len)
    embeddings_PxE, _ = gpt.transformer.forward_with_cache(
        key, tokens, caches, start_pos=0
    )
    kv_next_logits_V = gpt.lm_head(embeddings_PxE[-1])  # (vocab_size,)

    assert jnp.allclose(ref_next_logits_V, kv_next_logits_V, atol=1e-5), (
        "KV cache prefill produces different next-token logits than full forward pass.\n"
        f"  Max absolute diff: {jnp.max(jnp.abs(ref_next_logits_V - kv_next_logits_V))}"
    )


def test_decode_with_kv_cache_logits_match_full_forward_per_step():
    """
    At every decode step the KV-cache logits must match a full-sequence forward
    pass on the identical context seen so far.

    This is the fundamental correctness guarantee of KV caching: it is a pure
    speed optimisation that must never change what the model predicts.

    Why we test logits, not sampled tokens
    ---------------------------------------
    decode_slow and decode_with_kv_cache use different PRNG key-splitting
    schemes, so their sampled tokens can differ even when the underlying logits
    are bit-identical.  Comparing logits directly removes that ambiguity.

    Test procedure (with dropout=0 for full determinism):
      1. Prefill the prompt.  Compare last-position logits with a reference
         full forward pass on the same prompt.
      2. Greedily pick the highest-probability token (no randomness needed).
      3. Feed that token into the KV cache (one-step decode).
         Simultaneously build up a growing context array and run a full forward
         pass on it.  Compare the logits at the new last position.
      4. Repeat for n_steps steps.

    If any step's logits drift, it pinpoints *where* the cache goes wrong
    (RoPE position bug, wrong cache slot, mask error, etc.).
    """
    gpt, _ = _make_gpt(n_kv_heads=None, dropout=0.0)
    key = jax.random.PRNGKey(0)
    prompt = jnp.array([0, 2, 5, 1], dtype=jnp.int32)
    n_steps = 4  # number of greedy decode steps to verify

    max_seq_len = len(prompt) + n_steps
    caches = gpt.make_kv_cache(max_seq_len)

    # ── Prefill ──────────────────────────────────────────────────────────────
    embeddings_PxE, caches = gpt.transformer.forward_with_cache(
        key, prompt, caches, start_pos=0
    )

    context = prompt  # growing context for the reference full forward pass

    for step in range(n_steps):
        # Logits from KV cache path
        kv_logits_V = gpt.lm_head(embeddings_PxE[-1])  # (vocab_size,)

        # Logits from reference full forward pass on the same context
        ref_logits_all = gpt(key, context, inference=False)  # (ctx_len, vocab)
        ref_logits_V = ref_logits_all[-1]  # last position

        assert jnp.allclose(kv_logits_V, ref_logits_V, atol=1e-5), (
            f"Logit mismatch at decode step {step} (context length {len(context)}).\n"
            f"  Max absolute diff: "
            f"{jnp.max(jnp.abs(kv_logits_V - ref_logits_V)):.2e}"
        )

        # Greedy token selection — same token chosen by both paths
        next_token = jnp.argmax(kv_logits_V)
        context = jnp.append(context, next_token)

        # Advance the KV cache by one token
        current_pos = len(prompt) + step
        embeddings_PxE, caches = gpt.transformer.forward_with_cache(
            key, jnp.array([next_token]), caches, start_pos=current_pos
        )
