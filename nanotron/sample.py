from __future__ import annotations
import equinox as eqx
import os
import pickle
import jax
import typer

import jax.numpy as jnp
from jaxtyping import Integer, Array

from typing_extensions import Annotated
from typing import Callable, Tuple
from . import model, config

print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")


def get_latest_checkpoint(out_dir: str) -> str:
    """Loads the latest .eqx checkpoint from the given directory."""
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Checkpoint directory {out_dir} not found.")
    ckpts = [f for f in os.listdir(out_dir) if f.endswith(".eqx")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint (.eqx file) found in {out_dir}")
    # Sort by modification time to get the latest
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
    return os.path.join(out_dir, ckpts[0])


def read_char_tokenizer(
    out_dir: str,
) -> Tuple[Callable[[str], jnp.ndarray], Callable[[jnp.ndarray], str], int]:
    """Returns encoding and decoding functions based on the dataset's vocabulary.
    Loads from meta.pkl in out_dir.
    """
    meta_path = os.path.join(out_dir, "meta.pkl")

    # Ensure pickle is imported at the top of the file
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]

    # Debug information
    print(f"Loaded vocabulary of size {vocab_size}")
    print(f"Sample of stoi mapping: {dict(list(stoi.items())[:5])}")
    print(f"Sample of itos mapping: {dict(list(itos.items())[:5])}")
    # Verify itos maps integers to characters
    if not all(isinstance(k, int) for k in itos.keys()):
        print("WARNING: itos dictionary has non-integer keys!")

    def encode_fn(s: str) -> Integer[Array, "n"]:
        return jnp.array([stoi[c] for c in s if c in stoi], dtype=jnp.int32)

    def decode_fn(arr: Integer[Array, "n"]) -> str:
        return "".join([itos[int(t)] for t in arr if int(t) in itos])

    return encode_fn, decode_fn, vocab_size


app = typer.Typer()


@app.command()
def main(
    # Arguments from SampleScriptConfig
    out_dir: Annotated[
        str, typer.Option(help="Directory to load checkpoint from.")
    ] = config.TrainConfig.out_dir,
    prompt: Annotated[str, typer.Option(help="Prompt string")] = "\\\\n",
    num_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 3,
    max_new_tokens: Annotated[
        int, typer.Option(help="Number of tokens generated in each sample")
    ] = 100,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature (1.0 = no change)")
    ] = 0.8,
    top_k: Annotated[
        int,
        typer.Option(
            help="Retain only the top_k most likely tokens, 0 for no top-k filtering"
        ),
    ] = 0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    # Arguments from config.GPTConfig
    block_size: Annotated[
        int, typer.Option(help="Context block size for the model")
    ] = config.GPTConfig.block_size,
    vocab_size: Annotated[
        int, typer.Option(help="Vocabulary size (will be overridden by meta.pkl)")
    ] = config.GPTConfig.vocab_size,
    n_layers: Annotated[
        int, typer.Option(help="Number of transformer layers")
    ] = config.GPTConfig.n_layers,
    n_head: Annotated[
        int, typer.Option(help="Number of attention heads")
    ] = config.GPTConfig.n_head,
    n_embed: Annotated[
        int, typer.Option(help="Embedding dimension")
    ] = config.GPTConfig.n_embed,
    dropout: Annotated[
        float, typer.Option(help="Dropout rate")
    ] = config.GPTConfig.dropout,
    bias: Annotated[
        bool, typer.Option(help="Whether to use bias in Linear and LayerNorm layers")
    ] = config.GPTConfig.bias,
    slow: Annotated[
        bool, typer.Option(help="Whether to use the slow decoding method")
    ] = False,
):
    """
    Sample from a trained JAX-based GPT model.
    """

    encode_fn, decode_fn, vocab_size = read_char_tokenizer(out_dir)

    # Create config objects from Typer arguments
    gpt_config_obj = config.GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_head=n_head,
        n_embed=n_embed,
        dropout=dropout,
        bias=bias,
    )
    # sample_config is now directly the arguments: out_dir, start, num_samples etc.

    key = jax.random.PRNGKey(seed)
    model_init_key, generation_key = jax.random.split(key)
    empty_model = model.GPT(model_init_key, gpt_config_obj)

    try:
        ckpt_path = get_latest_checkpoint(out_dir)
        loaded_model = eqx.tree_deserialise_leaves(ckpt_path, empty_model)
    except FileNotFoundError as e:
        print(f"Error loading checkpoint: {e}")
        print(
            f"Please ensure --out_dir ('{out_dir}') contains a valid .eqx checkpoint and meta.pkl, or specify the correct directory."
        )
        return

    print(f"Model loaded. Generating {num_samples} samples...")

    start_ids = encode_fn(prompt)
    # Run generation loop
    for i in range(num_samples):  # Use the 'num_samples' argument
        generation_key, sample_key = jax.random.split(
            generation_key
        )  # New key for each sample

        print(f"--- Sample {i + 1}/{num_samples} ---")

        # Ensure top_k is None if 0, as per model.decode's expectation for no top-k
        current_top_k = top_k if top_k > 0 else None  # Use the 'top_k' argument

        print(start_ids)
        model_decode_fn = loaded_model.decode_slow if slow else loaded_model.decode
        generated_tokens = model_decode_fn(
            key=sample_key,
            initial_tokens=start_ids,
            max_new_tokens=max_new_tokens,  # Use the 'max_new_tokens' argument
            temperature=temperature,  # Use the 'temperature' argument
            top_k=current_top_k,
        )

        generated_text = decode_fn(generated_tokens)
        print("generated_text:\n")
        print(generated_text)
    print("---------------")


if __name__ == "__main__":
    app()
