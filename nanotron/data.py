from __future__ import annotations
import jax

from datasets import load_dataset
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from typing import Any, Dict, Generator, Optional, Tuple

dataset_name = "karpathy/tiny_shakespeare"
_cached_vocab_info: Optional[Dict[str, Any]] = None
# Cache for pre-encoded splits to avoid repeated Python-level encoding
_cached_encoded_splits: Dict[
    str, jnp.ndarray
] = {}  # key is the split like train, validation, test and value is the pre-encoded array


def get_dataset() -> Any:
    return load_dataset(dataset_name, trust_remote_code=True)


def get_vocabulary_info() -> Dict[str, Any]:
    global _cached_vocab_info
    if _cached_vocab_info is not None:
        return _cached_vocab_info

    dataset = get_dataset()
    # Use 'train' split to define the vocabulary
    train_text_list = dataset["train"]["text"]
    full_train_text = "".join(train_text_list)

    chars = sorted(list(set(full_train_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    _cached_vocab_info = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
    return _cached_vocab_info


def get_encoder() -> Dict[str, int]:
    return get_vocabulary_info()["stoi"]


def get_decoder() -> Dict[int, str]:
    return get_vocabulary_info()["itos"]


def get_vocab_size() -> int:
    return get_vocabulary_info()["vocab_size"]


def get_infinite_dataloader(
    key: PRNGKeyArray, split_type: str, batch_size: int, seq_len: int
) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
    """
    Get an infinite dataloader for a given split type, batch size, and sequence length.
    Args:
        key: Random key for randomness
        split_type: Type of split to get data from (train, validation, test)
        batch_size: Batch size
        seq_len: Sequence length
    Returns:
        Generator of tuples of (x, y) where x is a batch of sequences and y is a batch of next characters
        x is a batch of sequences of length seq_len
        y is a batch of next characters
        x and y are both of shape (batch_size, seq_len)
        x is the input sequences
        y is the next characters
    """
    dataset_obj = get_dataset()
    assert split_type in dataset_obj.keys(), f"Split {split_type} not found in dataset"

    # Prepare or retrieve pre-encoded array for this split
    if split_type not in _cached_encoded_splits:
        text_list = dataset_obj[split_type]["text"]
        full_text = "".join(text_list)
        vocab_info = get_vocabulary_info()
        stoi = vocab_info["stoi"]
        # tokens here are just integers for each character
        token_list = [stoi[c] for c in full_text if c in stoi]
        if len(token_list) < seq_len + 1:
            raise ValueError(
                f"Dataset split '{split_type}' has insufficient data ({len(token_list)} tokens) "
                f"for sequence length {seq_len}. Minimum required is {seq_len + 1}."
            )
        _cached_encoded_splits[split_type] = jnp.array(token_list, dtype=jnp.int32)
    data_array = _cached_encoded_splits[split_type]
    max_tokens = data_array.shape[0]
    while True:
        # Create new subkey for randomness
        key, subkey = jax.random.split(key)
        # sample a batch of random token indices of the size batch_size
        ix = jax.random.randint(
            key=subkey,
            shape=(batch_size,),
            minval=0,
            maxval=max_tokens
            - seq_len
            - 1,  # subtract seq_len to avoid out of bound while sampling sequences
        )  # shape (batch_size,)
        # ix = jnp.arange(batch_size).reshape(batch_size, ) # keep the batch same for debugging
        # broadcast the token indices to get the input sequences
        arange_seq = jnp.arange(seq_len)
        idx = ix[:, None] + arange_seq[None, :]
        x = jnp.take(data_array, idx)
        y = jnp.take(data_array, idx + 1)
        yield x, y
