from __future__ import annotations

import dataclasses
import jax
import einops
import os
import optax

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
from pprint import pprint
from simple_parsing import ArgumentParser
from typing import Tuple, Any

from . import model, data, config, utils

seed = 42

print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")


def get_optimizers(
    model: model.GPT, weight_decay: float, learning_rate: float, betas: Tuple
) -> optax.GradientTransformation:
    # Weight decay optimizer
    decay_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate,
            b1=betas[0],
            b2=betas[1],
            weight_decay=weight_decay,
        ),
    )
    # No weight decay optimizer
    nodecay_optimizer = optax.adamw(
        learning_rate=learning_rate, b1=betas[0], b2=betas[1]
    )

    # Divide parameters into groups based on their dimensionality
    def is_decay_weight(leaf: Any) -> bool:
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # all weight tensors in matmuls + embeddings decay.
        # all biases and layernorms don't.
        return eqx.is_array(leaf) and leaf.ndim >= 2

    def get_param_label(leaf: Any) -> str:
        if is_decay_weight(leaf):
            return "decay"
        return "nodecay"

    param_labels = jax.tree_util.tree_map(get_param_label, model)
    # eqx.tree_pprint(param_labels)
    # Combine the two optimizers into one
    optimizer = optax.multi_transform(
        transforms={
            "decay": decay_optimizer,
            "nodecay": nodecay_optimizer,
        },
        param_labels=lambda _: param_labels,
    )
    return optimizer


def get_loss(
    logits: Float[Array, "batch seq_len"], y: Float[Array, "batch seq_len"]
) -> Float[Array, "batch seq_len"]:
    logits = einops.rearrange(logits, "batch seq_len logits -> (batch seq_len) logits")
    y = einops.rearrange(y, "batch seq_len -> (batch seq_len)")
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)  # type: ignore
    # return jnp.mean(jax.nn.log_softmax(logits) * y)


@eqx.filter_value_and_grad
def compute_grads(
    model: model.GPT,
    key: PRNGKeyArray,
    x: Float[Array, "batch seq_len"],
    y: Float[Array, "batch seq_len"],
) -> Float[Array, "1"]:
    batch_size = x.shape[0]
    sub_keys = jax.random.split(key, batch_size)
    logits = jax.vmap(model, in_axes=(0, 0))(sub_keys, x)  # (batch_size,)
    loss = get_loss(logits, y)
    return jnp.mean(loss)


@eqx.filter_jit
def step(
    key: PRNGKeyArray,
    model: model.GPT,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    batch_data: Tuple[Float[Array, "batch seq_len"], Float[Array, "batch seq_len"]],
) -> Tuple[model.GPT, optax.OptState, Float[Array, "1"]]:
    x, y = batch_data
    loss, grads = compute_grads(model, key, x, y)
    model_params = eqx.filter(model, eqx.is_inexact_array)
    updates, new_state = optimizer.update(grads, state, model_params)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


def eval(
    key: PRNGKeyArray,
    model: model.GPT,
    val_data: Tuple[Float[Array, "batch seq_len"], Float[Array, "batch seq_len"]],
) -> Float[Array, "1"]:
    x, y = val_data
    logits = jax.vmap(model, in_axes=(None, 0))(key, x)  # (batch_size,)
    return jnp.mean(get_loss(logits, y))


def print_training_info(
    model: eqx.Module, model_config: config.GPTConfig, train_config: config.TrainConfig
) -> None:
    pprint(dataclasses.asdict(train_config), width=1)
    pprint(dataclasses.asdict(model_config), width=1)
    # Count parameters: filter to only arrays, then sum their sizes
    model_params = eqx.filter(model, eqx.is_inexact_array)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
    print(f"Model parameters: {num_params:,}")


def train(train_config: config.TrainConfig, model_config: config.GPTConfig) -> None:
    key = jax.random.PRNGKey(seed)
    train_key, eval_key, data_key, model_key = jax.random.split(key, 4)

    vocab_info = data.get_vocabulary_info()
    model_config.vocab_size = vocab_info[
        "vocab_size"
    ]  # TODO provide a way to override this

    gpt = model.GPT(model_key, model_config)
    model_params = eqx.filter(gpt, eqx.is_inexact_array)
    # Initialize the optimizer
    optimizer = get_optimizers(
        model_params,  # model_params has a __call__ method, which causes an error so we wrap it in a lambda
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
    )

    # Initialize the optimizer state
    state = optimizer.init(model_params)

    # Get the infinite dataloader
    train_data_key, val_data_key = jax.random.split(data_key)
    train_dataloader = data.get_infinite_dataloader(
        train_data_key, "train", train_config.batch_size, model_config.block_size
    )
    val_dataloader = data.get_infinite_dataloader(
        val_data_key, "validation", train_config.batch_size, model_config.block_size
    )

    print_training_info(gpt, model_config, train_config)

    for i in range(train_config.num_steps):
        train_key, step_key = jax.random.split(train_key)
        train_batch = next(train_dataloader)
        gpt, state, loss = step(step_key, gpt, optimizer, state, train_batch)

        if i % train_config.log_interval == 0:
            print(f"Step {i}, Loss: {loss}")

        if i % train_config.eval_interval == 0:
            eval_key, sub_eval_key = jax.random.split(eval_key)
            val_batch = next(val_dataloader)
            eval_loss = eval(sub_eval_key, gpt, val_batch)
            print(f"Eval Loss: {eval_loss}")
            if train_config.always_save_checkpoint:
                utils.save_model(gpt, train_config.out_dir, i, vocab_info)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(config.TrainConfig, dest="train_config")
    parser.add_arguments(config.GPTConfig, dest="model_config")
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        default=True,
        help="Clean the output directory before training. Enabled by default.",
    )
    args = parser.parse_args()

    if args.clean_output_dir:
        utils.clean_output_dir(args.train_config.out_dir)
        print(f"Cleaned output directory: {args.train_config.out_dir}")

    train(args.train_config, args.model_config)
