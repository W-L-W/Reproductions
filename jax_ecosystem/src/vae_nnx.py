"""Variational Autoencoder using Flax and JAX."""

from typing import Any, Dict
from functools import partial
from dataclasses import dataclass

from jaxtyping import Array, Float
import tensorflow_datasets as tfds
from flax import nnx
import distrax as dx
import tensorflow as tf

from jax.nn import leaky_relu
from einops import rearrange


def get_datasets(
    batch_size: int,
    train_steps: int,
    buffer_size: int = 1024,
) -> Dict[str, Any]:
    train_ds: tf.data.Dataset = tfds.load("mnist", split="train")
    test_ds: tf.data.Dataset = tfds.load("mnist", split="test")

    def ds_mapper(sample):
        return {
            "image": tf.cast(sample["image"], tf.float32) / 255.0,
            "label": sample["label"],
        }

    train_ds = train_ds.map(ds_mapper)
    test_ds = test_ds.map(ds_mapper)

    train_ds = train_ds.repeat().shuffle(buffer_size)
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the
    # next sample to improve latency
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    )
    test_ds = test_ds.batch(batch_size).prefetch(1)


@dataclass
class VAESpec:
    d_input: int = 28
    d_pre: int = 256
    d_latent: int = 10
    base_filters: int = 8

    @property
    def d_big(self):
        dim = (self.d_input / 4) ** 2 * (2 * self.base_filters)
        return int(dim)


class Encoder(nnx.Module):
    """Encoder for VAE."""

    def __init__(self, *, rngs: nnx.Rngs, vae_spec: VAESpec):
        self.vae_spec = vae_spec
        d_latent = vae_spec.d_latent
        base_filters = vae_spec.base_filters
        d_pre = vae_spec.d_pre
        d_big = (vae_spec.d_input / 4) ** 2 * (2 * base_filters)

        self.conv1 = nnx.Conv(1, base_filters, kernel_size=5, strides=2, rngs=rngs)
        self.conv2 = nnx.Conv(
            base_filters, 2 * base_filters, kernel_size=5, strides=2, rngs=rngs
        )
        self.lin_down = nnx.Linear(d_big, d_pre, rngs=rngs)
        self.lin_mean = nnx.Linear(d_pre, d_latent, rngs=rngs)
        self.lin_std = nnx.Linear(d_pre, d_latent, rngs=rngs)

    # TODO: resolve how to do batches of distributions
    # presume this is handled implicitly by batching throughout?
    def __call__(self, x: Float[Array, "... h w"]) -> dx.MultivariateNormalDiag:
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.lin_down(x))
        loc = self.lin_mean(x)
        scale_diag = nnx.softplus(self.lin_std(x))
        return dx.MultivariateNormalDiag(
            loc=loc,
            scale_diag=scale_diag,
        )


class Decoder(nnx.Module):
    """Decoder for VAE"""

    def __init__(self, *, rngs: nnx.Rngs, vae_spec: VAESpec):
        vs = self.vae_spec = vae_spec
        self.linear1 = nnx.Linear(vae_spec.d_latent, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 3136, rngs=rngs)
        self.conv1 = nnx.Conv(64, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 1, kernel_size=(3, 3), rngs=rngs)
        self.upsample = partial(nnx.upsample, scale_factor=(2, 2))

    def __call__(self, x):
        pass


if __name__ == "__main__":
    pass
