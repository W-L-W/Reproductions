from dataclasses import dataclass
from typing import Iterable
import jax.numpy as jnp
import flax.nnx as nnx
from jaxtyping import Array, Float
import distrax as dx

# plan: work through the deepmind version


@dataclass
class CNPSpec:
    """CNP specification."""

    x_dim: int
    y_dim: int
    r_dim: int
    encoder_layer_output_sizes: Iterable[int]
    decoder_hidden_output_sizes: Iterable[int]

    @property
    def encoder_input_size(self):
        return self.x_dim + self.y_dim

    @property
    def decoder_input_size(self):
        return self.x_dim + self.r_dim

    @property
    def decoder_output_size(self):
        return 2 * self.y_dim

    @property
    def encoder_layer_dim_pairs(self):
        full_sizes = [self.encoder_input_size] + self.encoder_layer_output_sizes
        return zip(full_sizes[:-1], full_sizes[1:])

    @property
    def decoder_layer_dim_pairs(self):
        full_sizes = (
            [self.decoder_input_size]
            + self.decoder_hidden_output_sizes
            + [self.decoder_output_size]
        )
        return zip(full_sizes[:-1], full_sizes[1:])


class Encoder(nnx.Module):
    """MLP encoder"""

    def __init__(self, *, cnp_spec: CNPSpec, rngs: nnx.Rngs):
        """ "
        Args:
            output_sizes: the widths for each output layer of MLP
        """
        self.layers = [
            nnx.Linear(d_in, d_out, rngs=rngs)
            for d_in, d_out in cnp_spec.encoder_layer_dim_pairs
        ]

    def __call__(
        self,
        context_x: Float[Array, "... nc dim_x"],
        context_y: Float[Array, "... nc dim_y"],
    ) -> Float[Array, "... dim_r"]:
        # Concatenate x and y along the feature dimension
        representation = jnp.concatenate([context_x, context_y], axis=-1)

        for layer in self.layers:
            representation = nnx.relu(layer(representation))

        # aggregate over the number of samples (penultimate dimension)
        return representation.mean(axis=-2)


class Decoder:
    def __init__(self, *, cnp_spec: CNPSpec, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(d_in, d_out, rngs=rngs)
            for d_in, d_out in cnp_spec.decoder_layer_dim_pairs
        ]

    def __call__(
        self,
        representation: Float[Array, "... dim_r"],
        target_x: Float[Array, "... nt dim_x"],
    ) -> dx.MultivariateNormalDiag:
        """Decodes individual targets."""
        # concatenate represenation with each target along the feature dimension
        # first to specify tiling shape need to read batch dimensions for target_x
        # tile_dims =
        tiled_representation = jnp.tile(
            representation, (1,) * (len(target_x.shape) - 2) + (target_x.shape[-2], 1)
        )
        combined_input = jnp.concatenate([tiled_representation, target_x], axis=-1)

        # pass through MLP
        for layer in self.layers[:-1]:
            combined_input = nnx.relu(layer(combined_input))

        # final layer
        locs, log_stds = jnp.split(self.layers[-1](combined_input), 2, axis=-1)

        return dx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.exp(log_stds))


class CNP(nnx.Module):

    def __init__(self, *, cnp_spec: CNPSpec, rngs: nnx.Rngs):
        self.encoder = Encoder(cnp_spec=cnp_spec, rngs=rngs)
        self.decoder = Decoder(cnp_spec=cnp_spec, rngs=rngs)

    def predictive(
        self,
        context_x: Float[Array, "... nc dim_x"],
        context_y: Float[Array, "... nc dim_y"],
        target_x: Float[Array, "... nt dim_x"],
    ) -> dx.MultivariateNormalDiag:
        representation = self.encoder(context_x, context_y)
        return self.decoder(representation, target_x)

    def log_prob(
        self,
        context_x: Float[Array, "... nc dim_x"],
        context_y: Float[Array, "... nc dim_y"],
        target_x: Float[Array, "... nt dim_x"],
        target_y: Float[Array, "... nt dim_y"],
    ) -> Float[Array, "..."]:
        return self.predictive(context_x, context_y, target_x).log_prob(target_y)
