from dataclasses import dataclass
from typing import Iterable
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
from jaxtyping import Array, Float
import distrax as dx
from typing import NamedTuple
import matplotlib.pyplot as plt
import einops as eo


# Constants
# These were included as numerical quantities in the original notebook
NUM_TARGETS_EVAL_MODE = 50
X_MIN = -2.0
X_MAX = 2.0


class Query(NamedTuple):
    context_x: Float[Array, "... nc dim_x"]
    context_y: Float[Array, "... nc dim_y"]
    target_x: Float[Array, "... nt dim_x"]


class CNPRegressionInstance(NamedTuple):
    query: Query
    target_y: Float[Array, "... nt dim_y"]


# class GPCurvesReaderOld:
#     """Generates curves using a Gaussian Process (GP).
#     Supports vector inputs (x) and vector outputs (y).
#     """

#     def __init__(
#         self,
#         batch_size: int,
#         max_num_context: int,
#         x_dim=1,
#         y_dim: int = 1,
#         lengthscale: float = 0.4,
#         sigma_scale: float = 1.0,
#         sigma_noise: float = 0.1,
#     ):
#         """Creates a regression dataset of functions sampled from a GP.
#         Args:
#           batch_size: An integer.
#           max_num_context: The max number of observations in the context.
#           x_dim: Dimension of input space.
#           y_dim: Dimension of output space.
#           lengthscale: Float, the lengthscale of the RBF kernel in input space.
#           sigma_scale: Float, the scale of standard deviation in the prior
#           sigma_noise: Float, the scale of the observation noise.
#         """
#         self._batch_size = batch_size
#         self._max_num_context = max_num_context
#         self._x_dim = x_dim
#         self._y_dim = y_dim
#         self._lengthscale = lengthscale
#         self._sigma_scale = sigma_scale
#         self._sigma_noise = sigma_noise

#         kernel = gpx.kernels.RBF(
#             lengthscale=self._lengthscale,
#             variance=self._sigma_scale**2,
#             n_dims=self._x_dim,
#         )
#         mean = gpx.mean_functions.Zero()
#         self.prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)

#     def _sample_split_sizes(self, eval_mode: bool = False, *, key: jr.PRNGKey):
#         """Determine number of context and target points for regression instance."""
#         context_key, target_key = jr.split(key, num=2)
#         num_context = int(
#             jr.uniform(
#                 context_key, shape=(), minval=3, maxval=self._max_num_context + 1
#             )
#         )
#         if eval_mode:
#             num_target = NUM_TARGETS_EVAL_MODE
#         else:
#             num_target = int(
#                 jr.uniform(
#                     target_key, shape=(), minval=2, maxval=self._max_num_context + 1
#                 )
#             )
#         num_total = num_context + num_target
#         return num_context, num_target, num_total

#     def generate_regression_instance(
#         self, eval_mode: bool = False, *, key: jr.PRNGKey
#     ) -> CNPRegressionInstance:
#         """Returns regression instance.
#         This has a batch dimension of size `batch_size`; each sample in the batch has the same
#         combined set of input (x) values, but different randomly generated target (y) values.
#         """
#         key_num_samples, key_x, key_y = jr.split(key, num=3)

#         num_context, num_target, num_total = self._sample_split_sizes(
#             eval_mode=eval_mode, key=key_num_samples
#         )
#         # Each sample is from a distribution whose coordinates are independent uniforms
#         shared_x_values = jr.uniform(
#             key=key_x,
#             shape=(num_total, self._x_dim),
#             minval=X_MIN,
#             maxval=X_MAX,
#         )
#         # Sample from the GP
#         y_dist = self.prior.predict(shared_x_values)
#         y_values_flat = y_dist.sample(seed=key_y, sample_shape=(self._batch_size,))

#         # Reshape the samples
#         x_values = jnp.tile(
#             eo.rearrange(shared_x_values, "n d -> 1 n d"), reps=(self._batch_size, 1, 1)
#         )
#         y_values = eo.rearrange(y_values_flat, "b n -> b n 1")

#         # Split into context and target
#         context_x = x_values[:, :num_context, :]
#         context_y = y_values[:, :num_context, :]
#         target_x = x_values[:, num_context:, :]
#         target_y = y_values[:, num_context:, :]

#         query = Query(context_x=context_x, context_y=context_y, target_x=target_x)
#         return CNPRegressionInstance(query=query, target_y=target_y)


class GPCurvesReader:
    """Generates curves using a Gaussian Process (GP).
    Supports vector inputs (x) and vector outputs (y).
    """

    def __init__(
        self,
        batch_size: int,
        max_num_context: int,
        x_dim=1,
        y_dim: int = 1,
        lengthscale: float = 0.4,
        sigma_scale: float = 1.0,
        sigma_noise: float = 0.1,
        jitter: float = 1e-6,
    ):
        """Creates a regression dataset of functions sampled from a GP.
        Args:
          batch_size: An integer.
          max_num_context: The max number of observations in the context.
          x_dim: Dimension of input space.
          y_dim: Dimension of output space.
          lengthscale: Float, the lengthscale of the RBF kernel in input space.
          sigma_scale: Float, the scale of standard deviation in the prior
          sigma_noise: Float, the scale of the observation noise.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._lengthscale = lengthscale
        self._sigma_scale = sigma_scale
        self._sigma_noise = sigma_noise
        self._sigma_noise_jittered = max(sigma_noise, jitter * sigma_scale)

    def _mvn(self, shared_x_values: Float[Array, "n d"]) -> dx.MultivariateNormalTri:
        """Return multivariate normal with mean 0 and covariance from RBF kernel.
        (Could use cola, but simpler to go direct for now)"""
        n = shared_x_values.shape[0]
        K = jnp.exp(
            -0.5
            * jnp.square(
                jnp.linalg.norm(
                    shared_x_values[:, None, :] - shared_x_values[None, :, :],
                    axis=-1,
                )
                / self._lengthscale
            )
        )

        scaled_K = K + jnp.eye(n) * self._sigma_noise_jittered**2

        return dx.MultivariateNormalTri(
            loc=jnp.zeros(n),
            scale_tri=jnp.linalg.cholesky(scaled_K),
        )

    def _sample_split_sizes(self, eval_mode: bool = False, *, key: jr.PRNGKey):
        """Determine number of context and target points for regression instance."""
        context_key, target_key = jr.split(key, num=2)
        num_context = int(
            jr.uniform(
                context_key, shape=(), minval=3, maxval=self._max_num_context + 1
            )
        )
        if eval_mode:
            num_target = NUM_TARGETS_EVAL_MODE
        else:
            num_target = int(
                jr.uniform(
                    target_key, shape=(), minval=2, maxval=self._max_num_context + 1
                )
            )
        num_total = num_context + num_target
        return num_context, num_target, num_total

    def generate_regression_instance(
        self, eval_mode: bool = False, *, key: jr.PRNGKey
    ) -> CNPRegressionInstance:
        """Returns regression instance.
        This has a batch dimension of size `batch_size`; each sample in the batch has the same
        combined set of input (x) values, but different randomly generated target (y) values.
        """
        key_num_samples, key_x, key_y = jr.split(key, num=3)

        num_context, _, num_total = self._sample_split_sizes(
            eval_mode=eval_mode, key=key_num_samples
        )
        # Each sample is from a distribution whose coordinates are independent uniforms
        shared_x_values = jr.uniform(
            key=key_x,
            shape=(num_total, self._x_dim),
            minval=X_MIN,
            maxval=X_MAX,
        )
        # Sample from the GP
        y_dist = self._mvn(shared_x_values)
        y_values_flat = y_dist.sample(seed=key_y, sample_shape=(self._batch_size,))

        # Reshape the samples
        x_values = jnp.tile(
            eo.rearrange(shared_x_values, "n d -> 1 n d"), reps=(self._batch_size, 1, 1)
        )
        y_values = eo.rearrange(y_values_flat, "b n -> b n 1")

        # Split into context and target
        context_x = x_values[:, :num_context, :]
        context_y = y_values[:, :num_context, :]
        target_x = x_values[:, num_context:, :]
        target_y = y_values[:, num_context:, :]

        query = Query(context_x=context_x, context_y=context_y, target_x=target_x)
        return CNPRegressionInstance(query=query, target_y=target_y)


@dataclass
class CNPSpec:
    """CNP specification."""

    x_dim: int
    y_dim: int
    r_dim: int
    encoder_hidden_output_sizes: Iterable[int]
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
        full_sizes = (
            [self.encoder_input_size] + self.encoder_hidden_output_sizes + [self.r_dim]
        )
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
        nt = target_x.shape[-2]

        tiled_representation = jnp.tile(
            eo.rearrange(representation, "... d -> ... 1 d"),
            (1,) * (len(target_x.shape) - 2) + (nt, 1),
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
        query: Query,
    ) -> dx.MultivariateNormalDiag:
        representation = self.encoder(query.context_x, query.context_y)
        return self.decoder(representation, query.target_x)

    def log_prob(
        self,
        regression_instance: CNPRegressionInstance,
    ) -> Float[Array, "..."]:
        return self.predictive(regression_instance.query).log_prob(
            regression_instance.target_y
        )


def plot_functions(
    query: Query,
    target_y: Float[Array, "... nt dim_y"],
    pred_y_mean: Float[Array, "... nt dim_y"],
    pred_y_std: Float[Array, "... nt dim_y"],
):
    """Plot predicted mean and variance as well as context and target points.
    Extracts only the first batch element.
    """
    # check that dimensionality of input output space is in fact 1
    msg = "Only 1D input and output space supported for this plotting utility"
    assert (query.context_x.shape[-1] == 1) and (query.target_x.shape[-1] == 1), msg
    # Sort context points
    sorted_context_indices = jnp.argsort(query.context_x[0, :, 0])
    xc = query.context_x[0, sorted_context_indices, 0]
    yc = query.context_y[0, sorted_context_indices, 0]

    # Sort target points
    sorted_target_indices = jnp.argsort(query.target_x[0, :, 0])
    xt = query.target_x[0, sorted_target_indices, 0]
    yt = target_y[0, sorted_target_indices, 0]
    ym = pred_y_mean[0, sorted_target_indices, 0]
    ys = pred_y_std[0, sorted_target_indices, 0]

    plt.plot(xt, ym, "b", linewidth=2, label="Mean")
    plt.plot(xt, yt, "kx", linewidth=2, label="Target points")
    plt.plot(xc, yc, "ko", markersize=10, label="Context points")
    plt.fill_between(xt, ym - ys, ym + ys, alpha=0.2, color="blue", interpolate=True)

    # Make the plot pretty
    # plt.yticks([X_MIN, 0, X_MAX], fontsize=16)
    plt.grid("off")
    ax = plt.gca()
    # ax.set_axis_bgcolor("white")
    ax.legend()
    plt.show()
