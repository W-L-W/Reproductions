"""GP"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Array, Float


class Kernel(ABC):
    """Kernel class."""

    def __init__(self):
        pass

    @abstractmethod
    def matrix(
        self, x1: Float[Array, "... n d"], x2: Float[Array, "... m d"]
    ) -> Float[Array, "... n m"]:
        """Returns the kernel matrix."""
        pass


class Mean:
    """Mean class."""

    def __init__(self):
        pass

    @abstractmethod
    def vector(self, x: Float[Array, "... n d"]) -> Float[Array, "... n"]:
        """Returns the mean vector."""
        pass


def RBFKernel(Kernel):
    """RBF Kernel class."""

    def __init__(self, lengthscale: Float[Array, "..."]):
        self.lengthscale = lengthscale
    
    def matrix(
        self, x1: Float[Array, "... N D"], x2: Float[Array, "... M D"]
    ) -> Float[Array, "... N M"]:
        # use trick below but make clearer
        #sqdist = jnp.sum(x1 ** 2, 1).reshape(-1, 1) + jnp.sum(x2 ** 2, 1) - 2 * jnp.dot(x1, x2.T)
        sqdist = jnp.einsum('... n d, ... n d -> ... n 1', x1, x1)[:, None] + jnp.einsum('... m d, ... m d -> ... m', x2, x2)[None, :] - 2 * jnp.einsum('... n d, ... m d -> ... n m', x1, x2)
        return jnp.exp(-0.5 / self.lengthscale ** 2 * sqdist)
class GP:
    """Gaussian Process class."""

    def __init__(self, mean: Mean, kernel: Kernel):
        pass

    def condition(self, x: Float[Array, "... n d"], y: Float[Array, "... n"]):
        """Returns a GP defined by conditioning"""
        pass


if __name__ == "__main__":
    print(GP)
