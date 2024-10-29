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


class RBFKernel(Kernel):
    """RBF Kernel class."""

    def __init__(self, lengthscale: Float[Array, "..."]):
        self.lengthscale = lengthscale
    
    def matrix(
        self, x1: Float[Array, "... N D"], x2: Float[Array, "... M D"]
    ) -> Float[Array, "... N M"]:
        # use trick below but make clearer
        #sqdist = jnp.sum(x1 ** 2, 1).reshape(-1, 1) + jnp.sum(x2 ** 2, 1) - 2 * jnp.dot(x1, x2.T)
        sqdist = (
            - 2 * jnp.einsum('... n d, ... m d -> ... n m', x1, x2)
            + jnp.einsum('... n d, ... n d -> ... n 1', x1, x1)
            + jnp.einsum('... m d, ... m d -> ... 1 m', x2, x2)
        )
        return jnp.exp(-0.5 / self.lengthscale ** 2 * sqdist)

class ConditionedKernel(Kernel):
    """Kernel obtained from conditioning"""
    def __init__(self, parent_kernel: Kernel, x: Float[Array, "... N D"]):
        self.parent_kernel = parent_kernel
        self.K = parent_kernel.matrix(x,x)
        self.Km1 = jnp.linalg.inv
        self.x = x

    def matrix(
        self, x1: Float[Array, "... N1 D"], x2: Float[Array, "... N2 D"]
    ) -> Float[Array, "... N1 N2"]:
        K12 = self.parent_kernel(x1, x2)
        K1 = self.parent_kernel(x1, x)
        K2 = self.parent_kernel(x2, x)
        return 

class GP:
    """Gaussian Process class."""

    def __init__(self, mean: Mean, kernel: Kernel):
        self.mean = mean
        self.kernel = kernel

    def condition(self, x: Float[Array, "... n d"], y: Float[Array, "... n"]):
        """Returns a GP defined by conditioning"""
        K11 = self.kernel.matrix(x,x)
        def create_matrix(x1, x2):



if __name__ == "__main__":
    print(GP)
