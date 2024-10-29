"""GP"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as r
import einops as eo

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

class RBFKernel(Kernel):
    """RBF Kernel class."""

    def __init__(self, lengthscale: Float):
        self.lengthscale = lengthscale
    
    def matrix(
        self, x1: Float[Array, "... N D"], x2: Float[Array, "... M D"]
    ) -> Float[Array, "... N M"]:
        # use trick below but make clearer
        #sqdist = jnp.sum(x1 ** 2, 1).reshape(-1, 1) + jnp.sum(x2 ** 2, 1) - 2 * jnp.dot(x1, x2.T)
        
        sqdist = (
            - 2 * jnp.einsum('... n d, ... m d -> ... n m', x1, x2)
            + eo.rearrange(jnp.einsum('... n d, ... n d -> ... n', x1, x1), '... n -> ... 1 n')
            + eo.rearrange(jnp.einsum('... m d, ... m d -> ... m', x2, x2), '... m -> ... m 1')
        )
        return jnp.exp(-0.5 * sqdist / (self.lengthscale ** 2) )

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

class Mean(ABC):
    """Mean class."""

    def __init__(self):
        pass

    @abstractmethod
    def vector(self, x: Float[Array, "... N D"]) -> Float[Array, "... N"]:
        """Returns the mean vector."""
        pass

class ZeroMean(Mean):
    """Mean of zero function"""
    def vector(self, x: Float[Array, "... N D"]) -> Float[Array, "... N"]:
        """Output shape looks like input shape but without the last dimension"""
        return jnp.zeros(x.shape[:-1])
    
class ConditionedMean(Mean):
    """Kernel obtained from conditioning"""
    def __init__(self, parent_kernel: Kernel, x: Float[Array, "... N D"], y: Float[Array, "... N"]):
        self.parent_kernel = parent_kernel
        K = parent_kernel.matrix(x,x)
        self.Km1y = jnp.linalg.inv(K) @ y
        self.x = x

    def vector(
        self, x1: Float[Array, "... N1 D"]
    ) -> Float[Array, "... N1 N2"]:
        K1 = self.parent_kernel(x1, self.x)
        return K1 @ self.Km1y




class GP:
    """Gaussian Process class."""

    def __init__(self, mean: Mean, kernel: Kernel):
        self.mean = mean
        self.kernel = kernel

    def sample(self, x: Float[Array, "... N D"], key: r.PRNGKey) -> Float[Array, "... N"]:
        """Returns a sample from the GP."""
        mean = self.mean.vector(x)
        K = self.kernel.matrix(x, x)
        return r.multivariate_normal(key=key, mean=mean, cov=K, method='svd')

    def condition(self, x: Float[Array, "... n d"], y: Float[Array, "... n"]):
        """Returns a GP defined by conditioning"""
        conditioned_kernel = ConditionedKernel(self.kernel, x)
        conditioned_mean = ConditionedMean(self.kernel, x, y)
        return GP(conditioned_mean, conditioned_kernel)



if __name__ == "__main__":
    print(GP)
