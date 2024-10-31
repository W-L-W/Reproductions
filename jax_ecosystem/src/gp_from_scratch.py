"""GP"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, TypeVar

import jax
import jax.numpy as jnp
import jax.random as r
import einops as eo

from jaxtyping import Array, Float
import distrax as dx
import optax as ox
import optax.tree_utils as otu

class Kernel(ABC):
    """Kernel class."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self, x1: Float[Array, "... n d"], x2: Float[Array, "... m d"]
    ) -> Float[Array, "... n m"]:
        """Returns the kernel matrix."""
        pass

class RBFKernel(Kernel):
    """RBF Kernel class."""

    def __init__(self, lengthscale: Float):
        self.lengthscale = lengthscale
    
    def __call__(
        self, x1: Float[Array, "... N D"], x2: Float[Array, "... M D"]
    ) -> Float[Array, "... N M"]:
        # use trick below but make clearer
        #sqdist = jnp.sum(x1 ** 2, 1).reshape(-1, 1) + jnp.sum(x2 ** 2, 1) - 2 * jnp.dot(x1, x2.T)
        
        sqdist = (
            - 2 * jnp.einsum('... n d, ... m d -> ... n m', x1, x2)
            + eo.rearrange(jnp.einsum('... n d, ... n d -> ... n', x1, x1), '... n -> ... n 1')
            + eo.rearrange(jnp.einsum('... m d, ... m d -> ... m', x2, x2), '... m -> ... 1 m')
        )
        
        return jnp.exp(-0.5 * sqdist / (self.lengthscale ** 2) )

class ConditionedKernel(Kernel):
    """Kernel obtained from conditioning"""
    def __init__(self, parent_kernel: Kernel, x: Float[Array, "N D"]):
        self.parent_kernel = parent_kernel
        self.K = parent_kernel(x,x)
        self.Km1 = jnp.linalg.inv
        self.x = x

    def __call__(
        self, x1: Float[Array, "N1 D"], x2: Float[Array, "N2 D"]
    ) -> Float[Array, "... N1 N2"]:
        K12 = self.parent_kernel(x1, x2)
        K1 = self.parent_kernel(x1, self.x)
        K2 = self.parent_kernel(x2, self.x)
        return K12 - K1 @ self.Km1 @ K2.T

class Mean(ABC):
    """Mean class."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: Float[Array, "... N D"]) -> Float[Array, "... N"]:
        """Returns the mean vector."""
        pass

class ZeroMean(Mean):
    """Mean of zero function"""
    def __call__(self, x: Float[Array, "... N D"]) -> Float[Array, "... N"]:
        """Output shape looks like input shape but without the last dimension"""
        return jnp.zeros(x.shape[:-1])
    

class ConditionedMean(Mean):
    """Kernel obtained from conditioning"""
    def __init__(self, parent_kernel: Kernel, x: Float[Array, "... N D"], y: Float[Array, "... N"]):
        self.parent_kernel = parent_kernel
        K = parent_kernel(x,x)
        # some notes:
        #  was better than jnp.linalg.inv(K) @ y
        self.Km1y = jnp.linalg.solve(K, y) #jnp.linalg.pinv(K, rtol=1e-4, hermitian=True) @ y
        self.x = x

    def __call__(
        self, x1: Float[Array, "... N1 D"]
    ) -> Float[Array, "... N1 N2"]:
        print(x1.shape, self.x.shape)
        K1 = self.parent_kernel(x1, self.x)
        return K1 @ self.Km1y




class GP:
    """Gaussian Process class."""

    def __init__(self, mean: Mean, kernel: Kernel, jitter: Float = 1e-6):
        self.mean = mean
        self.kernel = kernel
        self.jitter = jitter

    def mvn(self, x: Float[Array, "... N D"]):
        mean = self.mean(x)
        K = self.kernel(x, x)
        L = jnp.linalg.cholesky(K + self.jitter * jnp.eye(K.shape[0]))
        return dx.MultivariateNormalTri(loc=mean, scale_tri=L)

    def sample(self, x: Float[Array, "... N D"], key: r.PRNGKey) -> Float[Array, "... N"]:
        """Returns a sample from the GP."""
        return self.mvn(x).sample(seed=key)

    def log_prob(self, x: Float[Array, "... N D"], y: Float[Array, "... N"]):
        return self.mvn(x).log_prob(y)

    def condition(self, x: Float[Array, "... N D"], y: Float[Array, "... N"]):
        """Returns a GP defined by conditioning"""
        conditioned_kernel = ConditionedKernel(self.kernel, x)
        conditioned_mean = ConditionedMean(self.kernel, x, y)
        return GP(conditioned_mean, conditioned_kernel)


def run_lbfgs(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = ox.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = ox.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = otu.tree_get(state, 'count')
    grad = otu.tree_get(state, 'grad')
    err = otu.tree_l2_norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state


Params = TypeVar('param_space', bound=Dict[str, Any])
  
class ParametrisedGP(ABC):
    """Hierarchical Gaussian Process"""

    def __init__(self, param_log_prob: Callable[[Params], Float], spawn: Callable[[Params], GP], initializer: Callable[[r.PRNGKey], Params]):  
        """Note that log probabilities are un-normalised in general!"""
        self.param_log_prob = param_log_prob
        self.spawn = spawn
        self.initializer = initializer
    
    def compute_map_params_lbfgs(self, key: r.PRNGKey, max_iter: int = 1000, tol: float = 1e-6):
        opt = ox.lbfgs()
        init_params = self.initializer(key)
        loss = lambda params: -self.param_log_prob(params)

        # perform gradient ascent on the log density
        final_params, final_state = run_lbfgs(
            init_params, loss, opt, max_iter, tol
        )
        return final_params
    
    def sample_map(self, key: r.PRNGKey, max_iter: int = 1000, tol: float = 1e-6):
        params = self.compute_map_params_lbfgs(key, max_iter, tol)
        return self.spawn(params)
    
    
    def condition(self, x: Float[Array, "... N D"], y: Float[Array, "... N"]):
        """Returns the corresponding ParametrisedGP defined by conditioning"""
        # first get an updated log_prob
        def updated_log_prob(params):
            gp = self.spawn(params)
            # print('GP log_prob', gp.log_prob(x, y))
            # print('param log_prob', self.param_log_prob(params))
            return gp.log_prob(x, y) + self.param_log_prob(params)
        
        # def updated_log_prob(params: Iterable[Params]):
        #     """Vectorized version"""
        #     return jax.vmap(updated_log_prob_scalar)(params)
        
        # then get an updated spawn method using conditioning each spawned GP
        def updated_spawn(params):
            gp = self.spawn(params)
            return gp.condition(x, y)
        
        return ParametrisedGP(updated_log_prob, updated_spawn, self.initializer)


            




if __name__ == "__main__":
    print(GP)
