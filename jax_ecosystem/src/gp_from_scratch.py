import jax.numpy as jnp
from jaxtyping import Array, Float


class Kernel:
    """Kernel class."""

    def __init__(self):
        pass


class Mean:
    """Mean class."""

    def __init__(self):
        pass


class GP:
    """Gaussian Process class."""

    def __init__(self, mean: Mean, kernel: Kernel):
        pass

    def condition(self, x: Float[Array, "... n d"], y: Float[Array, "... n"]):
        """Returns a GP defined by conditioning"""
        pass


if __name__ == "__main__":
    print(GP)
