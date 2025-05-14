from functools import partial

import jax
from jax2torch.jax2torch import j2t, jax2torch, t2j


class NumPyro2TorchDistribution:
    def __init__(self, numpyro_dist, rng_key=jax.random.PRNGKey(0)):
        """Simple wrapper around a numpyro distribution that converts input tensors
        to JAX numpy arrays and vice versa for outputs.

        Args:
            numpyro_dist (_type_): _description_
            rng_key (_type_, optional): _description_. Defaults to jax.random.PRNGKey(0).
        """
        self.dist = numpyro_dist
        self.rng_key = rng_key
        self.log_prob = jax2torch(self._log_prob)

    def sample(self, shape):
        return j2t(self.dist.sample(self.rng_key, tuple(shape)))

    @partial(jax.jit, static_argnums=[0])
    def _log_prob(self, x):
        return self.dist.log_prob(x)
