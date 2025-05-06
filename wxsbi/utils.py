import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

import torch
import torch.distributions as torchdist

from collections import OrderedDict
from functools import partial

from jax2torch.jax2torch import jax2torch, j2t, t2j

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
    
class Torch2NumPyroDistribution(dist.Distribution):
    def __init__(self, torch_dist):
        self.dist = torch_dist
        super().__init__(self)
    
    def sample(self, rng_key, shape):
        # set torch global rng seed;
        # ideally there should be a way to set it only locally,
        # but unfortunately pytorch only provides a global option
        torch.manual_seed(rng_key[-1])
        return t2j(self.dist.sample(shape))
    
    def log_prob(self, x):
        return t2j(self.dist.log_prob(j2t(x)))
