from functools import partial, wraps
from inspect import signature

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
from jax import dlpack as jax_dlpack
from jax.tree_util import tree_map
from packaging import version
from torch.utils import dlpack as torch_dlpack


def j2t(x_jax):
    # https://github.com/lucidrains/jax2torch/issues/6

    if version.parse(jax.__version__) < version.parse("0.4.13"):
        x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    else:
        x_torch = torch_dlpack.from_dlpack(x_jax)
    return x_torch


def t2j(x_torch):
    # https://github.com/lucidrains/jax2torch/issues/6
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082

    if version.parse(jax.__version__) < version.parse("0.4.13"):
        x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    else:
        x_jax = jax_dlpack.from_dlpack(x_torch)
    return x_jax


def tree_t2j(x_torch):
    return tree_map(lambda t: t2j(t) if isinstance(t, torch.Tensor) else t, x_torch)


def tree_j2t(x_jax):
    return tree_map(lambda t: j2t(t) if isinstance(t, jnp.ndarray) else t, x_jax)


def jax2torch(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_t2j(args)
                y_, ctx.fun_vjp = jax.vjp(fn, *args)
                return tree_j2t(y_)

            @staticmethod
            def backward(ctx, *grad_args):
                grad_args = (
                    tree_t2j(grad_args) if len(grad_args) > 1 else t2j(grad_args[0])
                )
                grads = ctx.fun_vjp(grad_args)
                grads = tuple(
                    map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads)
                )
                return tree_j2t(grads)

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return JaxFun.apply(*bound.arguments.values())

    return inner


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
