import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

import torch
import torch.distributions as torchdist

from collections import OrderedDict
from functools import partial

from jax2torch.jax2torch import jax2torch, j2t, t2j

class StochasticFunctionDistribution(dist.Distribution):
    def __init__(self, fn, fn_args=(), fn_kwargs=dict(), rng_seed=0, unconstrained=False, validate_args=None):
        """
        Wraps a `numpyro` stochastic function and provides convenient `log_prob` and `sample` functions that allow
        it to be treated as a `Distribution`.
        """
        with numpyro.handlers.seed(rng_seed=rng_seed):
            tracer = numpyro.handlers.trace(fn)
            fn_trace = tracer.get_trace(*fn_args, **fn_kwargs)
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        self.params = [k for k,v in fn_trace.items() if v["type"] == "sample"]
        self.reconstructor = jax.flatten_util.ravel_pytree(OrderedDict([(k, fn_trace[k]["value"]) for k in self.params]))[1]
        self.unconstrained = unconstrained
        super(StochasticFunctionDistribution, self).__init__(
            batch_shape=(1,), event_shape=(len(self.params),), validate_args=validate_args
        )
        
    def set_args(self, *args, **kwargs):
        self.fn_args = args
        self.fn_kwargs = kwargs

    def log_prob(self, x):
        """Evaluate the log probability of `x` under the stochastic function.
        Note that if `unconstrained=True`, this method returns the potential energy;
        i.e. the log density without the Jacobian correction applied.

        Args:
            x (_type_): _description_
        """
        def lp(xi):
            params = self.reconstructor(xi)
            if self.unconstrained:
                neg_log_joint = numpyro.infer.util.potential_energy(self.fn, self.fn_args, self.fn_kwargs, params)
                return -neg_log_joint
            else:
                log_joint, _ = numpyro.infer.util.log_density(self.fn, self.fn_args, self.fn_kwargs, params)
                return log_joint
        # shape-dependent valuation of logprob function
        if len(x.shape) == 2:
            vlp = jax.vmap(lp)
            return vlp(x)
        elif len(x.shape) == 1:
            return lp(x)
        else:
            raise(Exception(f"invalid shape for sample: {x.shape}"))
    
    def sample(self, key=jax.random.PRNGKey(0), sample_shape=()):
        """Draw samples from the stochastic function according to `sample_shape`.

        Args:
            key (_type_, optional): _description_. Defaults to jax.random.PRNGKey(0).
            sample_shape (tuple, optional): _description_. Defaults to ().
        """
        def _sample_trace(_):
            tracer = numpyro.handlers.trace(self.fn)
            fn_trace = tracer.get_trace(*self.fn_args, **self.fn_kwargs)
            params = OrderedDict([(k, fn_trace[k]['value']) for k in self.params])
            x, _ = jax.flatten_util.ravel_pytree(params)
            return x
        # draw samples according to sample shape
        pad_shape = (1,) if sample_shape == () else sample_shape
        with numpyro.handlers.seed(rng_seed=key):
            vmap_sample = jax.vmap(_sample_trace, in_axes=tuple(range(len(pad_shape))))
            # draw vmapped samples
            samples = vmap_sample(jnp.ones(pad_shape))
            # reshape according to sample shape with the last dimension being the parameters
            if self.unconstrained:
                return self.unconstrain(samples.reshape((*sample_shape, samples.shape[-1])))
            else:
                return samples.reshape((*sample_shape, samples.shape[-1]))
        
    def unconstrain(self, x, as_dict=False):
        """Map the random variables of the stochastic function `x` to unconstrained
        space via `numpyro.infer.util.uconstrain_fn`.

        Args:
            x (_type_): _description_
            as_dict (bool, optional): _description_. Defaults to False.
        """
        def _unconstrain(x):
            constrained_params = self.reconstructor(x)
            unconstrained_params = numpyro.infer.util.unconstrain_fn(self.fn, self.fn_args, self.fn_kwargs, constrained_params)
            z, re = jax.flatten_util.ravel_pytree(OrderedDict([(p, unconstrained_params[p]) for p in self.params]))
            return re(z) if as_dict else z
        # vmap over batch dimensions if necesary
        if len(x.shape) > 1:
            vmap_uconstrain = jax.vmap(_unconstrain, in_axes=tuple(range(len(x.shape)-1)))
            return vmap_uconstrain(x)
        else:
            return _unconstrain(x)
        
    
    def constrain(self, z, as_dict=False):
        """Map the unconstrained random variables of the stochastic function `x` to
        the constrained sample space via `numpyro.infer.util.constrain_fn`.

        Args:
            z (_type_): _description_
            as_dict (bool, optional): _description_. Defaults to False.
        """
        def _constrain(z):
            unconstrained_params = self.reconstructor(z)
            constrained_params = numpyro.infer.util.constrain_fn(self.fn, self.fn_args, self.fn_kwargs, unconstrained_params)
            x, re = jax.flatten_util.ravel_pytree(OrderedDict([(p, constrained_params[p]) for p in self.params]))
            return re(x) if as_dict else x
        # vmap over batch dimensions if necesary
        if len(z.shape) > 1:
            vmap_uconstrain = jax.vmap(_constrain, in_axes=tuple(range(len(z.shape)-1)))
            return vmap_uconstrain(z)
        else:
            return _constrain(z)
        
    def dict2array(self, params: dict):
        """Converts the given dict of parameter samples to a matrix where the leading dimension is the batch dimension.
        Note that this method assumes the batch dimension to already be present in the parameter samples.
        """
        return jnp.concat([params[k] for k in self.params], axis=1)

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
