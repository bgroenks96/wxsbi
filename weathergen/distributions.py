import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.util import is_prng_key
from numpyro.distributions.util import promote_shapes

from collections import OrderedDict

def from_moments(disttype, mean, var):
    """
    Initialize a distribution of type `disttype` from first and second order moments (mean and variance).
    Currently, `Beta`, `Gamma`, and `LogNormal` are supported.
    """
    if disttype is dist.Beta:
        dispersion = mean*(1-mean) / var - 1
        a = mean*dispersion
        b = (1-mean)*dispersion
        return dist.Independent(dist.Beta(a, b), len(mean.shape))
    elif disttype is dist.Gamma:
        a = mean**2 / var
        b = mean / var
        return dist.Independent(dist.Gamma(a, b), len(mean.shape))
    elif disttype is dist.LogNormal:
        mu = jnp.log(mean / jnp.sqrt(var / mean**2 + 1))
        sigma = jnp.sqrt(jnp.log(var / mean**2 + 1))
        return dist.Independent(dist.LogNormal(mu, sigma), len(mean.shape))
    else:
        raise(Exception(f"{disttype} not recognized"))
    
def LogitNormal(mu, sigma):
    """
    Defines a `LogitNormal` distribution as a `TransformedDistribution` with Gaussian base
    and sigmoid transform.
    """
    return dist.TransformedDistribution(
        dist.Independent(dist.Normal(mu, sigma), len(mu.shape)),
        dist.transforms.SigmoidTransform()
    )

class BernoulliGamma(dist.Distribution):
    arg_constraints = {
        "prob": dist.constraints.unit_interval,
        "concentration": dist.constraints.positive,
        "rate": dist.constraints.positive,
    }
    support = dist.constraints.nonnegative
    reparametrized_params = ["concentration", "rate"]
    
    def __init__(self, prob, concentration, rate, *, validate_args=None):
        """Initialize a Bernoulli-Gamma mixture distribution with occurrence
        probability `prob` and Gamma parameters `concentration` and `rate`.

        Args:
            prob (_type_): Bernoulli occurrence probability.
            concentration (_type_): Gamma shape parameter.
            rate (float, optional): Gamma rate parameter.
            validate_args (_type_, optional): ???. Defaults to None.
        """        
        self.prob, self.concentration, self.rate = promote_shapes(prob, concentration, rate)
        self.bernoulli = dist.Bernoulli(prob)
        self.gamma = dist.Gamma(concentration, rate)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(prob), jnp.shape(concentration), jnp.shape(rate))
        super(BernoulliGamma, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )
    
    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        b = self.bernoulli.sample(key, sample_shape)
        y = self.gamma.sample(key, sample_shape)
        return b*y
    
    def log_prob(self, value):
        p = self.prob
        def lp(x):
            return jax.lax.cond(
                x > 0.0,
                lambda: jnp.log(p) + self.gamma.log_prob(x),
                lambda: jnp.log(1 - p),
            )
        vlp = jax.vmap(lp)
        return jnp.sum(vlp(value))
        
    @property
    def mean(self):
        return self.prob*self.gamma.mean
    
    @property
    def variance(self):
        return self.p*self.concentration*(1 + (1-self.p)*self.concentration)/self.rate^2
    
    def cdf(self, value):
        return jax.lax.cond(
            value > 0.0,
            lambda: 1 - self.prob + self.prob*self.gamma.cdf(value),
            lambda: 1 - self.prob,
        )
        
    def icdf(self, value):
        return jax.lax.cond(
            value > 0.0,
            lambda: self.gamma.icdf((value - 1 + self.prob) / self.prob),
            lambda: 0.0,
        )


class StochasticFunctionDistribution(dist.Distribution):
    """Allows a `numpyro` stochastic function/model to be represented as a product `Distribution`.
    All `sample` sites are considered random variables following their corresponding marginal densities.
    """
    
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
