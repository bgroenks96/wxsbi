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
