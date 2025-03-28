import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from . import utils

class AbstractEffects(ABC):
    def __init__(self, name: str, coef):
        assert len(coef.shape) == 1, "coefficients must be a vector!"
        self.name = name
        self.coef = coef
    
    def __len__(self):
        return len(self.coef)
    
    def get_coefs(self):
        return self.coef
    
    @abstractmethod
    def get_predictors(self, x):
        return NotImplemented

class LinearEffects(AbstractEffects):
    """
    Represents a set of linear effects in a generalied linear model.
    """
    
    def __init__(
        self,
        name: str,
        ndims: int,
        loc=0.0,
        scale_or_cov=1.0,
    ):
        # initialize prior mean and covariance for effects
        self.prior_loc = jnp.zeros(ndims) + loc
        self.prior_cov = jnp.eye(ndims)*scale_or_cov if len(jnp.shape(scale_or_cov)) < 2 else scale_or_cov
        # sample coefficients from isotropic normal
        coef = numpyro.sample(
            name,
            dist.MultivariateNormal(self.prior_loc, self.prior_cov)
        )
        super().__init__(name, coef)
    
    def get_predictors(self, x):
        assert x.shape[1] == len(self)
        return x
    
class SeasonalEffects(AbstractEffects):
    """
    Represents a set of seasonal effects, i.e. "Fourier features", with the given frequencies.
    The number of coefficients/feature-dimensions is equal to `2 * len(freqs)`.
    """
    
    def __init__(
        self,
        name: str,
        freqs=jnp.array([1/365.25]),
        loc=0.0,
        scale_or_cov=1.0,
    ):
        ndims = 2*len(freqs)
        self.freqs = freqs
        self.prior_loc = jnp.zeros(ndims) + loc
        self.prior_cov = jnp.eye(ndims)*scale_or_cov if len(jnp.shape(scale_or_cov)) < 2 else scale_or_cov
        coef = numpyro.sample(
            name,
            dist.MultivariateNormal(self.prior_loc, self.prior_cov)
        )
        super().__init__(name, coef)
    
    def get_predictors(self, t):
        assert len(t.shape) == 1, "input argument 't' must be a 1D vector"
        ff_t = utils.fourier_feats(t, self.freqs, intercept=False) * jnp.ones((t.shape[0], 1))
        return ff_t
    
class InteractionEffects(AbstractEffects):
    """
    Represents a set of linear interactions between two other sets of effects.
    """
    
    def __init__(
        self,
        name: str,
        e1: AbstractEffects,
        e2: AbstractEffects,
        loc=0.0,
        scale_or_cov=1.0,
    ):
        ndims = len(e1)*len(e2)
        self.e1 = e1
        self.e2 = e2
        self.prior_loc = jnp.zeros(ndims) + loc
        self.prior_cov = jnp.eye(ndims)*scale_or_cov if len(jnp.shape(scale_or_cov)) < 2 else scale_or_cov
        coef = numpyro.sample(
            name,
            dist.MultivariateNormal(self.prior_loc, self.prior_cov)
        )
        super().__init__(name, coef)
    
    def get_predictors(self, xy: Tuple):
        assert isinstance(xy, Tuple) and len(xy) == 2, "input to get_predictors for interaction effect must be a tuple of length 2"
        x, y = xy
        # outer product between predictors with batch dimension
        xy = jnp.einsum(
            "ij,ik->ijk",
            self.e1.get_predictors(x),
            self.e2.get_predictors(y)
        )
        # flatten both feature dimensions into one
        return xy.reshape((xy.shape[0], -1))

class LinkFunction(ABC):
    
    def __init__(self, fn, inv_fn):
        self.fn = fn
        self.inv_fn = inv_fn
        
    def __call__(self, x):
        return self.link(x)
    
    def link(self, x):
        return self.fn(x)
    
    def invlink(self, y):
        return self.inv_fn(y)
    
def identity(x):
    return x
    
class IdentityLink(LinkFunction):
    def __init__(self):
        super().__init__(identity, identity)
        
class LogLink(LinkFunction):
    def __init__(self):
        super().__init__(jnp.log, jnp.exp)
        
class LogitLink(LinkFunction):
    def __init__(self):
        super().__init__(jax.scipy.special.logit, jax.scipy.special.expit)

class GLM(ABC):
    
    def __init__(
        self,
        effect: AbstractEffects,
        *others: AbstractEffects,
        link: LinkFunction = IdentityLink(),
    ):
        assert isinstance(link, LinkFunction), "link function must be a subtype of LinkFunction"
        self.effects = [effect] + list(others)
        self.link = link
        super().__init__()
        
    def __call__(self, *xs):
        """Constructs predictors for all effects from the arguments and computes the dot product
        with the linear coefficients. Inputs must all be of shape `(batch,d)` where `d` is the
        number of feature dimensions. Interaction effects should be given as a tuple of inputs
        for each component effect.

        Returns:
            _type_: a tuple `(y, z)` where `y` is the linked predictand and `z` is the linear predictand.
        """
        assert len(xs) == len(self.effects), "Number of input arguments must match the number of effects."
        preds = jnp.concat([e.get_predictors(x) for e,x in zip(self.effects, xs)], axis=1)
        coefs = jnp.concat([e.get_coefs() for e in self.effects], axis=0)
        z = jnp.sum(preds*coefs, axis=1)
        return self.link.invlink(z), z
    