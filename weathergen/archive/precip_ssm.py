import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan

from .distributions import BernoulliGamma

# This currently does not work because of an apparent limitation of numpyro;
# the issue seems to be related to the sampling of the latent state `z` in
# the state transition.
def wxssm_v1(ts, prec_obs=None, latent_state_dims=4, hidden_dims=32):
    k = latent_state_dims
    # transition matrix
    A = numpyro.sample("A", dist.MatrixNormal(jnp.eye(k), jnp.eye(k), jnp.eye(k)))
    # transition noise
    s = numpyro.sample("s", dist.LogNormal(0,1))
    # hidden layer weights
    Wh = numpyro.sample("Wh", dist.Normal(jnp.zeros((k,hidden_dims)), jnp.ones((k,hidden_dims))))
    # precip output weights
    Wp = numpyro.sample("Wp", dist.Normal(jnp.zeros((hidden_dims,3)), jnp.ones((hidden_dims,3))))
    # initial state
    z0 = numpyro.sample("z0", dist.Normal(jnp.zeros(k), jnp.ones(k)))
    
    def transition(z_prev, t):
        # z = numpyro.sample("z", dist.Normal(jnp.matmul(A, z_prev), s*jnp.ones(k)))
        z = jnp.matmul(A, z_prev)
        h = jax.nn.leaky_relu(jnp.matmul(z, Wh))
        bgp = jnp.matmul(h, Wp)
        prob = jax.nn.sigmoid(bgp[0])
        conc = jax.nn.softplus(bgp[1])
        rate = jax.nn.softplus(bgp[2])
        pr = numpyro.sample("prec", BernoulliGamma(prob, conc, rate), obs=prec_obs[t])
        return z, pr
    
    # return transition(z0, 0.0)
    return scan(transition, z0, ts)
