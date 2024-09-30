import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from calendar import monthrange

from .distributions import BernoulliGamma

default_params = {
    "ptot_mean": 200.0,
    "ptot_std": 50.0,
    "ptot_monthly_means": jnp.ones(12),
    "precip_monthly_prob": 0.3*jnp.ones(12),
    "precip_monthly_disp": 50.0, # beta disperison; higher means less variance
    "precip_daily_min": 1.0,
}

def precip_hierarchical(year, params: dict=default_params):
    ptot = numpyro.sample("ptot", dist.TruncatedNormal(params["ptot_mean"], params["ptot_std"], low=0.0, high=jnp.inf))
    pr_month_w = numpyro.sample("pr_month_w", dist.Dirichlet(params["ptot_monthly_means"]))
    pr_month = pr_month_w*ptot
    
    for m in range(1,13):
        _, d_hi = monthrange(year, m)
        pr_prob_mean = params["precip_monthly_prob"][m-1]
        pr_prob_disp = params["precip_monthly_disp"]
        pr_prob = numpyro.sample(f"pr_prob_{m}", dist.Beta(pr_prob_mean*pr_prob_disp, (1-pr_prob_mean)*pr_prob_disp))
        pr_gamma_shape = numpyro.sample(f"pr_gamma_shape_{m}", dist.TruncatedNormal(1.0, 1.0, low=0.0, high=jnp.inf))
        # rate = p*shape/E[X]
        pr_gamma_rate = numpyro.deterministic(f"pr_gamma_rate_{m}", pr_prob*pr_gamma_shape/pr_month[m-1]*d_hi)
        with numpyro.plate("day", d_hi):
            precip = numpyro.sample(f"precip_{m}", BernoulliGamma(pr_prob, pr_gamma_shape, pr_gamma_rate))
    return precip
