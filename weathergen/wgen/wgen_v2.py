import pandas as pd

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from .. import utils

from ..distributions import BernoulliGamma, LogitNormal, from_moments

def prior(batch_size: int=1, gamma_shape_mean=0.5, **params):
    """Modified version of the Richardson (1981) WGEN Markov-type weather generator.

    Args:
        Tair_offset_var (float, optional): _description_. Defaults to 1.0.
        gamma_shape_mean (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    with numpyro.plate("batch", batch_size):
        # precipitation
        ## P(dry|dry)
        pdd_months = params["prec"]["pdd"].groupby(params["prec"]["pdd"].index.month)
        pdd = numpyro.sample("pdd", from_moments(dist.Beta, pdd_months.mean().values, pdd_months.var().values))
        ## P(wet|dry)
        pwd_months = params["prec"]["pwd"].groupby(params["prec"]["pwd"].index.month)
        pwd = numpyro.sample("pwd", from_moments(dist.Beta, pwd_months.mean().values, pwd_months.var().values))
        ## mean precipitation amount on wet days
        prec_mean_months = params["prec"]["mean"].groupby(params["prec"]["mean"].index.month)
        prec_mean = numpyro.sample("prec_mean", from_moments(dist.Gamma, prec_mean_months.mean().values, prec_mean_months.var().values))
        ## gamma shape parameter; smaller values imply more concentration near zero
        gamma_shape = numpyro.sample("prec_shape", dist.Independent(dist.Exponential(gamma_shape_mean*jnp.ones(12)), 1))
        ## derive gamma rate parameter from the shape and mean
        gamma_rate = gamma_shape / prec_mean
        
        # air temperature
        ## autocorrelation coefficient
        Tair_rho = numpyro.sample("Tair_rho", dist.Beta(params["Tair"]["rho"]*10, (1-params["Tair"]["rho"])*10))
        ## fourier feature coefficients centered on least squares values
        Tair_fourier_coef = params["Tair"]["coef"]
        Tair_mean_seasonal_effects = numpyro.sample(
            "Tair_mean_seasonal",
            dist.MultivariateNormal(Tair_fourier_coef["Tavg"], jnp.eye(Tair_fourier_coef["Tavg"].shape[0]))
        )
        ## wet vs. dry effects; note that this is slightly different than the formulation of Richardson.
        Tair_mean_wet_dry_effects = numpyro.sample(
            "Tair_mean_wet_dry", 
            dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        )
        Tair_mean_all_effects = jnp.concat([Tair_mean_seasonal_effects, Tair_mean_wet_dry_effects], axis=1)
        ## Tair range parameters
        Tair_range_seasonal_effects = numpyro.sample(
            "Tair_range_seasonal",
            dist.MultivariateNormal(Tair_fourier_coef["Trange"],jnp.eye(Tair_fourier_coef["Trange"].shape[0]))
        )
        Tair_range_wet_dry_effects = numpyro.sample(
            "Tair_range_wet_dry",
            dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        )
        Tair_range_all_effects = jnp.concat([Tair_range_seasonal_effects, Tair_range_wet_dry_effects], axis=1)
        Tair_range_sigma = numpyro.sample("Tair_range_sigma", dist.HalfNormal(0.2))
        ## Tair skew parameters
        Tair_skew_seasonal_effects = numpyro.sample(
            "Tair_skew_seasonal",
            dist.MultivariateNormal(Tair_fourier_coef["Tskew"], jnp.eye(Tair_fourier_coef["Tskew"].shape[0]))
        )
        Tair_skew_wet_dry_effects = numpyro.sample(
            "Tair_skew_wet_dry",
            dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        )
        Tair_skew_all_effects = jnp.concat([Tair_skew_seasonal_effects, Tair_skew_wet_dry_effects], axis=1)
        Tair_skew_sigma = numpyro.sample("Tair_skew_sigma", dist.HalfNormal(0.2))
        ## residual scale
        Tair_resid_scale = numpyro.sample("Tair_resid_scale", dist.HalfNormal(1/params["Tair"]["resid_std"]))
        
    def step(state, inputs, obs={'prec': None, 'Tavg': None, 'Trange': None, 'Tskew': None}):
        assert state.shape[0] == inputs.shape[0], "state and input batch dimensions do not match"
        prev_wet, Tavg_prev = state.T
        i, year, month, doy = inputs[:,:4].T
        month = jnp.astype(month, jnp.int32)
        # generate fourier features
        ff_t_prev = utils.fourier_feats(i-1, params["freqs"])*jnp.ones((state.shape[0],1))
        ff_t = utils.fourier_feats(i, params["freqs"])*jnp.ones((state.shape[0],1))
        # we need to select the m'th index along each batch dimension;
        # there is probably a more elegant way to do this...
        pdd_m = utils.batch_select(pdd, month)
        pwd_m = utils.batch_select(pwd, month)
        gamma_shape_m = utils.batch_select(gamma_shape, month)
        gamma_rate_m = utils.batch_select(gamma_rate, month)
        # determine step prob from previous state
        p_dry = prev_wet*pwd_m + (1-prev_wet)*pdd_m
        # sample precipication from bernoulli-gamma with prob 1 - p_dry
        prec = numpyro.sample("prec", BernoulliGamma(1 - p_dry, gamma_shape_m, gamma_rate_m), obs=obs['prec'])
        is_wet = jnp.sign(prec).reshape((-1,1))
        # compute predictors for previous and current state
        Tair_prev_features = jnp.concat([ff_t_prev, prev_wet.reshape((-1,1)), 1 - prev_wet.reshape((-1,1))], axis=1)
        Tair_prev_pred = jnp.sum(Tair_prev_features*Tair_mean_all_effects, axis=1)
        Tair_features = jnp.concat([ff_t, is_wet, 1 - is_wet], axis=1)
        Tair_pred = jnp.sum(Tair_features*Tair_mean_all_effects, axis=1)
        Tavg_mean = Tair_pred + Tair_rho*(Tavg_prev - Tair_prev_pred)
        # sample daily mean air temperature including residual variance
        Tavg = numpyro.sample("Tavg", dist.Normal(Tavg_mean, Tair_resid_scale), obs=obs['Tavg'])
        # sample range and skew
        eta_Trange = jnp.sum(Tair_features*Tair_range_all_effects, axis=1)
        Trange = numpyro.sample("Trange", dist.LogNormal(eta_Trange, Tair_range_sigma), obs=obs['Trange'])
        eta_Tskew = jnp.sum(Tair_features*Tair_skew_all_effects, axis=1)
        Tskew_mean = jax.scipy.special.expit(eta_Tskew) # inverse logit
        Tskew = numpyro.sample("Tskew", from_moments(dist.Beta, Tskew_mean, Tair_skew_sigma), obs=obs['Tskew'])
        Tmin = Tavg - Tskew*jnp.maximum(Trange, 1.0)
        Tmax = Tmin + Trange
        return jnp.stack([is_wet[:,0], Tavg]).T, (prec, Tmin, Tavg, Tmax)
    
    return step

def get_obs(df: pd.DataFrame, prec_name='pre', Tavg_name='tavg', Tmin_name='tmin', Tmax_name='tmax'):
    prec_obs = df[prec_name]
    Tavg_obs = df[Tavg_name]
    Trange_obs = df[Tmax_name] - df[Tmin_name]
    Tskew_obs = (Tavg_obs - df[Tmin_name]) / Trange_obs
    return {
        'prec': jnp.array(prec_obs.values).reshape((1,-1)),
        'Tavg': jnp.array(Tavg_obs.values).reshape((1,-1)),
        'Trange': jnp.array(Trange_obs.values).reshape((1,-1)),
        'Tskew': jnp.array(Tskew_obs.values).reshape((1,-1)),
    }
    
def get_initial_states(obs: dict, prec_thresh=0.0):
    wet_state = jnp.expand_dims(jnp.sign(obs['prec']), axis=-1)
    Tavg_state = jnp.expand_dims(obs['Tavg'], axis=-1)
    return jnp.concat([wet_state, Tavg_state], axis=-1)
