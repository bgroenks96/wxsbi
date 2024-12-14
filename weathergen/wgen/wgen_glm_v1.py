import pandas as pd

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from .. import utils

from ..distributions import BernoulliGamma, from_moments

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
    
def get_initial_states(obs: dict | tuple[int,int], prec_thresh=0.0):
    if isinstance(obs, tuple):
        return jnp.zeros((*obs, 2))
    wet_state = jnp.expand_dims(jnp.sign(obs['prec']), axis=-1)
    Tavg_state = jnp.expand_dims(obs['Tavg'], axis=-1)
    return jnp.concat([wet_state, Tavg_state], axis=-1)

def prior(num_predictors: int=1, gamma_shape_mean=0.5, **params):
    """GLM-based variant of the WGEN Markov-type weather generator. This version is currently deprecated.

    Args:
        num_predictors (int, optional): _description_. Defaults to 1.
        gamma_shape_mean (float, optional): _description_. Defaults to 0.5.
        params (kwargs): WGEN parameters as returned by `estimate_wgen_params`.

    Returns:
        _type_: _description_
    """
    assert num_predictors > 0, "number of predictors must be greater than zero"
    # precipitation
    ## precipitation occurrence effects
    seasonal_dims = 1 + 2*len(params["freqs"])
    precip_occ_seasonal_effects = numpyro.sample("precip_occ_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    precip_occ_wet_dry_effects = numpyro.sample("precip_occ_wet_dry", dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)))
    precip_occ_pred_effects = numpyro.sample("precip_occ_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.eye(num_predictors)))
    precip_occ_all_effects = jnp.concat([precip_occ_seasonal_effects, precip_occ_wet_dry_effects, precip_occ_pred_effects], axis=1)
    precip_mean_seasonal_effects = numpyro.sample("precip_mean_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    precip_mean_wet_dry_effects = numpyro.sample("precip_mean_wet_dry", dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)))
    precip_mean_pred_effects = numpyro.sample("precip_mean_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.eye(num_predictors)))
    precip_mean_all_effects = jnp.concat([precip_mean_seasonal_effects, precip_mean_wet_dry_effects, precip_mean_pred_effects], axis=1)
    ## gamma shape parameter; smaller values imply more concentration near zero
    gamma_shape = numpyro.sample("prec_shape", dist.Exponential(gamma_shape_mean))
    
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
    Tair_mean_pred_effects = numpyro.sample(
        "Tair_mean_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.eye(num_predictors))
    )
    Tair_mean_all_effects = jnp.concat([Tair_mean_seasonal_effects, Tair_mean_wet_dry_effects, Tair_mean_pred_effects], axis=1)
    ## Tair range parameters
    Tair_range_seasonal_effects = numpyro.sample(
        "Tair_range_seasonal",
        dist.MultivariateNormal(Tair_fourier_coef["Trange"], jnp.eye(Tair_fourier_coef["Trange"].shape[0]))
    )
    Tair_range_wet_dry_effects = numpyro.sample("Tair_range_wet_dry", dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)))
    Tair_range_pred_effects = numpyro.sample(
        "Tair_range_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.eye(num_predictors))
    )
    Tair_range_all_effects = jnp.concat([Tair_range_seasonal_effects, Tair_range_wet_dry_effects, Tair_range_pred_effects], axis=1)
    Tair_range_sigma = numpyro.sample("Tair_range_sigma", dist.Exponential(1.0))
    ## Tair skew parameters
    Tair_skew_seasonal_effects = numpyro.sample(
        "Tair_skew_seasonal",
        dist.MultivariateNormal(Tair_fourier_coef["Tskew"], jnp.eye(Tair_fourier_coef["Tskew"].shape[0])),
    )
    Tair_skew_wet_dry_effects = numpyro.sample(
        "Tair_skew_wet_dry",
        dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
    )
    Tair_skew_pred_effects = numpyro.sample(
        "Tair_skew_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.eye(num_predictors)),
    )
    Tair_skew_all_effects = jnp.concat([Tair_skew_seasonal_effects, Tair_skew_wet_dry_effects, Tair_skew_pred_effects], axis=1)
    Tair_skew_sigma = numpyro.sample("Tair_skew_sigma", dist.Exponential(1.0))
    ## residual scale
    Tair_resid_scale = numpyro.sample("Tair_resid_scale", dist.Exponential(1/params["Tair"]["resid_std"]))
        
    def step(state, inputs, obs={'prec': None, 'Tavg': None, 'Trange': None, 'Tskew': None}):
        assert state.shape[0] == inputs.shape[0], "state and input batch dimensions do not match"
        # unpack state and input tensors
        prev_wet, Tavg_prev = state.T
        i, year, month, doy = inputs[:,:4].T
        predictors = inputs[:,4:]
        # broadcast fourier features along batch dimension
        ff_t_prev = utils.fourier_feats(i-1, params["freqs"])*jnp.ones((state.shape[0],1))
        ff_t = utils.fourier_feats(i, params["freqs"])*jnp.ones((state.shape[0],1))
        # construct precipitation features
        prev_wet = prev_wet.reshape((-1,1))
        prev_wet_dry_feats = jnp.concat([prev_wet, 1 - prev_wet], axis=1)
        prec_features = jnp.concat([ff_t, prev_wet_dry_feats, predictors], axis=1)
        # linear predictors
        eta_precip_occ = jnp.sum(prec_features*precip_occ_all_effects, axis=1)
        eta_precip_mean = jnp.sum(prec_features*precip_mean_all_effects, axis=1)
        # apply link functions
        p_wet = jax.nn.sigmoid(eta_precip_occ)
        gamma_rate = gamma_shape / jnp.exp(eta_precip_mean)
        # sample precipication from bernoulli-gamma with prob p_wet
        prec = numpyro.sample("prec", BernoulliGamma(p_wet, gamma_shape, gamma_rate), obs=obs['prec'])
        # new precip state for current time step
        is_wet = jnp.sign(prec).reshape((-1,1))
        # compute predictors for previous and current state
        Tair_prev_features = jnp.concat([ff_t_prev, prev_wet, 1 - prev_wet, predictors], axis=1)
        Tair_prev_pred = jnp.sum(Tair_prev_features*Tair_mean_all_effects, axis=1)
        Tair_features = jnp.concat([ff_t, is_wet, 1 - is_wet, predictors], axis=1)
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
        # calculate min and max from range and skew
        Tmin = Tavg - Tskew*Trange
        Tmax = Tmin + Trange
        return jnp.stack([is_wet[:,0], Tavg]).T, (prec, Tmin, Tavg, Tmax)
    
    return step
