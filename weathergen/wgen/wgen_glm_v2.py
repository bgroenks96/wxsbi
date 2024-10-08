import pandas as pd

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import mask

from .. import utils, data
from ..distributions import BernoulliGamma, from_moments
        

def prior(num_predictors: int=1, pred_effect_scale=1.0,
          Tskew_scaled_dispersion_mean=1.0, Tair_freqs=[1/365.25], prec_freqs=[1/365.25], order=1, **kwargs):
    """Improved WGEN-GLM which generates daily weather variables according to the following procedure:
        1) Tavg(t) | Tavg(t-1), ...,Tavg(t-n)
        2) prec | prec(t-1), ..., prec(t-n), Tavg(t-1), ...,Tavg(t-n)
        3) Trange | prec, Tavg
        4) Tskew | prec, Tavg
        
        Mean daily air temperature is modelled as Student-T distribution with variable degrees of freedom.
        Precipitation is modelled as a bernoulli-Gamma mixture distribution.
        Daily air temperature range is modelled as a Gamma distribution.
        Daily air temperature skewness is modeled as a Beta distribution.
        
        Each of these observable variables are parameterized as GLMs defined over some set of linear predictors.

    Args:
        num_predictors (int, optional): number of exogeneous predictors. Defaults to 1.
        pred_effect_scale (float, optional): standard deviation of the predictor effect prior. Defaults to 1.0.
        Tskew_scaled_dispersion_mean (float, optional): prior mean of the Tskew dispersion parameter. Defaults to 1.0.
        Tair_freqs (list, optional): frequencies for air temperature seasonal effects. Defaults to the annual cycle: [1/365.25].
        prec_freqs (list, optional): frequencies for precipitation seasonal effects. Defaults to the annual cycle: [1/365.25].

    Returns:
        _type_: _description_
    """
    assert num_predictors > 0, "number of predictors must be greater than zero"
    
    # with numpyro.plate("batch", batch_size):
    # mean air temperature
    tair_mean_step = Tair_mean(num_predictors, pred_effect_scale, freqs=Tair_freqs, order=order, **kwargs)        
    # precipitation
    precip_step = precip(num_predictors, pred_effect_scale, freqs=prec_freqs, order=order, **kwargs)
    # air temperature range and skew
    tair_range_skew_step = Tair_range_skew(num_predictors, pred_effect_scale, Tskew_scaled_dispersion_mean, freqs=Tair_freqs, order=order, **kwargs)
    
    def step(state, inputs, obs={'prec': None, 'Tavg': None, 'Trange': None, 'Tskew': None}):
        assert state.shape[0] == inputs.shape[0], "state and input batch dimensions do not match"
        assert state.shape[1] == order, f"state lag dimension does not match order={order}"
        # unpack state and input tensors;
        # state is assumed to have shape (batch, lag, vars)
        prec_prev = state[:,:,0]
        Tavg_prev = state[:,:,1]
        i, year, month, doy = inputs[:,:4].T
        predictors = inputs[:,4:]
        # mean daily air temperature
        Tavg, Tavg_mean, Tavg_mean_seasonal = tair_mean_step(Tavg_prev, inputs, obs['Tavg'])
        Tavg_anom = (Tavg - Tavg_mean_seasonal).reshape((-1,1))
        # precipitation
        prec = precip_step((prec_prev, Tavg_anom), inputs, obs['prec'])
        # air temperature range and skew
        Trange, Tskew, Tmin, Tmax = tair_range_skew_step((Tavg, prec), inputs, obs['Trange'], obs['Tskew'])
        newstate = jnp.expand_dims(jnp.stack([prec, Tavg]).T, axis=1)
        return jnp.concat([state[:,1:,:], newstate], axis=1), (prec, Tmin, Tavg, Tmax)
    
    return step

def Tair_mean(num_predictors: int=1, pred_effect_scale=1.0, freqs=[1/365.25], order=1, **kwargs):
    ## autocorrelation for air temperature is bounded from [-1,1] to prevent divergence
    Tavg_lag_effects = numpyro.sample("Tavg_lag", dist.Uniform(-jnp.ones(order), jnp.ones(order)).to_event(1))
    ## fourier feature coefficients (seasonal effects)
    seasonal_dims = 2*len(freqs)
    Tavg_seasonal_effects = numpyro.sample("Tavg_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    Tavg_seasonal_lag1_effects = numpyro.sample("Tavg_seasonal&lag1", dist.MultivariateNormal(jnp.zeros(seasonal_dims), 0.5*jnp.eye(seasonal_dims)))
    Tavg_pred_effects = numpyro.sample("Tavg_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    Tavg_anom_effects = jnp.concat([Tavg_lag_effects, Tavg_seasonal_lag1_effects, Tavg_pred_effects], axis=-1)
    ## residual scale
    log_Tavg_scale_seasonal_effects = numpyro.sample("log_Tavg_scale_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    log_Tavg_scale_pred_effects = numpyro.sample("log_Tavg_scale_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    log_Tavg_scale_all_effects = jnp.concat([log_Tavg_scale_seasonal_effects, log_Tavg_scale_pred_effects], axis=-1)
    # Tavg_dof = numpyro.sample("Tavg_dof", dist.Exponential(1/Tavg_dof_mean))
    
    def step(state, inputs, Tavg_obs=None):
        Tavg_prev = state
        i, year, month, doy = inputs[:,:4].T
        predictors = inputs[:,4:]
        ff_t = utils.fourier_feats(i, freqs, intercept=False)*jnp.ones((Tavg_prev.shape[0],1))
        seasonal_lag1 = ff_t*Tavg_prev[:,-1:]
        # compute Tavg predictors for previous and current state
        Tavg_mean_seasonal = jnp.sum(ff_t*Tavg_seasonal_effects, axis=1)
        Tavg_mean_anom_features = jnp.concat([Tavg_prev, seasonal_lag1, predictors], axis=1)
        log_Tavg_scale_features = jnp.concat([ff_t, predictors], axis=1)
        Tavg_mean = numpyro.deterministic("Tavg_mean", Tavg_mean_seasonal + jnp.sum(Tavg_mean_anom_features*Tavg_anom_effects, axis=1))
        Tavg_scale = jnp.exp(jnp.sum(log_Tavg_scale_features*log_Tavg_scale_all_effects, axis=1))
        # sample daily mean air temperature including residual variance
        Tavg_mask = jnp.isfinite(Tavg_obs) if Tavg_obs is not None else True
        with mask(mask=Tavg_mask):
            Tavg = numpyro.sample("Tavg", dist.Normal(Tavg_mean, Tavg_scale), obs=Tavg_obs)
            # Tavg = numpyro.sample("Tavg", dist.StudentT(Tavg_dof, loc=Tavg_mean, scale=Tavg_scale), obs=Tavg_obs)
        return Tavg, Tavg_mean, Tavg_mean_seasonal
    
    return step

def precip(num_predictors: int=1, pred_effect_scale=1.0, freqs=[1/365.25], order=1, **kwargs):
    ## precipitation occurrence effects;
    ## note that the lag and Tavg effects are univariate but we use mvnormal so that the dimensions align
    seasonal_dims = 2*len(freqs)
    precip_occ_seasonal_effects = numpyro.sample("precip_occ_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)/seasonal_dims))
    precip_occ_lag_effects = numpyro.sample("precip_occ_lag", dist.Uniform(-jnp.ones(2*order), jnp.ones(2*order)).to_event(1))
    precip_occ_Tavg_effects = numpyro.sample("precip_occ_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1*jnp.eye(1)))
    precip_occ_pred_effects = numpyro.sample("precip_occ_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    # precip_occ_seasonal_lag_interaction = numpyro.sample("precip_occ_seasonal&lag", dist.Uniform(-jnp.ones(order), jnp.ones(order)).to_event(1))
    precip_occ_all_effects = jnp.concat([precip_occ_seasonal_effects, precip_occ_lag_effects, precip_occ_Tavg_effects, precip_occ_pred_effects], axis=-1)
    ## precipitation intensity effects
    precip_mean_seasonal_effects = numpyro.sample("precip_mean_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)/seasonal_dims))
    # precip_mean_Tavg_effects = numpyro.sample("precip_mean_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1*jnp.eye(1)))
    precip_mean_lag_effects = numpyro.sample("precip_mean_lag", dist.Uniform(-jnp.ones(2*order), jnp.ones(2*order)).to_event(1))
    precip_mean_seasonal_lag1_effects = numpyro.sample("precip_mean_seasonal&lag1", dist.Uniform(-jnp.ones(seasonal_dims), jnp.ones(seasonal_dims)).to_event(1))
    precip_mean_pred_effects = numpyro.sample("precip_mean_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    precip_mean_all_effects = jnp.concat([precip_mean_seasonal_effects, precip_mean_lag_effects, precip_mean_seasonal_lag1_effects, precip_mean_pred_effects], axis=-1)
    ## gamma shape parameter; smaller values imply more concentration near zero
    precip_gamma_shape = numpyro.sample("precip_gamma_shape", dist.Exponential(1.0))

    def step(state, inputs, prec_obs=None):
        prec_prev, Tavg_anom = state
        i, year, month, doy = inputs[:,:4].T
        predictors = inputs[:,4:]
        ff_t = utils.fourier_feats(i, freqs, intercept=False)*jnp.ones((prec_prev.shape[0],1))
        # construct precipitation features
        prev_wet = jnp.sign(prec_prev)
        log_prev_prec = jnp.log(1 + prec_prev)
        prec_occ_features = jnp.concat([ff_t, 1 - prev_wet, log_prev_prec, Tavg_anom, predictors], axis=1)
        prec_mean_features = jnp.concat([ff_t, 1 - prev_wet, log_prev_prec, ff_t*log_prev_prec[:,-1:], predictors], axis=1)
        # linear predictors
        eta_precip_occ = jnp.sum(prec_occ_features*precip_occ_all_effects, axis=1)
        eta_precip_mean = jnp.sum(prec_mean_features*precip_mean_all_effects, axis=1)
        # apply link functions
        p_wet = jax.nn.sigmoid(eta_precip_occ)
        precip_gamma_rate = numpyro.deterministic("precip_gamma_rate", precip_gamma_shape / jax.nn.softplus(eta_precip_mean))
        # sample precipication from bernoulli-gamma with prob p_wet
        # prec = numpyro.sample("prec", BernoulliGamma(p_wet, gamma_shape, gamma_rate), obs=obs['prec'])
        prec_occ_obs = prec_obs > 0.0 if prec_obs is not None else None
        prec_mask = prec_occ_obs if prec_occ_obs is not None else True
        with mask(mask=jnp.isfinite(prec_obs) if prec_obs is not None else True):
            prec_occ = numpyro.sample("prec_occ", dist.Bernoulli(p_wet), obs=prec_occ_obs)
        with mask(mask=prec_mask):
            prec_amount = numpyro.sample("prec_amount", dist.Gamma(precip_gamma_shape, precip_gamma_rate), obs=prec_obs)
        prec = numpyro.deterministic("prec", jnp.where(prec_occ, prec_amount, 0.0))
        return prec
    
    return step

def Tair_range_skew(num_predictors:int = 1, pred_effect_scale=1.0, Tskew_scaled_dispersion_mean=1.0,
                    freqs=[1/365.25], order=1, **kwargs):
    ## range mean
    seasonal_dims = 2*len(freqs)
    Trange_mean_seasonal_effects = numpyro.sample("Trange_mean_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    Trange_mean_Tavg_effects = numpyro.sample("Trange_mean_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1*jnp.eye(1)))
    Trange_mean_wet_effects = numpyro.sample("Trange_mean_wet", dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)))
    Trange_mean_pred_effects = numpyro.sample("Trange_mean_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    Trange_mean_all_effects = jnp.concat([Trange_mean_seasonal_effects, Trange_mean_Tavg_effects, Trange_mean_wet_effects, Trange_mean_pred_effects], axis=-1)
    ## range shape
    Trange_gamma_shape = numpyro.sample("Trange_gamma_shape", dist.Exponential(1.0))
    ## skew mean
    Tskew_seasonal_effects = numpyro.sample("Tskew_seasonal", dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)))
    Tskew_Tavg_effects = numpyro.sample("Tskew_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1*jnp.eye(1)))
    Tskew_wet_effects = numpyro.sample("Tskew_wet", dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)))
    Tskew_pred_effects = numpyro.sample("Tskew_pred", dist.MultivariateNormal(jnp.zeros(num_predictors), pred_effect_scale*jnp.eye(num_predictors)))
    Tskew_all_effects = jnp.concat([Tskew_seasonal_effects, Tskew_Tavg_effects, Tskew_wet_effects, Tskew_pred_effects], axis=-1)
    ## skew dispersion
    Tskew_scaled_dispersion = numpyro.sample("Tskew_scaled_dispersion", dist.Exponential(1/Tskew_scaled_dispersion_mean))
    Tskew_dispersion = Tskew_scaled_dispersion*100
    
    def step(state, inputs, Trange_obs=None, Tskew_obs=None):
        Tavg, prec = state
        i, year, month, doy = inputs[:,:4].T
        predictors = inputs[:,4:]
        ff_t = utils.fourier_feats(i, freqs, intercept=False)*jnp.ones((Tavg.shape[0],1))
        # wet/dry state for current time step
        is_wet = jnp.sign(prec).reshape((-1,1))
        # air temperature range and skew
        Trange_features = jnp.concat([ff_t, jnp.abs(Tavg.reshape((-1,1))), is_wet, predictors], axis=1)
        Tskew_features = jnp.concat([ff_t, Tavg.reshape((-1,1)), is_wet, predictors], axis=1)
        eta_Trange_mean = jnp.sum(Trange_features*Trange_mean_all_effects, axis=1)
        # parameterize Gamma distribution for range in terms of shape and mean
        Trange_rate = Trange_gamma_shape / jax.nn.softplus(eta_Trange_mean)
        Trange_mask = jnp.isfinite(Trange_obs) if Trange_obs is not None else True
        with mask(mask=Trange_mask):
            Trange = numpyro.sample("Trange", dist.Gamma(Trange_gamma_shape, Trange_rate), obs=Trange_obs)
        eta_Tskew = jnp.sum(Tskew_features*Tskew_all_effects, axis=1)
        Tskew_mean = jax.scipy.special.expit(eta_Tskew) # inverse logit
        Tskew_mask = jnp.isfinite(Tskew_obs) if Tskew_obs is not None else True
        with mask(mask=Tskew_mask):
            Tskew = numpyro.sample("Tskew", dist.Beta(Tskew_mean*Tskew_dispersion, (1-Tskew_mean)*Tskew_dispersion), obs=Tskew_obs)
        # calculate min and max from range and skew
        Tmin = numpyro.deterministic("Tmin", Tavg - Tskew*Trange)
        Tmax = numpyro.deterministic("Tmax", Tmin + Trange)
        return Trange, Tskew, Tmin, Tmax
        
    return step

def get_obs(data: pd.DataFrame):
    prec_obs = data["prec"]
    Tavg_obs = data["Tair_mean"]
    Trange_obs = data["Tair_max"] - data["Tair_min"]
    Tskew_obs = (Tavg_obs - data["Tair_min"]) / Trange_obs
    return {
        'prec': jnp.array(prec_obs.values).reshape((1,-1)),
        'Tavg': jnp.array(Tavg_obs.values).reshape((1,-1)),
        'Trange': jnp.array(Trange_obs.values).reshape((1,-1)),
        'Tskew': jnp.array(Tskew_obs.values).reshape((1,-1)),
    }
    
def get_initial_states(obs_or_shape: dict | tuple[int,int], order=1, dropna=True):
    if isinstance(obs_or_shape, tuple):
        # default to batch_size = 1 and one time step
        return jnp.zeros((*obs_or_shape,order,2))
    # initialize from observations
    obs = obs_or_shape
    prec_state = jnp.expand_dims(obs['prec'], axis=[-1,-2])
    Tavg_state = jnp.expand_dims(obs['Tavg'], axis=[-1,-2])
    state = jnp.concat([prec_state, Tavg_state], axis=-1)    
    # concatenate lagged states
    timelen = state.shape[1]
    return jnp.concat([state[:,i:timelen-order+i,:,:] for i in range(order)], axis=-2)