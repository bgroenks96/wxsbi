from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import logging

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.infer import Predictive, SVI, TraceGraph_ELBO, MCMC, NUTS, HMCECS
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from numpyro.handlers import mask
from numpyro.contrib.control_flow import scan, cond

from scipy.stats import gamma

from . import wgen_glm_v2
from ..distributions import StochasticFunctionDistribution
from ..utils import fourier_lsq, extract_time_vars

class WGEN(ABC):
    def __init__(self, data: pd.DataFrame, model=wgen_glm_v2, *args, predictors=[], order=1, **kwargs):
        super().__init__()
        self.model = model
        self.model_args = args
        self.model_kwargs = kwargs
        self.order = order
        self.obs = self.get_obs(data)
        self.timestamps = extract_time_vars(data.index)
        self.initial_states = self.get_initial_states(self.obs, order)
        self.valid_idx = self._valid_indices(self.initial_states, self.obs, order)
        self.first_valid_idx = int(self.valid_idx[0])
        self.predictors = jnp.concat([jnp.ones((1, len(data.index), 1)), jnp.expand_dims(data[predictors].values, axis=0)], axis=-1)
        self.guide = None
    
    def _valid_indices(self, initial_states, obs, order):
        # filter out time indices where initial_states is NaN or Inf at time t and t+1;
        # here indices 0 and 1 correspond to the observed variable and batch dimensions respectively
        valid_mask_obs = jnp.stack([jnp.isfinite(v) for k,v in obs.items()]).any(axis=[0,1])
        # here indices 0, 2, and 3 correspond to the batch, lag, and var dimensions
        valid_mask_initial_states = jnp.isfinite(initial_states).all(axis=[0,2,3])
        valid_mask = jnp.logical_and(valid_mask_obs[order:], valid_mask_initial_states)
        valid_idx = jnp.where(jnp.logical_and(valid_mask[:-1], valid_mask[1:]))[0]
        if len(valid_idx) < initial_states.shape[1]:
            logging.warn(f"dropped {initial_states.shape[1] - len(valid_idx)} nan/inf timesteps")
        return valid_idx
    
    def get_obs(self, data: pd.DataFrame):
        return self.model.get_obs(data)
    
    def get_initial_states(self, data: pd.DataFrame, order):
        initial_states = self.model.get_initial_states(data, order=order)
        return initial_states
        
    def prior(self, **extra_kwargs):
        return self.model.prior(*self.model_args, order=self.order, num_predictors=self.predictors.shape[-1], **self.model_kwargs, **extra_kwargs)
    
    def step(self, timestamps=None, predictors=None, initial_states=None, batch_idx=0, prior_mask=True, subsample_time=None, **kwargs):
        """Evaluates the forward model at each time step independently. Time is used as the batch dimension in the forward evaluation.
           This should be the preferred method for parameter calibration.

        Args:
            timestamps (_type_, optional): _description_. Defaults to None.
            predictors (_type_, optional): _description_. Defaults to None.
            initial_states (_type_, optional): _description_. Defaults to None.
            batch_idx (int, optional): _description_. Defaults to 0.
            prior_mask (bool, optional): _description_. Defaults to True.
            subsample_time (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        obs = self.obs
        timestamps = timestamps if timestamps is not None else self.timestamps[batch_idx,:,:]
        predictors = predictors if predictors is not None else self.predictors[batch_idx,:,:]
        initial_states = initial_states if initial_states is not None else self.initial_states[batch_idx,:,:]
        order = self.order
        assert len(timestamps.shape) == 2 and timestamps.shape[1] == 4, "timestamps must have shape (timesteps, 4)"
        assert len(predictors.shape) == 2, "predictors must have shape (timesteps, variables)"
        assert predictors.shape[0] == initial_states.shape[0] + order, "leading time dimension for predictors and initial states must match"
        
        with mask(mask=prior_mask):
            step = self.prior(**kwargs)
            
        # plate over time dimension starting from the second index
        with numpyro.plate("time", len(self.valid_idx), subsample_size=subsample_time) as i:
            idx = self.valid_idx[i]
            inputs_i = jnp.concat([timestamps[idx+order,:], predictors[idx+order,:]], axis=-1)
            initial_state = initial_states[idx,:]
            obs_i = dict([(k,v[0,idx+order]) for k,v in obs.items()])
            # evaluate step function with time as the batch dimension
            pred_states, outputs = step(initial_state, inputs_i, obs_i)
            
        return pred_states, outputs
    
    def simulate(self, timestamps=None, predictors=None, initial_state=None, observable=None, batch_size=None, prior_mask=True, **kwargs):
        """Run the weather generator over the given timestamps and predictors starting from the given `initial_state`.

        Args:
            timestamps (_type_, optional): time stamps in the same format given by `extract_time_vars`. Defaults to None.
            predictors (_type_, optional): predictors for each time step. Defaults to None.
            initial_state (_type_, optional): initial state at start of rollout. Defaults to None.
            observable (_type_, optional): observable function. Defaults to None.
            batch_size (_type_, optional): batch size over which to broadcast predictors; defaults to the batch shape of the given predictors.
            prior_mask (bool, optional): True if the prior should be included in the log density of the model, False otherwise. Defaults to True.

        Returns:
            tuple[array,array | None]: outputs, observable
        """
        timestamps = timestamps if timestamps is not None else self.timestamps[:,self.first_valid_idx:,:]
        assert len(timestamps.shape) == 3 and timestamps.shape[2] == 4, "timestamps must have shape (batch_size, timesteps, 4)"
        predictors = predictors if predictors is not None else self.predictors[:,self.first_valid_idx:,:]
        if batch_size is None:
            batch_size = predictors.shape[0]
        else:
            predictors = predictors*jnp.ones((batch_size, 1, 1))
            timestamps = timestamps*jnp.ones((batch_size, 1, 1))
        initial_state = initial_state if initial_state is not None else self.initial_states[:,self.first_valid_idx,:,:]
        initial_state = initial_state*jnp.ones((batch_size, *initial_state.shape[1:]))
        # sample prior
        with mask(mask=prior_mask):
            step = self.prior(**kwargs)
            
        inputs = jnp.concat([timestamps, predictors], axis=-1)
        # scan over inputs with step/transition function;
        # note that we need to swap the batch and time dimension since scan runs over the leading axis.
        _, outputs = scan(step, initial_state, jnp.swapaxes(inputs, 0, 1))
        if observable is not None:
            obsv = observable(timestamps, *outputs)
            return outputs, obsv
        else:
            return outputs, None
        
    def simulator(self, timestamps=None, predictors=None, observable=None, rng_seed=0):
        """Constructs a wrapper function `f(theta)` that invokes `simulate` with the given `observable`.
           The parameters `theta` are assumed to be in the unconstrained (transformed) sample space of the model.
           Returns the simulator function as well as the corresponding prior distribution.

        Args:
            observable (_type_, optional): observable function for `simulate`. Defaults to None.
            rng_seed (int, optional): random seed. Defaults to 0.

        Returns:
            tuple: simulator_fn, prior
        """
        prior = StochasticFunctionDistribution(self.prior, unconstrained=True, rng_seed=rng_seed)
        timestamps = self.timestamps if observable is None else timestamps
        predictors = self.predictors if observable is None else predictors
        def simulator(_theta):
            theta = _theta if len(_theta.shape) > 1 else _theta.reshape((1,-1))
            batch_size = theta.shape[0]
            timestamps = timestamps*jnp.ones((batch_size, *timestamps.shape[1:]))
            predictors = predictors*jnp.ones((batch_size, *predictors.shape[1:]))
            params = {k: v for k,v in prior.constrain(theta, as_dict=True).items()}
            with numpyro.handlers.seed(rng_seed=rng_seed):
                with numpyro.handlers.condition(data=params):
                    if observable is not None:
                        outputs, obs = self.simulate(timestamps, predictors, observable=observable, batch_size=batch_size)
                        return obs.T
                    else:
                        outputs = self.simulate(timestamps, predictors)
                        return jnp.permute_dims(jnp.stack(outputs), (2,1,0))
        return simulator, prior
    
    def fit(self, method="svi", *args, **kwargs):
        if method == "svi":
            return self._fit_svi(*args, **kwargs)
        elif method == "hmcecs":
            return self._fit_hmcecs(*args, **kwargs)
        else:
            raise(Exception(f"fit method {method} not recognized"))
        
    def predict(self, fit_result, **kwargs):
        if fit_result is numpyro.infer.svi.SVIRunResult:
            return self._predict_svi(fit_result.params, **kwargs)
        else:
            raise(Exception("unrecognized fit_result type"))
        
    def _predict_svi(self, params, guide=None, timestamps=None, predictors=None, observable=None, num_samples=100, rng_seed=0):
        prng = jax.random.PRNGKey(rng_seed)
        guide = self.guide if guide is None else guide
        posterior_sampler = Predictive(guide, params=params, num_samples=num_samples)
        posterior_params = posterior_sampler(prng)
        sim, prior = self.simulator(timestamps, predictors, observable, rng_seed)
        theta = prior.dict2array(posterior_params)
        outputs = sim(theta)
        return outputs
    
    def _fit_svi(self, iterations=None, guide=None, optimizer=numpyro.optim.Adam(1e-3), rng=jax.random.PRNGKey(1234),
                 loss=TraceGraph_ELBO(), **kwargs):
        # if guide is not specified, default to multivariate normal
        if guide is None:
            guide = AutoMultivariateNormal(self.step, init_loc_fn=numpyro.infer.init_to_median, init_scale=0.1)
        self.guide = guide
        svi = SVI(self.step, guide, optimizer, loss=loss)
        if iterations is None:
            return svi.init(rng, **kwargs)
        else:
            return svi.run(rng, iterations, **kwargs)
        
    def _fit_hmcecs(self, num_samples=1000, num_warmup=1000, num_blocks=10, num_chains=4, chain_method="parallel",
                    init_strategy=numpyro.infer.init_to_median, svi_iterations=10000, rng=jax.random.PRNGKey(1234),
                    **kwargs):
        # use autodelta to get MAP estimate
        map_guide = AutoDelta(self.step, init_loc_fn=numpyro.infer.init_to_median)
        map_result, _ = self._fit_svi(svi_iterations, map_guide, rng=rng, **kwargs)
        with numpyro.handlers.seed(rng_seed=0):
            # get parameter dict; for some reason setting the seed is necessary...
            map_params = map_guide(map_result.params)
        # build taylor proxy around mdoe for HMCECS
        proxy = HMCECS.taylor_proxy(map_params)
        # set up and run HMCECS
        mcmc_kernel = HMCECS(NUTS(self.step, init_strategy=init_strategy), num_blocks=num_blocks, proxy=proxy)
        mcmc = MCMC(mcmc_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method)
        mcmc.run(rng, **kwargs)
        return mcmc

def estimate_wgen_params(data: pd.DataFrame, precip_name='prec', Tavg_name='Tair_mean', Tmin_name='Tair_min',
                         Tmax_name='Tair_max', Trange_name='Tair_range', Tskew_name='Tair_skew',
                         num_harmonics=1, prec_thresh=0.0):
    """Legacy function that estimates classic WGEN parameters from data.

    Args:
        data (_type_): _description_
        precip_name (str, optional): _description_. Defaults to 'pre'.
        Tavg_name (str, optional): _description_. Defaults to 'tavg'.
        Tmin_name (str, optional): _description_. Defaults to 'tmin'.
        Tmax_name (str, optional): _description_. Defaults to 'tmax'.
        Trange_name (str, optional): _description_. Defaults to 'trange'.
        Tskew_name (str, optional): _description_. Defaults to 'tskew'.
        num_harmonics (int, optional): _description_. Defaults to 1.
        prec_thresh (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    # precipitation
    prec = data[precip_name]
    prec_series = prec.reset_index().drop(["time"], axis=1)
    pdd_ref = (prec_series.iloc[:-1].reset_index() <= prec_thresh) & (prec_series.iloc[1:].reset_index() <= prec_thresh)
    pdd_ref["time"] = prec.index[1:]
    pwd_ref = (prec_series.iloc[:-1].reset_index() > prec_thresh) & (prec_series.iloc[1:].reset_index() <= prec_thresh)
    pwd_ref["time"] = prec.index[1:]
    pdd_ref_monthly = pdd_ref.drop(["index"], axis=1).set_index("time").resample("1ME").mean()
    pwd_ref_monthly = pwd_ref.drop(["index"], axis=1).set_index("time").resample("1ME").mean()
    prec_gamma_monthly = prec.resample("1ME").agg(lambda x: pd.DataFrame(np.array(gamma.fit(x[x > 0])).reshape((1,-1)), columns=["gamma_shape", "gamma_loc", "gamma_scale"]))
    ptot_monthly = prec.resample("1ME").sum()
    prec_avg_monthly = prec[prec > prec_thresh].resample("1ME").mean()
    wet_days_monthly = (prec > prec_thresh).resample("1ME").sum()
    # air temperature
    tavg = data[Tavg_name]
    tmin = data[Tmin_name]
    tmax = data[Tmax_name]
    trange = data[Tmax_name] - data[Tmin_name]
    tskew = (data[Tavg_name] - data[Tmin_name]) / trange
    tavg_wet_monthly = tavg[prec > prec_thresh].resample("1ME").mean()
    tmax_wet_monthly = tmax[prec > prec_thresh].resample("1ME").mean()
    tmin_wet_monthly = tmin[prec > prec_thresh].resample("1ME").mean()
    trange_wet_monthly = trange[prec > prec_thresh].resample("1ME").mean()
    tskew_wet_monthly = trange[prec > prec_thresh].resample("1ME").mean()
    tavg_dry_monthly = tavg[prec <= prec_thresh].resample("1ME").mean()
    tmax_dry_monthly = tmax[prec <= prec_thresh].resample("1ME").mean()
    tmin_dry_monthly = tmin[prec <= prec_thresh].resample("1ME").mean()
    trange_dry_monthly = trange[prec <= prec_thresh].resample("1ME").mean()
    tskew_dry_monthly = trange[prec <= prec_thresh].resample("1ME").mean()
    t_idx = jnp.arange(0, data.shape[0])
    f0 = 1/365.25
    freqs = jnp.arange(0,num_harmonics)*f0
    coef_tavg, ff = fourier_lsq(t_idx, jnp.array(tavg.values), freqs)
    Tavg_resid = tavg - jnp.dot(ff, coef_tavg)
    # fit fourier coefficients for range and skew in transformed space
    coef_trange, _ = fourier_lsq(t_idx, jnp.log(trange.values), freqs)
    coef_tskew, _ = fourier_lsq(t_idx, jax.scipy.special.logit(tskew.values), freqs)
    # aggregate
    agg_ops = {Tavg_name: 'mean', Tmin_name: 'mean', Tmax_name: 'mean', precip_name: 'sum'}
    data_monthly = data[[Tavg_name, Tmin_name, Tmax_name, precip_name]].resample("1ME").aggregate(agg_ops)
    trange_monthly = trange.resample("1ME").mean()
    tskew_monthly = tskew.resample("1ME").mean()
    para_monthly = pd.concat([
        data_monthly,
        trange_monthly.to_frame(name=Trange_name),
        tskew_monthly.to_frame(name=Tskew_name),
        tavg_wet_monthly.to_frame().rename({Tavg_name: f'{Tavg_name}_wet'}, axis=1),
        tmax_wet_monthly.to_frame().rename({Tmax_name: f'{Tmax_name}_wet'}, axis=1),
        tmin_wet_monthly.to_frame().rename({Tmin_name: f'{Tmin_name}_wet'}, axis=1),
        trange_wet_monthly.to_frame().rename({0: f'{Trange_name}_wet'}, axis=1),
        tskew_wet_monthly.to_frame().rename({0: f'{Tskew_name}_wet'}, axis=1),
        tavg_dry_monthly.to_frame().rename({Tavg_name: f'{Tavg_name}_dry'}, axis=1),
        tmax_dry_monthly.to_frame().rename({Tmax_name: f'{Tmax_name}_dry'}, axis=1),
        tmin_dry_monthly.to_frame().rename({Tmin_name: f'{Tmin_name}_dry'}, axis=1),
        trange_dry_monthly.to_frame().rename({0: f'{Trange_name}_dry'}, axis=1),
        tskew_dry_monthly.to_frame().rename({0: f'{Tskew_name}_dry'}, axis=1),
        pdd_ref_monthly.rename({precip_name: 'pdd'}, axis=1),
        pwd_ref_monthly.rename({precip_name: 'pwd'}, axis=1),
        prec_gamma_monthly.reset_index().drop('index', axis=1).set_index(data_monthly.index),
        ptot_monthly.to_frame().rename({precip_name: 'ptot'}, axis=1),
        prec_avg_monthly.to_frame().rename({precip_name: 'prec_avg'}, axis=1),
        wet_days_monthly.to_frame().rename({precip_name: 'wet_days'}, axis=1),
    ], axis=1)
    para = {
        'freqs': freqs,
        'prec': {
            'pdd': para_monthly.pdd,
            'pwd': para_monthly.pwd,
            'mean': para_monthly.prec_avg,
        },
        'Tair': {
            'coef': {'Tavg': coef_tavg, 'Trange': coef_trange, 'Tskew': coef_tskew},
            'mean': data_monthly[Tavg_name],
            'range': para_monthly[Trange_name],
            'skew': para_monthly[Tskew_name],
            'rho': Tavg_resid.autocorr(),
            'resid_std': Tavg_resid.std(),
            'max_offset_dry': (tmax_dry_monthly - tavg_dry_monthly).mean(),
            'max_offset_wet': (tmax_wet_monthly - tavg_wet_monthly).mean(),
            'min_offset_dry': (tmin_dry_monthly - tavg_dry_monthly).mean(),
            'min_offset_wet': (tmin_wet_monthly - tavg_wet_monthly).mean(),
        },
    }
    return para, para_monthly