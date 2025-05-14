import logging
from abc import ABC
from typing import List, Tuple

import h5py as h5
import jax
import jax.numpy as jnp
import numpyro
import pandas as pd
from numpyro.contrib.control_flow import scan
from numpyro.handlers import mask
from numpyro.infer import HMCECS, MCMC, NUTS, SVI, Predictive, TraceGraph_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal

from ..distributions import StochasticFunctionDistribution
from ..utils import extract_time_vars
from .wgen_gamlss import WGEN_GAMLSS


class WGEN(ABC):

    def __init__(
        self,
        data: pd.DataFrame,
        model=WGEN_GAMLSS(),
        predictors=[],
        order=1,
        **kwargs,
    ):
        """Initializes the WGEN model with the given observation dataset and model

        Args:
            data (pd.DataFrame): _description_
            model (_type_, optional): _description_. Defaults to WGEN_GAMLSS().
            predictors (list, optional): _description_. Defaults to [].
            order (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        self.model = model
        self.model_kwargs = kwargs
        self.order = order
        self.obs = self.model.get_obs(data)
        self.timestamps = extract_time_vars(data.index)
        self.initial_states = self.model.get_initial_states(self.obs, order)
        self.valid_idx = self._valid_indices(self.initial_states, self.obs, order)
        self.first_valid_idx = int(self.valid_idx[0])
        self.predictors = jnp.concat(
            [
                jnp.ones((1, len(data.index), 1)),
                jnp.expand_dims(data[predictors].values, axis=0),
            ],
            axis=-1,
        )
        self.guide = None

    def _valid_indices(self, initial_states, obs, order):
        # filter out time indices where initial_states is NaN or Inf at time t and t+1;
        # here indices 0 and 1 correspond to the observed variable and batch dimensions respectively
        valid_mask_obs = jnp.stack([jnp.isfinite(v) for k, v in obs.items()]).any(axis=[0, 1])
        # here indices 0, 2, and 3 correspond to the batch, var, and lag dimensions
        valid_mask_initial_states = jnp.isfinite(initial_states).all(axis=[0, 2, 3])
        valid_mask = jnp.logical_and(valid_mask_obs[order:], valid_mask_initial_states)
        valid_idx = jnp.where(jnp.logical_and(valid_mask[:-1], valid_mask[1:]))[0]
        if len(valid_idx) < initial_states.shape[1]:
            logging.warning(f"dropped {initial_states.shape[1] - len(valid_idx)} nan/inf timesteps")
        return valid_idx

    def get_obs(self, data: pd.DataFrame):
        return self.model.get_obs(data)

    def get_initial_states(self, data: pd.DataFrame, order):
        initial_states = self.model.get_initial_states(data, order=order)
        return initial_states

    def prior(self, predictors, initial_states, **extra_kwargs):
        return self.model.prior(predictors, initial_states, **self.model_kwargs, **extra_kwargs)

    def step(
        self,
        timestamps=None,
        predictors=None,
        initial_states=None,
        batch_idx=0,
        prior_mask=True,
        subsample_time=None,
        **kwargs,
    ):
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
        timestamps = timestamps if timestamps is not None else self.timestamps[batch_idx, :, :]
        predictors = predictors if predictors is not None else self.predictors[batch_idx, :, :]
        initial_states = initial_states if initial_states is not None else self.initial_states[batch_idx, :, :]
        order = self.order
        assert len(timestamps.shape) == 2 and timestamps.shape[1] == 4, "timestamps must have shape (timesteps, 4)"
        assert len(predictors.shape) == 2, "predictors must have shape (timesteps, variables)"
        assert (
            predictors.shape[0] == initial_states.shape[0] + order
        ), "leading time dimension for predictors and initial states must match"

        with mask(mask=prior_mask):
            step = self.prior(predictors, initial_states, **kwargs)

        # plate over time dimension starting from the second index
        with numpyro.plate("time", len(self.valid_idx), subsample_size=subsample_time) as i:
            idx = self.valid_idx[i]
            inputs_i = jnp.concat([timestamps[idx + order, :], predictors[idx + order, :]], axis=-1)
            initial_state = initial_states[idx, :, :]
            obs_i = dict([(k, v[0, idx + order]) for k, v in obs.items()])
            # evaluate step function with time as the batch dimension
            pred_states, outputs = step(initial_state, inputs_i, obs_i)

        return pred_states, outputs

    def simulate(
        self,
        timestamps=None,
        predictors=None,
        initial_state=None,
        observable=None,
        batch_size=None,
        prior_mask=True,
        **kwargs,
    ):
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
        timestamps = timestamps if timestamps is not None else self.timestamps[:, self.first_valid_idx :, :]
        assert (
            len(timestamps.shape) == 3 and timestamps.shape[2] == 4
        ), "timestamps must have shape (batch_size, timesteps, 4)"
        predictors = predictors if predictors is not None else self.predictors[:, self.first_valid_idx :, :]
        if batch_size is None:
            batch_size = predictors.shape[0]
        else:
            predictors = predictors * jnp.ones((batch_size, 1, 1))
            timestamps = timestamps * jnp.ones((batch_size, 1, 1))
        initial_state = (
            initial_state if initial_state is not None else self.initial_states[:, self.first_valid_idx, :, :]
        )
        initial_state = initial_state * jnp.ones((batch_size, *initial_state.shape[1:]))
        # sample prior
        with mask(mask=prior_mask):
            step = self.prior(predictors, initial_state, **kwargs)

        inputs = jnp.concat([timestamps, predictors], axis=-1)
        # scan over inputs with step/transition function;
        # note that we need to swap the batch and time dimension since scan runs over the leading axis.
        _, outputs = scan(step, initial_state, jnp.swapaxes(inputs, 0, 1))
        if observable is not None:
            obsv = observable(timestamps, *outputs)
            return outputs, obsv
        else:
            return outputs, None

    def simulator(
        self,
        timestamps=None,
        predictors=None,
        initial_state=None,
        observable=None,
        rng_seed=0,
        **prior_kwargs,
    ):
        """Constructs a wrapper function `f(theta)` that invokes `simulate` with the given `observable`.
           The parameters `theta` are assumed to be in the unconstrained (transformed) sample space of the model.
           Returns the simulator function as well as the corresponding prior distribution.

        Args:
            observable (_type_, optional): observable function for `simulate`. Defaults to None.
            rng_seed (int, optional): random seed. Defaults to 0.

        Returns:
            tuple: simulator_fn, prior
        """
        timestamps = self.timestamps if timestamps is None else timestamps
        predictors = self.predictors if predictors is None else predictors
        initial_state = (
            initial_state if initial_state is not None else self.initial_states[:, self.first_valid_idx, :, :]
        )
        prior = StochasticFunctionDistribution(
            self.prior,
            fn_args=(predictors, initial_state),
            fn_kwargs=prior_kwargs,
            unconstrained=True,
            rng_seed=rng_seed,
        )

        def simulator(_theta):
            theta = _theta if len(_theta.shape) > 1 else _theta.reshape((1, -1))
            batch_size = theta.shape[0]
            t = timestamps * jnp.ones((batch_size, *timestamps.shape[1:]))
            X = predictors * jnp.ones((batch_size, *predictors.shape[1:]))
            params = {
                k: v for k, v in prior.constrain(theta, as_dict=True).items()
            }  # .squeeze(1) when using older models
            with numpyro.handlers.seed(rng_seed=rng_seed):
                with numpyro.handlers.condition(data=params):
                    if observable is not None:
                        outputs, obs = self.simulate(t, X, observable=observable, batch_size=batch_size)
                        return obs.T
                    else:
                        outputs, _ = self.simulate(t, X)
                        return jnp.permute_dims(jnp.stack(outputs), (2, 1, 0))

        return simulator, prior

    def fit(self, *args, method="svi", **kwargs):
        if method == "svi":
            return self._fit_svi(*args, **kwargs)
        elif method == "hmcecs":
            return self._fit_hmcecs(*args, **kwargs)
        elif method == "mcmc":
            return self._fit_mcmc(*args, **kwargs)
        else:
            raise (Exception(f"fit method {method} not recognized"))

    def save(self, fit_result, filepath):
        if fit_result is numpyro.infer.svi.SVIRunResult:
            with h5.File(filepath, "w") as f:
                for k, v in fit_result.params.items():
                    f.create_dataset(k, data=v)
        else:
            raise (Exception("unrecognized fit_result type"))

    def load(self, filepath):
        params = dict()
        with h5.File(filepath, "r") as f:
            for k, v in f.items():
                params[k] = jnp.array(v)
        return params

    def sample(self, fit_result, **kwargs):
        if fit_result is numpyro.infer.svi.SVIRunResult:
            return self._sample_svi(fit_result.params, **kwargs)
        else:
            raise (Exception("unrecognized fit_result type"))

    def _sample_svi(
        self, params, guide=None, timestamps=None, predictors=None, observable=None, num_samples=100, rng_seed=0
    ):
        prng = jax.random.PRNGKey(rng_seed)
        guide = self.guide if guide is None else guide
        posterior_sampler = Predictive(guide, params=params, num_samples=num_samples)
        posterior_params = posterior_sampler(prng)
        sim, prior = self.simulator(timestamps, predictors, observable, rng_seed)
        theta = prior.dict2array(posterior_params)
        outputs = sim(theta)
        return outputs

    def _fit_svi(
        self,
        iterations=None,
        guide=None,
        optimizer=numpyro.optim.Adam(1e-3),
        rng=jax.random.PRNGKey(1234),
        loss=TraceGraph_ELBO(),
        **kwargs,
    ):
        # if guide is not specified, default to multivariate normal
        if guide is None:
            guide = AutoMultivariateNormal(self.step, init_loc_fn=numpyro.infer.init_to_median, init_scale=0.1)
        self.guide = guide
        svi = SVI(self.step, guide, optimizer, loss=loss)
        if iterations is None:
            return svi.init(rng, **kwargs)
        else:
            return svi.run(rng, iterations, **kwargs)

    def _fit_hmcecs(
        self,
        num_samples=1000,
        num_warmup=1000,
        num_blocks=10,
        num_chains=4,
        chain_method="parallel",
        init_strategy=numpyro.infer.init_to_median,
        svi_iterations=10000,
        rng=jax.random.PRNGKey(1234),
        **kwargs,
    ):
        # use autodelta to get MAP estimate
        map_guide = AutoDelta(self.step, init_loc_fn=numpyro.infer.init_to_median)
        map_result = self._fit_svi(svi_iterations, map_guide, rng=rng, **kwargs)
        with numpyro.handlers.seed(rng_seed=0):
            # get parameter dict; for some reason setting the seed is necessary...
            map_params = map_guide(map_result.params)
        # build taylor proxy around mdoe for HMCECS
        proxy = HMCECS.taylor_proxy(map_params)
        # set up and run HMCECS
        mcmc_kernel = HMCECS(
            NUTS(self.step, init_strategy=init_strategy),
            num_blocks=num_blocks,
            proxy=proxy,
        )
        mcmc = MCMC(
            mcmc_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
        )
        mcmc.run(rng, **kwargs)
        return mcmc

    def _fit_mcmc(
        self,
        num_samples=1000,
        num_warmup=1000,
        num_chains=4,
        chain_method="parallel",
        init_strategy=numpyro.infer.init_to_median,
        mcmc_kernel=NUTS,
        rng=jax.random.PRNGKey(1234),
        kernel_kwargs=dict(),
        **fn_kwargs,
    ):
        mcmc_kernel = mcmc_kernel(self.step, init_strategy=init_strategy, **kernel_kwargs)
        mcmc = MCMC(
            mcmc_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
        )
        mcmc.run(rng, **fn_kwargs)
        return mcmc
