import pandas as pd

from abc import ABC

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import mask

from .. import utils, data
from ..distributions import BernoulliGamma, from_moments


class WGEN_GLM_v5(ABC):

    def get_initial_states(self, data: pd.DataFrame | tuple[int, int], order):
        """Returns initial states"""

        if isinstance(data, tuple):
            # default to batch_size = 1 and one time step
            return jnp.zeros((*data, order, 2))

        # initialize from observations
        obs = data
        prec_state = jnp.expand_dims(obs["prec"], axis=[-1, -2])
        Tavg_state = jnp.expand_dims(obs["Tavg"], axis=[-1, -2])
        Trange_state = jnp.expand_dims(obs["Trange"], axis=[-1, -2])

        state = jnp.concat([prec_state, Tavg_state, Trange_state], axis=-2)

        # concatenate lagged states
        timelen = state.shape[1]
        return jnp.concat([state[:, i : timelen - order + i, :, :] for i in range(order)], axis=-2)

    def get_obs(self, data: pd.DataFrame):
        """Returns the observations"""

        prec_obs = data["prec"]
        Tavg_obs = data["Tair_mean"]
        Trange_obs = data["Tair_max"] - data["Tair_min"]
        Tskew_obs = (Tavg_obs - data["Tair_min"]) / Trange_obs
        return {
            "prec": jnp.array(prec_obs.values).reshape((1, -1)),
            "Tavg": jnp.array(Tavg_obs.values).reshape((1, -1)),
            "Trange": jnp.array(Trange_obs.values).reshape((1, -1)),
            "Tskew": jnp.array(Tskew_obs.values).reshape((1, -1)),
        }

    def prior(
        self,
        predictors,
        initial_states,
        pred_effect_scale=jnp.ones(1),
        Tskew_scaled_dispersion_mean=1.0,
        Tair_freqs=[1 / 365.25],
        prec_freqs=[1 / 365.25],
        **kwargs,
    ):
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
            Tavg_dof_mean (float, optional): if not None, specifies the prior mean of the DoF parameter for a Student-t likelihood. Defaults to None, i.e. Gaussian likelihood.
            pred_effect_scale (float, optional): standard deviation of the predictor effect prior. Defaults to 1.0.
            Tskew_scaled_dispersion_mean (float, optional): prior mean of the Tskew dispersion parameter. Defaults to 1.0.
            Tair_freqs (list, optional): frequencies for air temperature seasonal effects. Defaults to the annual cycle: [1/365.25].
            prec_freqs (list, optional): frequencies for precipitation seasonal effects. Defaults to the annual cycle: [1/365.25].

        Returns:
            _type_: _description_
        """
        num_predictors = predictors.shape[-1]
        order = initial_states.shape[-1]
        assert num_predictors > 0, "number of predictors must be greater than zero"
    
        # mean air temperature
        tair_mean_step = wgen_glm_v5_Tair_mean(
            num_predictors,
            pred_effect_scale,
            freqs=Tair_freqs,
            order=order,
            **kwargs,
        )
        # precipitation
        precip_step = wgen_glm_v5_precip(num_predictors, pred_effect_scale, freqs=prec_freqs, order=order, **kwargs)
        # air temperature range and skew
        tair_range_skew_step = wgen_glm_v5_Tair_range_skew(
            num_predictors,
            pred_effect_scale,
            Tskew_scaled_dispersion_mean,
            freqs=Tair_freqs,
            order=order,
            **kwargs,
        )

        def step(state, inputs, obs={"prec": None, "Tavg": None, "Trange": None, "Tskew": None}):
            assert state.shape[0] == inputs.shape[0], "state and input batch dimensions do not match"
            assert state.shape[1] == order, f"state lag dimension does not match order={order}"
            # unpack state and input tensors;
            # state is assumed to have shape (batch, lag, vars)
            prec_prev = state[:, 0, :]
            Tavg_prev = state[:, 1, :]
            Trange_prev = state[:, 2, :]
            # i, year, month, doy = inputs[:, :4].T
            # predictors = inputs[:, 4:]
            # mean daily air temperature
            Tavg = tair_mean_step(Tavg_prev, inputs, obs["Tavg"])  # Tavg_loc, Tavg_seasonal_anomaly
            Tavg_for_precip = Tavg.reshape((-1, 1))  # Tavg_seasonal_anomaly.reshape((-1, 1))
            # precipitation
            prec = precip_step((prec_prev, Tavg_for_precip), inputs, obs["prec"])
            # air temperature range and skew
            Trange, Tskew, Tmin, Tmax = tair_range_skew_step(
                (Tavg, prec, Trange_prev), inputs, obs["Trange"], obs["Tskew"]
            )
            newstate = jnp.expand_dims(jnp.stack([prec, Tavg, Trange]).T, axis=-1)
            return jnp.concat([state[:, :, 1:], newstate], axis=-1), (prec, Tmin, Tavg, Tmax)

        return step


def wgen_glm_v5_Tair_mean(
    num_predictors: int = 1,
    pred_effect_scale=jnp.ones(1),
    freqs=[1 / 365.25, 2 / 365.2],
    order=1,
    Tavg_lag_in_scale=False,
    **kwargs,
):
    seasonal_dims = 2 * len(freqs)

    # Location
    Tavg_loc_lag_effects = numpyro.sample(
        "Tavg_loc_lag", dist.MultivariateNormal(jnp.zeros(order), 0.2 * jnp.eye(order))
    )
    Tavg_loc_seasonal_effects = numpyro.sample(
        "Tavg_loc_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Tavg_loc_seasonal_lag_interaction_effects = numpyro.sample(
        "Tavg_loc_seasonal_lag_interaction",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims * order), jnp.eye(seasonal_dims * order)),
    )
    Tavg_loc_pred_effects = numpyro.sample(
        "Tavg_loc_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    Tavg_loc_effects = jnp.concat(
        [
            Tavg_loc_seasonal_effects,
            Tavg_loc_lag_effects,
            Tavg_loc_seasonal_lag_interaction_effects,
            Tavg_loc_pred_effects,
        ],
        axis=-1,
    )

    ## Scale
    Tavg_scale_seasonal_effects = numpyro.sample(
        "Tavg_loc_scale_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Tavg_scale_pred_effects = numpyro.sample(
        "Tavg_loc_scale_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    if Tavg_lag_in_scale:
        Tavg_scale_lag_effects = numpyro.sample(
            "Tavg_scale_lag", dist.MultivariateNormal(jnp.zeros(order), 0.2 * jnp.eye(order))
        )
        Tavg_scale_effects = jnp.concat(
            [Tavg_scale_seasonal_effects, Tavg_scale_lag_effects, Tavg_scale_pred_effects], axis=-1
        )
    else:
        Tavg_scale_effects = jnp.concat([Tavg_scale_seasonal_effects, Tavg_scale_pred_effects], axis=-1)

    def step(state, inputs, Tavg_obs=None):
        Tavg_prev = state
        i, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]

        ff_t = utils.fourier_feats(i, freqs, intercept=False) * jnp.ones((Tavg_prev.shape[0], 1))
        seasonal_lag_interactions = jnp.concat([ff_t * Tavg_prev[:, i : (i + 1)] for i in range(order)], axis=1)

        Tavg_loc_features = jnp.concat([ff_t, Tavg_prev, seasonal_lag_interactions, predictors], axis=1)

        if Tavg_lag_in_scale:
            Tavg_scale_features = jnp.concat([ff_t, jnp.log(jnp.square(Tavg_prev)), predictors], axis=1)
        else:
            Tavg_scale_features = jnp.concat([ff_t, predictors], axis=1)

        Tavg_loc = numpyro.deterministic(
            "Tavg_loc",
            jnp.sum(Tavg_loc_features * Tavg_loc_effects, axis=1),
        )
        Tavg_scale = numpyro.deterministic(
            "Tavg_scale",
            jnp.exp(jnp.sum(Tavg_scale_features * Tavg_scale_effects, axis=1)),
        )

        # Sample Tavg
        Tavg_mask = jnp.isfinite(Tavg_obs) if Tavg_obs is not None else True
        with mask(mask=Tavg_mask):
            Tavg = numpyro.sample("Tavg", dist.Normal(Tavg_loc, Tavg_scale), obs=Tavg_obs)

        Tavg_sample_seasonal_anomaly = numpyro.deterministic(
            "Tavg_sample_seasonal_anomaly", Tavg - jnp.sum(Tavg_loc_seasonal_effects * ff_t, axis=1)
        )
        return Tavg  # , Tavg_loc, Tavg_sample_seasonal_anomaly

    return step


def wgen_glm_v5_precip(
    num_predictors: int = 1,
    pred_effect_scale=jnp.ones(1),
    freqs=[1 / 365.25],
    order=1,
    **kwargs,
):
    seasonal_dims = 2 * len(freqs)

    # Occurrence
    precip_occ_seasonal_effects = numpyro.sample(
        "precip_occ_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    precip_occ_lag_effects = numpyro.sample(
        "precip_occ_lag",
        dist.MultivariateNormal(jnp.zeros(2 * order), 0.2 * jnp.eye(2 * order)),
    )
    precip_occ_lag_seasonal_interaction_effects = numpyro.sample(
        "precip_occ_lag_seasonal_interaction",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims * order * 2), 0.2 * jnp.eye(seasonal_dims * order * 2)),
    )
    precip_occ_Tavg_effects = numpyro.sample("precip_occ_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1 * jnp.eye(1)))
    precip_occ_Tavg_seasonal_interaction_effects = numpyro.sample(
        "precip_occ_Tavg_seasonal_interaction", dist.MultivariateNormal(jnp.zeros(2), 0.1 * jnp.eye(2))
    )
    precip_occ_pred_effects = numpyro.sample(
        "precip_occ_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    precip_occ_all_effects = jnp.concat(
        [
            precip_occ_seasonal_effects,
            precip_occ_lag_effects,
            precip_occ_lag_seasonal_interaction_effects,
            precip_occ_Tavg_effects,
            precip_occ_Tavg_seasonal_interaction_effects,
            precip_occ_pred_effects,
        ],
        axis=-1,
    )

    # Amounts: location
    precip_loc_seasonal_effects = numpyro.sample(
        "precip_loc_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    precip_loc_Tavg_effects = numpyro.sample("precip_loc_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1 * jnp.eye(1)))
    precip_loc_Tavg_seasonal_interaction_effects = numpyro.sample(
        "precip_loc_Tavg_seasonal_interaction", dist.MultivariateNormal(jnp.zeros(2), 0.1 * jnp.eye(2))
    )
    precip_loc_lag_effects = numpyro.sample(
        "precip_loc_lag",
        dist.MultivariateNormal(jnp.zeros(2 * order), 0.2 * jnp.eye(2 * order)),
    )
    precip_loc_lag_seasonal_interaction_effects = numpyro.sample(
        "precip_loc_lag_seasonal_interaction",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims * order * 2), 0.2 * jnp.eye(seasonal_dims * order * 2)),
    )
    precip_loc_pred_effects = numpyro.sample(
        "precip_loc_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    precip_loc_all_effects = jnp.concat(
        [
            precip_loc_seasonal_effects,
            precip_loc_lag_effects,
            precip_loc_lag_seasonal_interaction_effects,
            precip_loc_Tavg_effects,
            precip_loc_Tavg_seasonal_interaction_effects,
            precip_loc_pred_effects,
        ],
        axis=-1,
    )
    # Amounts shape
    precip_shape_seasonal_effects = numpyro.sample(
        "precip_shape_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    precip_shape_pred_effects = numpyro.sample(
        "precip_shape_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    precip_shape_all_effects = jnp.concat(
        [
            precip_shape_seasonal_effects,
            precip_shape_pred_effects,
        ],
        axis=-1,
    )

    def step(state, inputs, prec_obs=None):
        prec_prev, Tavg = state
        i, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]

        Tavg = jnp.sign(Tavg) * jnp.log(jnp.square(Tavg)) / 2

        prev_dry = 1 - jnp.sign(prec_prev)
        log_prec_prev = jnp.log(1 + prec_prev)

        ff_t = utils.fourier_feats(i, freqs, intercept=False) * jnp.ones((prec_prev.shape[0], 1))

        # Seasonal interactions
        seasonal_lag_interactions_amounts = jnp.concat(
            [ff_t * log_prec_prev[:, i : (i + 1)] for i in range(order)], axis=1
        )
        seasonal_lag_interactions_occ = jnp.concat([ff_t * prev_dry[:, i : (i + 1)] for i in range(order)], axis=1)
        seasonal_Tavg_interactions = (
            utils.fourier_feats(i, freqs=[1 / 365.25], intercept=False) * jnp.ones((prec_prev.shape[0], 1)) * Tavg
        )

        prec_occ_features = jnp.concat(
            [
                ff_t,
                prev_dry,
                log_prec_prev,
                seasonal_lag_interactions_occ,
                seasonal_lag_interactions_amounts,
                Tavg,
                seasonal_Tavg_interactions,
                predictors,
            ],
            axis=1,
        )
        prec_mean_features = jnp.concat(
            [
                ff_t,
                prev_dry,
                log_prec_prev,
                seasonal_lag_interactions_occ,
                seasonal_lag_interactions_amounts,
                Tavg,
                seasonal_Tavg_interactions,
                predictors,
            ],
            axis=1,
        )
        prec_shape_features = jnp.concat([ff_t, predictors], axis=1)

        # Parameters
        p_wet = numpyro.deterministic(
            "p_wet", jax.nn.sigmoid(jnp.sum(prec_occ_features * precip_occ_all_effects, axis=1))
        )
        precip_gamma_loc = jnp.exp(jnp.sum(prec_mean_features * precip_loc_all_effects, axis=1))
        precip_gamma_shape = numpyro.deterministic(
            "precip_gamma_shape", jnp.exp(jnp.sum(prec_shape_features * precip_shape_all_effects, axis=1))
        )
        precip_gamma_rate = numpyro.deterministic("precip_gamma_rate", precip_gamma_shape / precip_gamma_loc)

        # Sample precipication from bernoulli-gamma with prob p_wet
        prec_occ_obs = prec_obs > 0.0 if prec_obs is not None else None
        prec_mask = prec_occ_obs if prec_occ_obs is not None else True
        with mask(mask=jnp.isfinite(prec_obs) if prec_obs is not None else True):
            prec_occ = numpyro.sample("prec_occ", dist.Bernoulli(p_wet), obs=prec_occ_obs)
        with mask(mask=prec_mask):
            prec_amount = numpyro.sample(
                "prec_amount",
                dist.Gamma(precip_gamma_shape, precip_gamma_rate),
                obs=prec_obs,
            )
        prec = numpyro.deterministic("prec", jnp.where(prec_occ, prec_amount, 0.0))
        return prec

    return step


def wgen_glm_v5_Tair_range_skew(
    num_predictors: int = 1,
    pred_effect_scale=1.0,
    Tskew_scaled_dispersion_mean=1.0,
    freqs=[1 / 365.25],
    order=1,
    Trange_max=22.0,
    **kwargs,
):
    ## Trange alpha
    freqs = [
        1 / 365.25,
        2 / 365.25,
    ]  # 3/365.25]
    seasonal_dims = 2 * len(freqs)
    Trange_alpha_seasonal_effects = numpyro.sample(
        "Trange_alpha_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Trange_alpha_Trange_prev_effects = numpyro.sample(
        "Trange_alpha_Trange_prev", dist.MultivariateNormal(jnp.zeros(order), 0.1 * jnp.eye(order))
    )
    Trange_alpha_pred_effects = numpyro.sample(
        "Trange_alpha_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    Trange_alpha_seasonal_lag_interaction_effects = numpyro.sample(
        "Trange_alpha_seasonal_lag_interaction",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims * order), 0.2 * jnp.eye(seasonal_dims * order)),
    )
    Trange_alpha_all_effects = jnp.concat(
        [
            Trange_alpha_seasonal_effects,
            Trange_alpha_Trange_prev_effects,
            Trange_alpha_seasonal_lag_interaction_effects,
            Trange_alpha_pred_effects,
        ],
        axis=-1,
    )
    ## Trange beta
    Trange_beta_seasonal_effects = numpyro.sample(
        "Trange_beta_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Trange_beta_Trange_prev = numpyro.sample(
        "Trange_beta_Trange_prev", dist.MultivariateNormal(jnp.zeros(order), 0.2 * jnp.eye(order))
    )
    Trange_beta_pred_effects = numpyro.sample(
        "Trange_beta_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    Trange_beta_seasonal_lag_interaction_effects = numpyro.sample(
        "Trange_beta_seasonal_lag_interaction",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims * order), 0.2 * jnp.eye(seasonal_dims * order)),
    )
    Trange_beta_all_effects = jnp.concat(
        [
            Trange_beta_seasonal_effects,
            Trange_beta_Trange_prev,
            Trange_beta_seasonal_lag_interaction_effects,
            Trange_beta_pred_effects,
        ],
        axis=-1,
    )

    ## Tskew alpha
    Tskew_alpha_seasonal_effects = numpyro.sample(
        "Tskew_alpha_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Tskew_alpha_Tavg_effects = numpyro.sample(
        "Tskew_alpha_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1 * jnp.eye(1))
    )
    Tskew_alpha_pred_effects = numpyro.sample(
        "Tskew_alpha_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    Tskew_alpha_all_effects = jnp.concat(
        [
            Tskew_alpha_seasonal_effects,
            Tskew_alpha_Tavg_effects,
            Tskew_alpha_pred_effects,
        ],
        axis=-1,
    )
    ## Tskew beta
    Tskew_beta_seasonal_effects = numpyro.sample(
        "Tskew_beta_seasonal",
        dist.MultivariateNormal(jnp.zeros(seasonal_dims), jnp.eye(seasonal_dims)),
    )
    Tskew_beta_Tavg_effects = numpyro.sample("Tskew_beta_Tavg", dist.MultivariateNormal(jnp.zeros(1), 0.1 * jnp.eye(1)))
    Tskew_beta_pred_effects = numpyro.sample(
        "Tskew_beta_pred",
        dist.MultivariateNormal(jnp.zeros(num_predictors), jnp.diag(pred_effect_scale)),
    )
    Tskew_beta_all_effects = jnp.concat(
        [
            Tskew_beta_seasonal_effects,
            Tskew_beta_Tavg_effects,
            Tskew_beta_pred_effects,
        ],
        axis=-1,
    )

    def step(state, inputs, Trange_obs=None, Tskew_obs=None):

        Trange_obs_scaled = Trange_obs
        if Trange_obs is not None:
            # Rescale to [0, 1]
            Trange_obs_scaled = Trange_obs / Trange_max

        Tavg, prec, Trange_prev = state
        Trange_prev_scaled = Trange_prev / Trange_max
        i, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]
        ff_t = utils.fourier_feats(i, freqs, intercept=False) * jnp.ones((Tavg.shape[0], 1))

        # Features
        Trange_features = jnp.concat(
            [
                ff_t,
                jnp.log(Trange_prev_scaled),
                jnp.concat([ff_t * Trange_prev_scaled[:, i : (i + 1)] for i in range(order)], axis=1),
                predictors,
            ],
            axis=1,
        )
        Tskew_features = jnp.concat([ff_t, Tavg.reshape((-1, 1)), predictors], axis=1)

        # Parameters Trange and Tskew
        Trange_alpha = jnp.exp(
            jnp.sum(Trange_features * Trange_alpha_all_effects, axis=1)
        )  # jax.scipy.special.expit *100
        Trange_beta = jnp.exp(jnp.sum(Trange_features * Trange_beta_all_effects, axis=1))  # *100

        Tskew_alpha = jnp.exp(jnp.sum(Tskew_features * Tskew_alpha_all_effects, axis=1))
        Tskew_beta = jnp.exp(jnp.sum(Tskew_features * Tskew_beta_all_effects, axis=1))

        # Sample
        Trange_mask = jnp.isfinite(Trange_obs) if Trange_obs is not None else True
        with mask(mask=Trange_mask):
            Trange_scaled = numpyro.sample(
                "Trange_scaled",
                dist.Beta(Trange_alpha, Trange_beta),
                obs=Trange_obs_scaled,
            )
            Trange = numpyro.deterministic("Trange", Trange_scaled * Trange_max)

        Tskew_mask = jnp.isfinite(Tskew_obs) if Tskew_obs is not None else True
        with mask(mask=Tskew_mask):
            Tskew = numpyro.sample(
                "Tskew",
                dist.Beta(Tskew_alpha, Tskew_beta),
                obs=Tskew_obs,
            )

        # calculate min and max from range and skew
        Tmin = numpyro.deterministic("Tmin", Tavg - Tskew * Trange)
        Tmax = numpyro.deterministic("Tmax", Tmin + Trange)
        return Trange, Tskew, Tmin, Tmax

    return step
