from abc import ABC

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.handlers import mask

from .. import glm, utils


class WGEN_GAMLSS(ABC):

    def get_initial_states(self, obs_or_shape: dict | tuple[int, int], order=1, dropna=True):
        if isinstance(obs_or_shape, tuple):
            # default to batch_size = 1 and one time step
            return jnp.zeros((*obs_or_shape, order, 2))
        # initialize from observations
        obs = obs_or_shape
        prec_state = jnp.expand_dims(obs["prec"], axis=[-1, -2])
        Tavg_state = jnp.expand_dims(obs["Tavg"], axis=[-1, -2])
        Trange_state = jnp.expand_dims(obs["Trange"], axis=[-1, -2])
        state = jnp.concat([prec_state, Tavg_state, Trange_state], axis=-2)
        # concatenate lagged states
        timelen = state.shape[1]
        return jnp.concat([state[:, i : timelen - order + i, :, :] for i in range(order)], axis=-1)

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
            "Tmax": jnp.array(data["Tair_max"].values).reshape((1, -1)),
            "Tmin": jnp.array(data["Tair_min"].values).reshape((1, -1)),
        }

    def prior(
        self,
        predictors,
        initial_states,
        pred_effect_scale=jnp.ones(1),
        Tair_freqs=[1 / 365.25],
        prec_freqs=[1 / 365.25],
        **kwargs,
    ):
        """Improved WGEN-glm.GLM which generates daily weather variables according to the following procedure:
            1) Tavg(t) | Tavg(t-1), ...,Tavg(t-n)
            2) prec | prec(t-1), ..., prec(t-n), Tavg(t-1), ...,Tavg(t-n)
            3) Trange | prec, Tavg
            4) Tskew | prec, Tavg

            Mean daily air temperature is modelled as Student-T distribution with variable degrees of freedom.
            Precipitation is modelled as a bernoulli-Gamma mixture distribution.
            Daily air temperature range is modelled as a Gamma distribution.
            Daily air temperature skewness is modeled as a Beta distribution.

            Each of these observable variables are parameterized as glm.GLMs defined over some set of linear predictors.

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
        tair_mean_step = Tavg_model(
            num_predictors,
            pred_effect_scale,
            freqs=Tair_freqs,
            order=order,
            **kwargs,
        )
        # precipitation
        precip_step = precip_model(num_predictors, pred_effect_scale, freqs=prec_freqs, order=order, **kwargs)
        # air temperature range and skew
        tair_range_skew_step = Trange_skew_model(
            num_predictors,
            pred_effect_scale,
            freqs=Tair_freqs,
            order=order,
            **kwargs,
        )

        def step(state, inputs, obs={"prec": None, "Tavg": None, "Trange": None, "Tskew": None}):
            assert state.shape[0] == inputs.shape[0], "state and input batch dimensions do not match"
            assert state.shape[2] == order, f"state lag dimension does not match order={order}"
            # unpack state and input tensors;
            # state is assumed to have shape (batch, vars, lag)
            prec_prev = state[:, 0, :]
            Tavg_prev = state[:, 1, :]
            Trange_prev = state[:, 2, :]
            # mean daily air temperature
            Tavg = tair_mean_step(Tavg_prev, inputs, obs["Tavg"])  # Tavg_loc, Tavg_seasonal_anomaly
            Tavg_var = Tavg.reshape((-1, 1))
            # precipitation
            prec = precip_step((prec_prev, Tavg_var), inputs, obs["prec"])
            # air temperature range and skew
            Trange, Tskew, Tmin, Tmax = tair_range_skew_step(
                (Tavg_var, prec, Trange_prev), inputs, obs["Trange"], obs["Tskew"]
            )
            newstate = jnp.expand_dims(jnp.stack([prec, Tavg, Trange]).T, axis=-1)
            return jnp.concat([state[:, :, 1:], newstate], axis=-1), (prec, Tmin, Tavg, Tmax)

        return step


def Tavg_model(
    num_predictors: int = 1,
    pred_effect_scale=jnp.ones(1),
    freqs=[1 / 365.25, 2 / 365.2],
    order=1,
    Tavg_lag_in_scale=False,
    **kwargs,
):
    # Location
    Tavg_loc_lag_effects = glm.LinearEffects("Tavg_loc_lag", order, scale_or_cov=0.2)
    Tavg_loc_seasonal_effects = glm.SeasonalEffects("Tavg_loc_seasonal", freqs)
    Tavg_loc_seasonal_lag_interaction = glm.InteractionEffects(
        "Tavg_loc_seasonal_lag_interaction",
        Tavg_loc_lag_effects,
        Tavg_loc_seasonal_effects,
    )
    Tavg_loc_pred_effects = glm.LinearEffects("Tavg_loc_pred", num_predictors, scale_or_cov=jnp.diag(pred_effect_scale))
    Tavg_loc_glm = glm.GLM(
        Tavg_loc_lag_effects,
        Tavg_loc_seasonal_effects,
        Tavg_loc_seasonal_lag_interaction,
        Tavg_loc_pred_effects,
    )

    ## Scale
    Tavg_scale_seasonal_effects = glm.SeasonalEffects("Tavg_loc_scale_seasonal", freqs)
    Tavg_scale_pred_effects = glm.LinearEffects(
        "Tavg_loc_scale_pred", num_predictors, scale_or_cov=jnp.diag(pred_effect_scale)
    )
    if Tavg_lag_in_scale:
        Tavg_scale_lag_effects = glm.LinearEffects("Tavg_scale_lag", order, scale_or_cov=0.2)
        Tavg_scale_glm = glm.GLM(
            Tavg_scale_seasonal_effects,
            Tavg_scale_pred_effects,
            Tavg_scale_lag_effects,
            link=glm.LogLink(),
        )
    else:
        Tavg_scale_glm = glm.GLM(
            Tavg_scale_seasonal_effects,
            Tavg_scale_pred_effects,
            link=glm.LogLink(),
        )

    def step(state, inputs, Tavg_obs=None):
        Tavg_prev = state
        t, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]

        Tavg_loc, _ = Tavg_loc_glm(Tavg_prev, t, (Tavg_prev, t), predictors)
        Tavg_loc = numpyro.deterministic(
            "Tavg_loc",
            Tavg_loc,
        )

        if Tavg_lag_in_scale:
            Tavg_scale = numpyro.deterministic(
                "Tavg_scale",
                Tavg_scale_glm(t, predictors, Tavg_prev)[0],
            )
        else:
            Tavg_scale = numpyro.deterministic(
                "Tavg_scale",
                Tavg_scale_glm(t, predictors)[0],
            )

        # Sample Tavg
        Tavg_mask = jnp.isfinite(Tavg_obs) if Tavg_obs is not None else True
        with mask(mask=Tavg_mask):
            Tavg = numpyro.sample("Tavg", dist.Normal(Tavg_loc, Tavg_scale), obs=Tavg_obs)

        Tavg_sample_seasonal_anomaly = numpyro.deterministic(
            "Tavg_sample_seasonal_anomaly",
            Tavg - jnp.sum(Tavg_loc_seasonal_effects.get_coefs() * Tavg_loc_seasonal_effects.get_predictors(t), axis=1),
        )
        return Tavg  # , Tavg_loc, Tavg_sample_seasonal_anomaly

    return step


def precip_model(
    num_predictors: int = 1,
    pred_effect_scale=jnp.ones(1),
    freqs=[1 / 365.25],
    order=1,
    **kwargs,
):
    # Occurrence
    precip_occ_seasonal_effects = glm.SeasonalEffects("precip_occ_seasonal", freqs)
    precip_occ_lag_effects = glm.LinearEffects("precip_occ_lag", 2 * order, scale_or_cov=0.2)
    precip_occ_lag_seasonal_interaction_effects = glm.InteractionEffects(
        "precip_occ_lag_seasonal_interaction",
        precip_occ_lag_effects,
        precip_occ_seasonal_effects,
        scale_or_cov=0.5,
    )
    precip_occ_Tavg_effects = glm.LinearEffects("precip_occ_Tavg", 1)
    precip_occ_Tavg_seasonal_interaction_effects = glm.InteractionEffects(
        "precip_occ_Tavg_seasonal_interaction",
        precip_occ_Tavg_effects,
        precip_occ_seasonal_effects,
        scale_or_cov=0.5,
    )
    precip_occ_pred_effects = glm.LinearEffects("precip_occ_pred", num_predictors, scale_or_cov=pred_effect_scale)
    precip_occ_glm = glm.GLM(
        precip_occ_seasonal_effects,
        precip_occ_lag_effects,
        precip_occ_lag_seasonal_interaction_effects,
        precip_occ_Tavg_effects,
        precip_occ_Tavg_seasonal_interaction_effects,
        precip_occ_pred_effects,
        link=glm.LogitLink(),
    )

    # Amounts: location
    precip_loc_seasonal_effects = glm.SeasonalEffects("precip_loc_seasonal", freqs)
    precip_loc_Tavg_effects = glm.LinearEffects("precip_loc_Tavg", 1, scale_or_cov=0.1)
    precip_loc_Tavg_seasonal_interaction_effects = glm.InteractionEffects(
        "precip_loc_Tavg_seasonal_interaction",
        precip_loc_Tavg_effects,
        precip_loc_seasonal_effects,
    )
    precip_loc_lag_effects = glm.LinearEffects("precip_loc_lag", order, scale_or_cov=0.2)
    precip_loc_lag_seasonal_interaction_effects = glm.InteractionEffects(
        "precip_loc_lag_seasonal_interaction",
        precip_loc_lag_effects,
        precip_loc_seasonal_effects,
        scale_or_cov=0.2,
    )
    precip_loc_pred_effects = glm.LinearEffects("precip_loc_pred", num_predictors, scale_or_cov=pred_effect_scale)
    precip_loc_glm = glm.GLM(
        precip_loc_seasonal_effects,
        precip_loc_Tavg_effects,
        precip_loc_Tavg_seasonal_interaction_effects,
        precip_loc_lag_effects,
        precip_loc_lag_seasonal_interaction_effects,
        precip_loc_pred_effects,
        link=glm.LogLink(),
    )

    # Amounts shape
    precip_shape_seasonal_effects = glm.SeasonalEffects("precip_shape_seasonal", freqs)
    precip_shape_pred_effects = glm.LinearEffects("precip_shape_pred", num_predictors)
    precip_shape_glm = glm.GLM(
        precip_shape_seasonal_effects,
        precip_shape_pred_effects,
        link=glm.LogLink(),
    )

    def step(state, inputs, prec_obs=None):
        prec_prev, Tavg = state
        t, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]

        Tavg = jnp.sign(Tavg) * jnp.log(1 + jnp.square(Tavg)) / 2

        prev_dry = 1 - jnp.sign(prec_prev)
        prev_log_prec = jnp.log(1 + prec_prev)
        lag_preds = jnp.concat([prev_dry, prev_log_prec], axis=1)

        # Seasonal interactions
        # seasonal_lag_interactions_amounts = jnp.concat(
        #     [ff_t * log_prec_prev[:, i : (i + 1)] for i in range(order)], axis=1
        # )
        # seasonal_lag_interactions_occ = jnp.concat([ff_t * prev_dry[:, i : (i + 1)] for i in range(order)], axis=1)
        # seasonal_Tavg_interactions = (
        #     utils.fourier_feats(i, freqs=[1 / 365.25], intercept=False) * jnp.ones((prec_prev.shape[0], 1)) * Tavg
        # )

        # Parameters
        p_wet, _ = precip_occ_glm(t, lag_preds, (lag_preds, t), Tavg, (Tavg, t), predictors)
        p_wet = numpyro.deterministic("p_wet", p_wet)
        precip_gamma_loc, _ = precip_loc_glm(t, Tavg, (Tavg, t), prev_log_prec, (prev_log_prec, t), predictors)
        precip_gamma_loc = numpyro.deterministic("precip_gamma_loc", precip_gamma_loc)
        precip_gamma_shape, _ = precip_shape_glm(t, predictors)
        precip_gamma_shape = numpyro.deterministic("precip_gamma_shape", precip_gamma_shape)
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


def Trange_skew_model(
    num_predictors: int = 1,
    pred_effect_scale=1.0,
    freqs=[1 / 365.25, 2 / 365.25],
    order=1,
    Trange_max_default=22.0,
    **kwargs,
):
    # Trange mean
    Trange_mean_seasonal_effects = glm.SeasonalEffects("Trange_mean_seasonal", freqs)
    Trange_mean_lag_effects = glm.LinearEffects("Trange_mean_lag", order, scale_or_cov=0.1)
    Trange_mean_pred_effects = glm.LinearEffects("Trange_mean_pred", num_predictors, scale_or_cov=pred_effect_scale)
    Trange_mean_seasonal_lag_interaction_effects = glm.InteractionEffects(
        "Trange_mean_lag_seasonal_interaction",
        Trange_mean_lag_effects,
        Trange_mean_seasonal_effects,
        scale_or_cov=0.2,
    )
    Trange_mean_Tavg_effects = glm.LinearEffects("Trange_mean_Tavg", 1, scale_or_cov=0.1)
    Trange_mean_seasonal_Tavg_interaction_effects = glm.InteractionEffects(
        "Trange_mean_Tavg_seasonal_interaction",
        Trange_mean_Tavg_effects,
        Trange_mean_seasonal_effects,
        scale_or_cov=0.2,
    )
    Trange_mean_glm = glm.GLM(
        Trange_mean_seasonal_effects,
        Trange_mean_lag_effects,
        Trange_mean_pred_effects,
        Trange_mean_seasonal_lag_interaction_effects,
        Trange_mean_Tavg_effects,
        Trange_mean_seasonal_Tavg_interaction_effects,
        link=glm.LogitLink(),
    )

    # Trange dispersion
    Trange_disp_seasonal_effects = glm.SeasonalEffects("Trange_disp_seasonal", freqs)
    Trange_disp_lag_effects = glm.LinearEffects("Trange_disp_lag", order, scale_or_cov=0.1)
    Trange_disp_pred_effects = glm.LinearEffects("Trange_disp_pred", num_predictors, scale_or_cov=pred_effect_scale)
    Trange_disp_seasonal_lag_interaction_effects = glm.InteractionEffects(
        "Trange_disp_lag_seasonal_interaction",
        Trange_disp_lag_effects,
        Trange_disp_seasonal_effects,
        scale_or_cov=0.2,
    )
    Trange_disp_glm = glm.GLM(
        Trange_disp_seasonal_effects,
        Trange_disp_lag_effects,
        Trange_disp_pred_effects,
        Trange_disp_seasonal_lag_interaction_effects,
        link=glm.LogLink(),
    )

    # Tskew mean
    Tskew_mean_seasonal_effects = glm.SeasonalEffects("Tskew_mean_seasonal", freqs)
    Tskew_mean_Tavg_effects = glm.LinearEffects("Tskew_mean_Tavg", 1, scale_or_cov=0.1)
    Tskew_mean_seasonal_Tavg_interaction_effects = glm.InteractionEffects(
        "Tskew_mean_Tavg_seasonal_interaction",
        Tskew_mean_Tavg_effects,
        Tskew_mean_seasonal_effects,
        scale_or_cov=0.2,
    )
    Tskew_mean_pred_effects = glm.LinearEffects("Tskew_mean_pred", num_predictors, scale_or_cov=pred_effect_scale)
    Tskew_mean_glm = glm.GLM(
        Tskew_mean_seasonal_effects,
        Tskew_mean_Tavg_effects,
        Tskew_mean_pred_effects,
        Tskew_mean_seasonal_Tavg_interaction_effects,
        link=glm.LogitLink(),
    )

    # Tskew diepersion
    Tskew_disp_seasonal_effects = glm.SeasonalEffects("Tskew_disp_seasonal", freqs)
    Tskew_disp_Tavg_effects = glm.LinearEffects("Tskew_disp_Tavg", 1, scale_or_cov=0.1)
    Tskew_disp_pred_effects = glm.LinearEffects("Tskew_disp_pred", num_predictors, scale_or_cov=pred_effect_scale)
    Tskew_disp_glm = glm.GLM(
        Tskew_disp_seasonal_effects,
        Tskew_disp_Tavg_effects,
        Tskew_disp_pred_effects,
        link=glm.LogLink(),
    )

    def step(state, inputs, Trange_obs=None, Tskew_obs=None):

        Trange_obs_scaled = Trange_obs
        Trange_max = Trange_max_default
        if Trange_obs is not None:
            # Rescale observations to [0, 1]
            Trange_obs_scaled = Trange_obs / Trange_max

        Tavg, prec, Trange_prev = state
        Trange_prev_scaled = Trange_prev / Trange_max
        t, year, month, doy = inputs[:, :4].T
        predictors = inputs[:, 4:]

        is_dry = 1 - jnp.sign(prec)
        lag_preds = jax.scipy.special.logit(Trange_prev_scaled)
        Tavg_scaled = jnp.sign(Tavg) * jnp.log(1 + jnp.square(Tavg)) / 2

        # Trange
        ## parameterize alpha and beta with mean and dispersion;
        ## we use the inverse of the modeled variable so that larger effects imply a larger dispersion.
        Trange_mean, _ = Trange_mean_glm(
            t,
            lag_preds,
            predictors,
            (lag_preds, t),
            Tavg_scaled,
            (Tavg_scaled, t),
        )
        Trange_disp_inv, _ = Trange_disp_glm(t, lag_preds, predictors, (lag_preds, t))
        Trange_alpha = Trange_mean * 10 / Trange_disp_inv
        Trange_beta = (1 - Trange_mean) * 10 / Trange_disp_inv

        # Tskew
        Tskew_mean, _ = Tskew_mean_glm(t, Tavg_scaled, predictors, (Tavg_scaled, t))
        Tskew_disp_inv, _ = Tskew_disp_glm(t, Tavg_scaled, predictors)
        Tskew_alpha = Tskew_mean * 10 / Tskew_disp_inv
        Tskew_beta = (1 - Tskew_mean) * 10 / Tskew_disp_inv

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
        Tmin = numpyro.deterministic("Tmin", Tavg.squeeze() - Tskew * Trange)
        Tmax = numpyro.deterministic("Tmax", Tmin + Trange)
        return Trange, Tskew, Tmin, Tmax

    return step
