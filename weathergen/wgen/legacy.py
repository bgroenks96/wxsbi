import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import gamma

from ..utils import fourier_lsq


def estimate_wgen_params(
    data: pd.DataFrame,
    precip_name="prec",
    Tavg_name="Tair_mean",
    Tmin_name="Tair_min",
    Tmax_name="Tair_max",
    Trange_name="Tair_range",
    Tskew_name="Tair_skew",
    num_harmonics=1,
    prec_thresh=0.0,
):
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
    prec_gamma_monthly = prec.resample("1ME").agg(
        lambda x: pd.DataFrame(
            np.array(gamma.fit(x[x > 0])).reshape((1, -1)), columns=["gamma_shape", "gamma_loc", "gamma_scale"]
        )
    )
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
    f0 = 1 / 365.25
    freqs = jnp.arange(0, num_harmonics) * f0
    coef_tavg, ff = fourier_lsq(t_idx, jnp.array(tavg.values), freqs)
    Tavg_resid = tavg - jnp.dot(ff, coef_tavg)
    # fit fourier coefficients for range and skew in transformed space
    coef_trange, _ = fourier_lsq(t_idx, jnp.log(trange.values), freqs)
    coef_tskew, _ = fourier_lsq(t_idx, jax.scipy.special.logit(tskew.values), freqs)
    # aggregate
    agg_ops = {Tavg_name: "mean", Tmin_name: "mean", Tmax_name: "mean", precip_name: "sum"}
    data_monthly = data[[Tavg_name, Tmin_name, Tmax_name, precip_name]].resample("1ME").aggregate(agg_ops)
    trange_monthly = trange.resample("1ME").mean()
    tskew_monthly = tskew.resample("1ME").mean()
    para_monthly = pd.concat(
        [
            data_monthly,
            trange_monthly.to_frame(name=Trange_name),
            tskew_monthly.to_frame(name=Tskew_name),
            tavg_wet_monthly.to_frame().rename({Tavg_name: f"{Tavg_name}_wet"}, axis=1),
            tmax_wet_monthly.to_frame().rename({Tmax_name: f"{Tmax_name}_wet"}, axis=1),
            tmin_wet_monthly.to_frame().rename({Tmin_name: f"{Tmin_name}_wet"}, axis=1),
            trange_wet_monthly.to_frame().rename({0: f"{Trange_name}_wet"}, axis=1),
            tskew_wet_monthly.to_frame().rename({0: f"{Tskew_name}_wet"}, axis=1),
            tavg_dry_monthly.to_frame().rename({Tavg_name: f"{Tavg_name}_dry"}, axis=1),
            tmax_dry_monthly.to_frame().rename({Tmax_name: f"{Tmax_name}_dry"}, axis=1),
            tmin_dry_monthly.to_frame().rename({Tmin_name: f"{Tmin_name}_dry"}, axis=1),
            trange_dry_monthly.to_frame().rename({0: f"{Trange_name}_dry"}, axis=1),
            tskew_dry_monthly.to_frame().rename({0: f"{Tskew_name}_dry"}, axis=1),
            pdd_ref_monthly.rename({precip_name: "pdd"}, axis=1),
            pwd_ref_monthly.rename({precip_name: "pwd"}, axis=1),
            prec_gamma_monthly.reset_index().drop("index", axis=1).set_index(data_monthly.index),
            ptot_monthly.to_frame().rename({precip_name: "ptot"}, axis=1),
            prec_avg_monthly.to_frame().rename({precip_name: "prec_avg"}, axis=1),
            wet_days_monthly.to_frame().rename({precip_name: "wet_days"}, axis=1),
        ],
        axis=1,
    )
    para = {
        "freqs": freqs,
        "prec": {
            "pdd": para_monthly.pdd,
            "pwd": para_monthly.pwd,
            "mean": para_monthly.prec_avg,
        },
        "Tair": {
            "coef": {"Tavg": coef_tavg, "Trange": coef_trange, "Tskew": coef_tskew},
            "mean": data_monthly[Tavg_name],
            "range": para_monthly[Trange_name],
            "skew": para_monthly[Tskew_name],
            "rho": Tavg_resid.autocorr(),
            "resid_std": Tavg_resid.std(),
            "max_offset_dry": (tmax_dry_monthly - tavg_dry_monthly).mean(),
            "max_offset_wet": (tmax_wet_monthly - tavg_wet_monthly).mean(),
            "min_offset_dry": (tmin_dry_monthly - tavg_dry_monthly).mean(),
            "min_offset_wet": (tmin_wet_monthly - tavg_wet_monthly).mean(),
        },
    }
    return para, para_monthly
