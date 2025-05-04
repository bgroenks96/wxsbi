import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import scipy.stats as stats

def batch_select(x, idx):
    return jax.vmap(lambda x,idx: x[idx])(x*jnp.ones((idx.shape[0],x.shape[1])), idx)

def check_if_list_in_string(l, x):
    for elem in l:
        if elem in x:
            return True
    return False

def extract_time_vars(timestamps):
    if isinstance(timestamps, pd.DatetimeIndex):
        batch_size = 1
        tvars = jnp.array(list(zip(jnp.arange(0, len(timestamps)), timestamps.year, timestamps.month, timestamps.dayofyear)))
        return jnp.expand_dims(tvars, axis=0)
    else:
        batch_size = len(timestamps)
        return jnp.concat([extract_time_vars(ts) for ts in timestamps], axis=0)

def dry_spells(precip, threshold=0.0):
    """Counts the length of each dry spell and stores the result on the first wet
    day after the dry spell ends. The returned time series will always have length
    `N+1` where `N` is the length of `precip`.
     
    Args:
        precip (_type_): A sequence of daily precipitation amounts.
        threshold (float, optional): Dry day threshold. Defaults to 0.0

    Returns:
        _type_: `N+1` length sequence of dry spell lengths.
    """     
    # Get mask of each dry day with one additional False entry appended to
    # ensure that dry spells at the end of the series are counted.
    is_dry_day = jnp.concatenate([precip <= threshold, jnp.zeros((1,*precip.shape[1:]))], axis=0)

    # Use `jax.lax.scan` to accumulate the number of consecutive dry days
    def scan_fn(cdd, x):
        # cdd is the count of consecutive dry days so far
        # x is the current day's dry day status (True or False)
        new_cdd = cdd * x + x
        return new_cdd, cdd*(1-x)

    # Initialize the carry as 0 (no dry days yet)
    initial_carry = jnp.zeros(precip.shape[1:])

    # Scan over the is_dry_day array
    _, dry_spells = jax.lax.scan(scan_fn, initial_carry, is_dry_day)

    return dry_spells

def cumulative_dry_days(precip, threshold=0.0, axis=0):
    """Computes average cumulative dry days (CDD) as the mean length of all dry spells.

    Args:
        precip (_type_): A sequence of daily precipitation amounts.
        threshold (float, optional): Dry day threshold. Defaults to 0.0.

    Returns:
        _type_: Scalar average CDD
    """    
    spells = dry_spells(precip, threshold)
    cdd = jnp.nanmean(jnp.where(spells > 0, spells, jnp.nan), axis=axis)
    return jnp.where(jnp.isfinite(cdd), cdd, 0.0)

def precip_summary_stats(pr, cdd_thresh=0.0, axis=0):
    pmean = jnp.mean(pr, axis=axis)
    p99 = jnp.nanquantile(jnp.where(pr > 0, pr, jnp.nan), 0.99, axis=axis)
    fwd = jnp.mean(pr > 0, axis=axis)
    cdd = cumulative_dry_days(pr, threshold=cdd_thresh, axis=axis)
    return jnp.stack([pmean, p99, fwd, cdd])

def tair_summary_stats(Tair_mean, Tair_min, Tair_max):
    Tair90 = jnp.quantile(Tair_mean, 0.99, axis=0)
    Tair50 = jnp.quantile(Tair_mean, 0.50, axis=0)
    Tair10 = jnp.quantile(Tair_mean, 0.01, axis=0)
    Trange = jnp.mean(Tair_max - Tair_min, axis=0)
    Tskew = jnp.mean((Tair_mean - Tair_min) / (Tair_max - Tair_min), axis=0)
    tdd = jnp.mean(jnp.where(Tair_mean > 0, Tair_mean, 0), axis=0)
    fdd = jnp.mean(jnp.where(Tair_mean <= 0, Tair_mean, 0), axis=0)
    return jnp.stack([Tair10, Tair50, Tair90, Trange, Tskew, tdd, fdd])

def tair_minmax_summary_stats(Tair_min, Tair_max):
    Tair_mid = (Tair_min + Tair_max)/2
    Tair90 = jnp.quantile(Tair_mid, 0.99, axis=0)
    Tair50 = jnp.quantile(Tair_mid, 0.50, axis=0)
    Tair10 = jnp.quantile(Tair_mid, 0.01, axis=0)
    hot_days = jnp.sum(Tair_max > Tair90, axis=0)
    cold_days = jnp.sum(Tair_min < Tair10, axis=0)
    return jnp.array([Tair50, hot_days, cold_days])

def fourier_feats(t, freqs=[1/365.25], intercept=True):
    """Returns a matrix of Fourier series features with the given frequencies over time stamps `t`.
    
    Args:
        t (_type_): timestamps for input to sine/cosine functions
        freqs (list, optional): list of frequencies to use. Defaults to [1/365.2] (annual).
    """
    t = jnp.array(t).reshape((-1,1))
    freqs = jnp.array(freqs).reshape((1,-1))
    shift = jnp.ones((t.shape[0],1))
    s = jnp.sin(2*jnp.pi*jnp.matmul(t, freqs))
    c = jnp.cos(2*jnp.pi*jnp.matmul(t, freqs))
    xs = [shift, s, c] if intercept else [s,c]
    X = jnp.concatenate(xs, axis=1)
    return X

def fourier_lsq(t, y, freqs=[1/365.25], intercept=True):
    """Linear least squares of a sum of sine/cosine waves with the given frequencies and observed signal `y`.
    Returns a tuple `(coef, X)` where `coef` are the coefficients and `X` is the design matrix.

    Args:
        t (_type_): timestamps for input to sine/cosine functions
        y (_type_): signal observations
        freqs (list, optional): list of frequencies to use. Defaults to [1/365.2] (annual).
    """
    assert t.shape[0] == y.shape[0], "number of timesteps must match number of observations"
    t = jnp.array(t).reshape((-1,1))
    y = jnp.array(y).reshape((-1,))
    X = fourier_feats(t, freqs, intercept)
    coef = jnp.matmul(jnp.linalg.pinv(X), y)
    return coef, X

def truncated_fft(X, nfreqs=5, axis=0, detrend=True, return_input=False):
    """Truncated Fourier analysis of signal X with the given sample axis and number of frequencies.
    Returns a DataFrame with the frequencies, amplitudes, and phases of the corresponding
    truncated fourier series.

    Args:
        X (_type_): _description_
        nfreqs (int, optional): _description_. Defaults to 1.
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    X_de = signal.detrend(X, type='linear', axis=axis) if detrend else X
    X_fft = fft.fft(X_de, norm="forward")
    freqs = fft.fftfreq(X_de.shape[axis])
    fft_sortidx = np.argsort(np.abs(X_fft[:X_fft.shape[0]//2]))[::-1]
    topfreqs = fft_sortidx[:nfreqs]
    amp = np.sqrt(X_fft[topfreqs].real**2 + X_fft[topfreqs].imag**2)
    phase = np.arctan(X_fft[topfreqs].imag / X_fft[topfreqs].real)
    freq = freqs[topfreqs]
    result = pd.DataFrame.from_dict({'amp': amp, 'phase': phase, 'freq': freq, 'period': 1/freq})
    if return_input:
        return result, X_de
    else:
        return result

# def quantile_map(obs: pd.Series, model: pd.Series, num_quantiles=100_000, interpolation="linear"):
#     model_cdf = ecdf(model.values).cdf
#     model_qs = model_cdf.evaluate(model.loc[obs.index].values)
#     xs = np.linspace(0, 1, num_quantiles)
#     obs_qs = obs.quantile(xs, interpolation=interpolation)
#     if interpolation == "linear":
#         ys = np.interp(model_qs, xs, obs_qs)
#     else:
#         raise(Exception(f"unrecognized interpolation mode '{interpolation}'"))
#     return ys

# The following functions are copied from:
# https://github.com/ecmwf-projects/ibicus/blob/main/ibicus/utils/_math_utils.py

def IECDF(x):
    """
    Get the inverse empirical cdf of an array of data:

    x = np.random.random(1000)
    iecdf = IECDF(x)
    iecdf(0.2)

    Up to numerical accuracy this returns the same as np.quantile(x, q, method = "inverted_cdf") but is much faster.

    Parameters
    ----------
    x : array
        Array containing values for which the empirical cdf shall be calculated.

    Returns
    -------
    lambda
        Function to calculate the inverse empirical cdf-value for a given quantile q.
    """
    y = np.sort(x)
    n = y.shape[0]
    return lambda q: y[np.floor((n - 1) * q).astype(int)]


def iecdf(x: np.ndarray, p: np.ndarray, method: str = "inverted_cdf", **kwargs):
    """
    Return the values of the the inverse empirical CDF of x evaluated at p:

    The call is delegated to :py:func:`np.quantile` with the method-argument determining what method is used.

    Examples
    --------

    >>> x = np.random.normal(size = 1000)
    >>> p = np.linspace(0, 1, 100)
    >>> iecdf(x, p)


    Parameters
    ----------
    x : np.ndarray
        Array containing values with which the inverse empirical cdf is defined.
    p : np.ndarray
        Array containing values between [0, 1] for which the inverse empirical cdf is evaluated.
    method : string
        Method string for :py:func:`np.quantile`.
    **kwargs
        Passed to :py:func:`np.quantile`.

    Returns
    -------
    array
        Values of the inverse empirical cdf of x evaluated at p.
    """
    if method == "inverted_cdf":
        # This method is much faster actually than using np.quantile.
        # Although it is slightly akward for the sake of efficiency we refer to IECDF.
        iecdf = IECDF(x)
        return iecdf(p)
    else:
        return np.quantile(x, p, method=method, **kwargs)


def ecdf(x: np.ndarray, y: np.ndarray, method: str = "linear_interpolation") -> np.ndarray:
    """
    Return the values of the empirical CDF of x evaluated at y.

    Three methods existd determined by method.

    1. ``method = "kernel_density"``: A kernel density estimate of the ecdf is used, using :py:class:`scipy.stats.rv_histogram`.

    2. ``method = "linear_interpolation"``: Linear interpolation is used, starting from a grid of CDF-values.

    3. ``method = "step_function"``: The classical step-function.


    Examples
    --------

    >>> x = np.random.random(1000)
    >>> y = np.random.random(100)
    >>> ecdf(x, y)


    Parameters
    ----------
    x : np.ndarray
        Array containing values with which the empirical cdf is defined.
    y : np.ndarray
        Array containing values on which the empirical cdf is evaluated.
    method : str
        Method with which the ecdf is calculated. One of ["kernel_density", "linear_interpolation", "step_function"].

    Returns
    -------
    np.ndarray
        Values of the empirical cdf of x evaluated at y.
    """
    if method == "kernel_density":
        rv_histogram_fit = stats.rv_histogram(np.histogram(x, bins="auto"))
        return rv_histogram_fit.cdf(y)
    elif method == "linear_interpolation":
        p_grid = np.linspace(0.0, 1.0, x.size)
        q_vals_for_p_grid = np.quantile(x, p_grid)
        return np.interp(y, q_vals_for_p_grid, p_grid)
    # elif method == "step_function":
    #     step_function = statsmodels.distributions.empirical_distribution.ECDF(x)
    #     return step_function(y)
    else:
        raise ValueError(
            'method needs to be one of ["kernel_density", "linear_interpolation", "step_function"] '
        )


def quantile_map_nonparametic(
    x: np.ndarray,
    y: np.ndarray,
    vals: np.ndarray,
    ecdf_method: str = "linear_interpolation",
    iecdf_method: str = "inverted_cdf",
    **kwargs,
) -> np.ndarray:
    """
    Quantiles maps a vector of values vals using empirical distributions defined by vectors x and y.
    Quantiles of values in vals are first found using the ecdf of the values in x. Afterwards they are transformed onto y using the empirical inverse cdf of y.

    Parameters:
        x: np.ndarray
            Values defining an empirical distribution with whose ecdf the quantiles are transformed.
        y: np.ndarray
            Values defining an empirical distribution with whose iecdf the quantiles are transformed.
        vals: np.ndarray
            Values to quantile map non parametically.
        ecdf_method: str
            Method to use for the ecdf (transformation of x). Passed to ecdf.
        iecdf_method: str
            Method to use for the iecdf (transformation of the quantiles). Passed to iecdf.
        **kwargs:
            Passed to iecdf.
    """

    return iecdf(y, ecdf(x, vals, method=ecdf_method), method=iecdf_method, **kwargs)
