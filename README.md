# wxsbi
Toolkit for building stochastic weather generators using [numpyro](https://num.pyro.ai/en/stable/index.html) and scenario-neutral calibration via [simulation-based inference](https://sbi-dev.github.io/sbi/0.22/).

Authors: Brian Groenke (AWI Potsdam, UFZ Leipzig, TU Berlin), Jakob Wessel (University of Exeter)

## Quick start

The WGEN-type weather generators can be easily defined and calibrated on (daily) meterological time series data as follows:

```python
# define name map for columns (optional)
name_map = wx.data.data_var_name_map(prec="pre", Tair_mean="tavg", Tair_min="tmin", Tair_max="tmax")
# load time series dataset with optional name_map argument
data = wx.data.load_time_series_csv("path/to/data_file.csv", name_map)
# construct WGEN
wgen = wx.WGEN(data)
# fit using stochastic variational inference
fit_result = wgen.fit("svi")
```

Currently, the WGEN-GLM weather generator supports mean, minimum, and maximum daily air temperature as well as precipitation.

## Scenario-neutral calibration with simulation-based inference (SBI)

WIP

## Disclaimer

This is experimental research software and is not intended for use in production.
