# wxsbi
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15448999.svg)](https://doi.org/10.5281/zenodo.15448999)

A Pyton toolkit for building stochastic weather generators using [numpyro](https://num.pyro.ai/en/stable/index.html) and [simulation-based inference](https://sbi-dev.github.io/sbi/0.22/).

Authors: Brian Groenke (Potsdam Institute for Climate Impact Research - PIK), Jakob Wessel (University of Exeter)

## Disclaimer

This is experimental research software and is not intended for use in production.

The code is still under heavy development and needs further refactoring, so breaking changes should be expected in the near future.

## Summary

[Stochastic weather generators](https://journals.sagepub.com/doi/abs/10.1177/030913339902300302) are statistical models that aim to simulate time series of meteorological variables based on historical observations.
This should be distinguished from weather forecasting or prediction, in that the goal is not necessarily to predict future states of the atmosphere, but rather to obtain a reasonable estimate of the distribution over weather patterns for a given location or region.
Weather generation is closely related to the problem of [statistical downscaling](https://en.wikipedia.org/wiki/Downscaling), though the primary difference is that weather generators do not necessarily need to be based on large-scale variables as predictors.

There are four basic types of weather generators:
1. Stochastic process models
2. Hierarchical and state-space models
3. Autoregressive and Markov-type models
4. Resampling and analog methods

This package deals primarily with type 3 (autoregressive/Markov-type models), though in principle, all of these models would be possible to implement using a PPL such as `numpyro`.

## Quick start

A WGEN-type weather generator can be easily defined and calibrated on (daily) meterological time series data as follows:

```python
# define name map for columns (optional)
name_map = wx.data.data_var_name_map(prec="pre", Tair_mean="tavg", Tair_min="tmin", Tair_max="tmax")
# load time series dataset with optional name_map argument
data = wx.data.load_time_series_csv("path/to/data_file.csv", name_map)
# construct WGEN
wgen = wx.WGEN(data)
# fit using stochastic variational inference
fit_result = wgen.fit()
```
The `WGEN` class encapsulates all relevant data, parameters, and configuration settings for the weather generator. Different model confifgurations can be selected via the `model` keyword argument. The default model is the one used in the manuscript, `WGEN_GAMLSS`.

Currently, the WGEN-GAMLSS weather generator supports mean, minimum, and maximum daily air temperature as well as precipitation.

## Climatology matching with simulation-based inference (SBI)

*Climatology matching* refers to the problem of generating meterological time series with a deisred set of climatological characteristics, typically defined using *summary statistics*.

As first discussed by [Guo et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0022169416301305), this can be posed as a type of inverse problem where the goal is to recover one or more parameters for a particular weather generator given a set of target summary statistics.

The downstream objective of climatology matching is to facilitate so-called "scenario-netural" impact analyses which allow for the analysis of climate change impacts decoupled from large-scale socioeconomic emissions scenarios.

The `wxsbi` module allows for easy application of simulation-based inference (SBI) to do climatology matching on WGEN-type weather generators.

## Installation

Currently, the python package(s) can only be installed directly from GitHub:

```
pip install git+https://github.com/bgroenks96/wxsbi
```

Note that the repository currently consists of two separate modules with different dependencies: `weathergen` for weather generation and `wxsbi` for climatology matching.

## Acknowledgements

This work was made possible through funding from the Helmholtz Center for Environmental Research (UFZ) Leipzig and the Helmholtz Association.
