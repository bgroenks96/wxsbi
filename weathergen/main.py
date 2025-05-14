# %%
import json
import os
import sys

import click

sys.path.append("../../src/python")

# jax
import jax
import jax.numpy as jnp
import jax.random as random

# plotting
import matplotlib.pyplot as plt
import numpy as np
import numpyro

# data loading
import pandas as pd
from jax.lib import xla_bridge
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal

import weathergen as wx

MODEL_DICT = {"wgen_glm_v2": wx.wgen_glm_v1}

WGEN_DEVICE = os.environ.get("WGEN_DEVICE", "cpu")


# %%
def set_up_wgen(config: dict):
    data_config = config["data"]
    model_config = config["model"]
    data = wx.data.load_time_series_csv(data_config["filepath"], data_config["name_map"])
    model = MODEL_DICT[model_config["name"]]
    model_kwargs = {k: v for k, v in model_config.items() if k is not "name"}
    return wx.WGEN(model, data, **model_kwargs)


def calibrate_svi(wgen: wx.WGEN, iterations: int, optim_name: str, learning_rate: float, rng_seed: int):
    guide = AutoMultivariateNormal(wgen.step, init_loc_fn=numpyro.infer.init_to_median, init_scale=0.01)
    optimizer = numpyro.optim.Adam(1e-3)
    svi = SVI(wgen.step, guide, optimizer, loss=Trace_ELBO())
    prng = random.PRNGKey(1234)


@click.command(help="Calibrate weather generator on observational data")
@click.argument("config_file", type=click.STRING, required=True)
@click.option("--workers", type=click.INT, default=4)
@click.option("--seed", type=click.INT, default=1234)
@click.option("--debug", type=click.BOOL, default=False)
def calibrate(config_file, workers, seed, debug):
    numpyro.set_host_device_count(workers)
    config = json.load(config_file)
    wgen = set_up_wgen(config)


# %%
@click.group()
def main():
    pass


main.add_command(calibrate)

# %%
if __name__ == "__main__":
    if any([d.platform == "gpu" for d in jax.devices()]) and WGEN_DEVICE is "gpu":
        numpyro.set_platform("gpu")
    print(f"current XLA device: {xla_bridge.get_backend().platform}")
    main()
