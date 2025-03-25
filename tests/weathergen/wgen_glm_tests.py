import sys

sys.path.append(".")

import numpy as np
import pandas as pd
import unittest

import weathergen as wx

from numpyro.handlers import trace, seed

def make_test_data():
    time_axis = pd.DatetimeIndex(pd.date_range("2000", "2001"))
    Tair_mean = np.random.normal(0, 1, size=time_axis.shape)
    Tair_min = Tair_mean - np.random.normal(0, 1, size=time_axis.shape) ** 2
    Tair_max = Tair_mean + np.random.normal(0, 1, size=time_axis.shape) ** 2
    prec = np.random.normal(0, 2, size=time_axis.shape) ** 2
    data = {
        "time": time_axis,
        "Tair_mean": Tair_mean,
        "Tair_min": Tair_min,
        "Tair_max": Tair_max,
        "prec": prec,
    }
    return pd.DataFrame(data).set_index("time")


def wgen_glm_smoke_test(data, model):
    wgen = wx.WGEN(data, model)
    with seed(rng_seed=0):
        # run step
        _, step_outputs = wgen.step()
        # run simulate
        sim_outputs, _ = wgen.simulate()
    # run fit for one iteration
    fit_result = wgen.fit(1)


class Test_WGEN_GLM(unittest.TestCase):

    def test_wgen_glm(self):
        data = make_test_data()
        wgen_glm_smoke_test(data, wx.wgen_glm_v2)

    def test_wgen_gamlss(self):
        data = make_test_data()
        wgen_glm_smoke_test(data, wx.WGEN_GAMLSS())


if __name__ == "__main__":
    unittest.main()
