import sys

sys.path.append(".")

import numpy as np
import scipy as sp
import pandas as pd
import unittest

import weathergen as wx

from weathergen import glm

from numpyro.handlers import seed

class Test_GLM_types(unittest.TestCase):
    
    def test_linear_univariate(self):
        with seed(rng_seed=0):
            linear_univariate = glm.LinearEffects("test", 1)
            assert len(linear_univariate) == 1
            assert len(linear_univariate.get_coefs()) == 1
            preds = linear_univariate.get_predictors(np.ones((1,1)))
            assert preds.shape == (1,1)
            
    def test_linear_multivariate(self):
        with seed(rng_seed=0):
            linear_multivariate = glm.LinearEffects("test", 2)
            assert len(linear_multivariate) == 2
            assert len(linear_multivariate.get_coefs()) == 2
            preds = linear_multivariate.get_predictors(np.ones((1,2)))
            assert preds.shape == (1,2)
            
    def test_seasonal_effects(self):
        with seed(rng_seed=0):
            seasonal1 = glm.SeasonalEffects("test1", [1/2])
            seasonal2 = glm.SeasonalEffects("test2", [1/2,1/4])
            assert len(seasonal1) == 2
            assert len(seasonal2) == 4
            t = np.array([0.0,0.5,1.0,1.5]).reshape((-1,1))
            preds1 = seasonal1.get_predictors(t)
            assert preds1.shape == (4,2), f"{preds1.shape} {preds1}"
            assert np.isclose(preds1[:,0], np.array([0.0,1.0,0.0,-1.0]), atol=1e-6).all(), f"{preds1[:,0]}"
            assert np.isclose(preds1[:,1], np.array([1.0,0.0,-1.0,0.0]), atol=1e-6).all(), f"{preds1[:,1]}"
            
    def test_interaction_effects(self):
        with seed(rng_seed=0):
            linear1 = glm.LinearEffects("test1", 2)
            linear2 = glm.LinearEffects("test2", 2)
            interaction = glm.InteractionEffects("test12", linear1, linear2)
            assert len(interaction) == len(linear1)*len(linear2)
            x1 = np.array([[1.0,2.0]])
            x2 = np.array([[3.0,4.0]])
            preds = interaction.get_predictors((x1,x2))
            assert preds.shape == (1,4), f"{preds.shape}"
            assert np.isclose(preds, np.array([[3.0,4.0,6.0,8.0]])).all(), f"{preds}"
            
    def test_glm_single_effect_identity_link(self):
        with seed(rng_seed=0):
            linear1 = glm.LinearEffects("test", 1)
            f = glm.GLM(linear1)
            y, z = f(np.ones((1,1)))
            assert y.shape == z.shape == (1,)
            assert np.isclose(z, linear1.get_coefs().sum()).all()
            assert np.isclose(y, z).all()
            
    def test_glm_two_effects_identity_link(self):
        with seed(rng_seed=0):
            linear1 = glm.LinearEffects("test1", 1)
            linear2 = glm.LinearEffects("test2", 1)
            f = glm.GLM(linear1, linear2)
            y, z = f(np.ones((1,1)), np.ones((1,1)))
            assert y.shape == z.shape == (1,)
            assert np.isclose(z, linear1.get_coefs().sum() + linear2.get_coefs().sum()).all()
            assert np.isclose(y, z).all()
            
    def test_glm_single_effect_log_link(self):
        with seed(rng_seed=0):
            linear1 = glm.LinearEffects("test", 1)
            f = glm.GLM(linear1, link=glm.LogLink())
            y, z = f(np.ones((1,1)))
            assert y.shape == z.shape == (1,)
            assert np.isclose(y, np.exp(z)).all()
            
    def test_glm_single_effect_logit_link(self):
        with seed(rng_seed=0):
            linear1 = glm.LinearEffects("test", 1)
            f = glm.GLM(linear1, link=glm.LogitLink())
            y, z = f(np.ones((1,1)))
            assert y.shape == z.shape == (1,)
            assert np.isclose(y, sp.special.expit(z)).all()
            
if __name__ == "__main__":
    unittest.main()
