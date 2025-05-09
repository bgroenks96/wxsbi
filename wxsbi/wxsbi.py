import jax
import jax.numpy as jnp

import torch
import torch.distributions as torchdist

import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal

import logging
logger = logging.getLogger(__name__)

from abc import ABC
from jax2torch.jax2torch import j2t, t2j
from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior
from tqdm import tqdm

from weathergen.distributions import StochasticFunctionDistribution
from weathergen.types import AbstractTimeSeriesModel

from typing import Union

from .summarizers import Summarizer
from .utils import *


class BatchSimulator(ABC):
    """Wrapper type for simulator functions that allows for batched evaluation.
    """
    
    def __init__(self, sim_func, summarizer, prior, default_prng=jax.random.PRNGKey(0)):
        self.sim_func = sim_func
        self.summarizer = summarizer
        self.prior = prior
        self.default_prng = default_prng
        super().__init__()
        
    def __call__(self, theta, batch_size=None, prng=None):    
        assert len(theta.shape) == 2, "theta must be a matrix of shape n_samples x n_params"
        num_samples = theta.shape[0]
        batch_size = num_samples if batch_size is None else batch_size
        prng = self.default_prng if prng is None else prng
        # Run simulations
        if batch_size < num_samples:
            xs = []
            for i in tqdm(range(0, num_samples, batch_size)):
                lo = i
                hi = min(i + batch_size, num_samples)
                xs.append(self.sim_func(theta[lo:hi], prng))
            x = jnp.concat(xs, axis=0)
        else:
            x = self.sim_func(theta, prng)
        return x

class SBIResults:
    def __init__(
        self,
        simulator: BatchSimulator,
        sbi_prior,
        sbi_posterior,
        calibration_posterior,
        summary_target,
        samples: dict,
        simulations: dict,
    ):
        self.simulator = simulator
        self.sbi_prior = sbi_prior
        self.sbi_posterior = sbi_posterior
        self.calibration_posterior = calibration_posterior
        self.summary_target = summary_target
        self.samples = samples
        self.simulations = simulations
    
    def with_target(self, summary_target, simulation_batch_size=None, rng_seed=1234, map_kwargs=dict()):
        """Resamples from the SBI posterior for an alternative choice of `summary_target` and returns a
        new `SBIResults` object with the updated posterior samples and simulations.

        Args:
            summary_target (_type_): new target to use
            simulation_batch_size (_type_, optional): batch size for simulations. Defaults to None.
            rng_seed (int, optional): random seed. Defaults to 1234.
            map_kwargs (_type_, optional): keyword arguments for MAP estimation. Defaults to dict().

        Returns:
            _type_: _description_
        """
        num_samples = self.samples["sbi_posterior"].shape[0]
        simulation_batch_size = num_samples if simulation_batch_size is None else simulation_batch_size
        (theta_post, x_post), (theta_map, x_map) = simulate_from_sbi_posterior(
            self.simulator,
            self.sbi_posterior,
            summary_target,
            num_samples,
            simulation_batch_size,
            rng_seed,
            map_kwargs,
        )
        samples = self.samples.copy()
        simulations = self.simulations.copy()
        samples["sbi_posterior"] = theta_post
        simulations["sbi_posterior"] = x_post
        samples["sbi_posterior_map"] = theta_map
        simulations["sbi_posterior_map"] = x_map
        return SBIResults(self.simulator, self.sbi_prior, self.sbi_posterior, self.calibration_posterior, summary_target, samples, simulations)

def build_simulator(
    model: AbstractTimeSeriesModel,
    summarizer: Summarizer,
    *prior_args,
    parallel=True,
    rng_seed=0,
    **prior_kwargs,
):
    """Constructs a wrapper function `f(theta)` that invokes `simulate` with the given `summarizer`.
    The parameters `theta` are assumed to be in the unconstrained (transformed) sample space of the model.
    Returns a `BatchSimulator` wrapper and optionally the corresponding prior distribution.

    Args:
        model (AbstractTimeSeriesModel): generative time series model to build simulator from.
        summarizer (Summarizer): summary stats function, generated with `@summarystats` decorator.
        parallel (bool, optional): whether or not to execute simulations in parallel on CPU or GPU. Defaults to True.
        rng_seed (int, optional): random seed. Defaults to 0.

    Returns:
        BatchSimulator: batched simulator wrapper
    """
    assert isinstance(model, AbstractTimeSeriesModel), "model must be an implementation of "
    prior = StochasticFunctionDistribution(
        model.prior,
        fn_args=prior_args,
        fn_kwargs=prior_kwargs,
        unconstrained=True,
        rng_seed=rng_seed,
    )
    
    default_prng = jax.random.PRNGKey(rng_seed)

    def simulator(_theta, prng=default_prng):
        theta = _theta if len(_theta.shape) > 1 else _theta.reshape((1, -1))
        # batch_size = theta.shape[0]
        params = {
            k: v for k, v in prior.constrain(theta, as_dict=True).items()
        }
        # run simulations using numpyro predictive
        predictive = Predictive(model.simulate, posterior_samples=params, parallel=parallel)
        preds = predictive(prng)
        # splat predicted variables into summary stats function
        return summarizer(**preds)

    return BatchSimulator(simulator, summarizer, prior, default_prng)

def run_sbi(
    simulator: BatchSimulator,
    summary_target: jax.Array,
    prior: dist.Distribution = None,
    calibration_posterior: dist.Distribution = None,
    num_samples: int = 1000,
    num_rounds: int = 1,
    simulations_per_round: int = 1000,
    simulation_batch_size: int = None,
    sbi_alg = SNPE,
    map_kwargs: dict = dict(),
    rng_seed: int = 1234,
):
    """
    Run SBI on a weather generator to calibrate it to a set of target statistics.

    Args:
        simulator: A Simulator which maps from parameters to simulation ou tputs (summary statistics).
        summary_target: A vector of target summary statistics.
        prior (dist.Distribution): a prior (and initial proposal) distribution to use for SBI as numpyro distribution.
        
    Returns:
        SBIResults
    """
    # parameter checks
    simulation_batch_size = simulations_per_round if simulation_batch_size is None else simulation_batch_size
    assert simulation_batch_size <= simulations_per_round, "batch size must be <= number of simulations per round"
    assert num_rounds >= 1, "number of rounds must be >= 1"
    assert isinstance(simulator, BatchSimulator)
    
    # PRNG
    prng = jax.random.PRNGKey(rng_seed)
    
    # Dict for collecting results
    samples = dict()
    simulations = dict()
    
    if calibration_posterior is not None:
        # Posterior mean
        logger.info(f"Running {num_samples} simulations for calibration posterior mean")
        theta_cm = calibration_posterior.mean.reshape((1,-1))*jnp.ones((num_samples,1))
        x_cm = simulator(theta_cm, batch_size=simulation_batch_size, prng=prng)
        # Full calibration posterior
        logger.info(f"Running {num_samples} simulations for full calibration posterior")
        theta_cp = calibration_posterior.sample(prng, (num_samples,))
        x_cp = simulator(theta_cp, batch_size=simulation_batch_size, prng=prng)
        # store results in dict(s)
        samples["calibration_posterior_mean"] = theta_cm
        simulations["calibration_posterior_mean"] = x_cm
        samples["calibration_posterior"] = theta_cp
        simulations["calibration_posterior"] = x_cp
    
    # Generate prior samples/simulations
    logger.info(f"Running {num_samples} simulations for SBI prior")
    prior = simulator.prior if prior is None else prior
    theta_prior = prior.sample(prng, (num_samples,))
    x_prior = simulator(theta_prior, batch_size=simulation_batch_size, prng=prng)
    samples["sbi_prior"] = theta_prior
    simulations["sbi_prior"] = x_prior
    
    # Convert prior to torch distribution
    if isinstance(prior, dist.MultivariateNormal):
        torchprior = torchdist.MultivariateNormal(j2t(prior.mean), j2t(prior.covariance_matrix))
    else:
        torchprior, *_ = process_prior(NumPyro2TorchDistribution(prior))

    logger.info("Running SBI...")

    # Run SBI
    dummy_sample = torchprior.sample((1,))
    device = "cpu" if dummy_sample.is_cpu else dummy_sample.device
    sbi_alg = sbi_alg(prior=torchprior, device=device)
    proposal = torchprior
    for i in range(num_rounds):
        logger.info(f"Starting round {i+1}/{num_rounds}")
        # Sample from proposal
        theta = proposal.sample((simulations_per_round,))
        # Run batched simulations
        logger.info(f"Running {simulations_per_round} simulations...")
        x = simulator(t2j(theta), batch_size=simulation_batch_size, prng=prng)
        # Append simulations to estimtaor
        sbi_alg.append_simulations(theta, j2t(x), proposal, exclude_invalid_x=True)
        # Train estimator
        logger.info("Training estimator")
        density_estimator = sbi_alg.train()
        # Build posterior and update proposal
        sbi_posterior = sbi_alg.build_posterior(density_estimator).set_default_x(j2t(summary_target.squeeze()))
        proposal = sbi_posterior
        
    (theta_post, x_post), (theta_map, x_map) = simulate_from_sbi_posterior(
        simulator,
        sbi_posterior,
        summary_target,
        num_samples,
        simulation_batch_size,
        rng_seed,
        map_kwargs,
    )
    samples["sbi_posterior"] = theta_post
    simulations["sbi_posterior"] = x_post
    samples["sbi_posterior_map"] = theta_map
    simulations["sbi_posterior_map"] = x_map
        
    return SBIResults(simulator, prior, sbi_posterior, calibration_posterior, summary_target, samples, simulations)

def simulate_from_sbi_posterior(
    simulator: BatchSimulator,
    sbi_posterior,
    summary_target,
    num_samples=1000,
    simulation_batch_size=None,
    rng_seed=1234,
    map_kwargs=dict(),
):
    """Convenience method for sampling from the given SBI posterior and corresponding posterior predictive.

    Args:
        simulator (_type_): _description_
        sbi_posterior (_type_): _description_
        summary_target (_type_): _description_
        num_samples (int, optional): _description_. Defaults to 1000.
        simulation_batch_size (_type_, optional): _description_. Defaults to None.
        rng_seed (int, optional): _description_. Defaults to 1234.
        map_kwargs (_type_, optional): _description_. Defaults to dict().

    Returns:
        _type_: _description_
    """
    # reset torch RNG seed
    torch.manual_seed(rng_seed)
    theta_post = t2j(sbi_posterior.sample((num_samples,), x=j2t(summary_target)))
  
    logger.info(f"Running {num_samples} simulations for SBI posterior")
    prng = jax.random.PRNGKey(rng_seed)
    x_post = simulator(theta_post, batch_size=simulation_batch_size, prng=prng)
        
    # obtain MAP estimate
    logger.info(f"Finding MAP estimate")
    theta_map = t2j(sbi_posterior.set_default_x(j2t(summary_target)).map(**map_kwargs))
    
    # broadcast to num_samples and run simulations
    logger.info(f"Running {num_samples} simulations for SBI posterior MAP estimate")
    x_map = simulator(theta_map.reshape((1,-1))*jnp.ones((num_samples,1)), batch_size=simulation_batch_size, prng=prng)
    return (theta_post, x_post), (theta_map, x_map)

def get_rescaled_svi_posterior(
    guide: AutoMultivariateNormal,
    svi_result,
    parameter_mask = None,
    scale_factor = 2.0,
    diagonal = True,
):
    """Rescales the variance of the given SVI multivariate normal posterior.

    Args:
        guide (AutoMultivariateNormal): SVI autoguide; currently only `AutoMultivariateNormal` is supported.
        svi_result (_type_): SVI result
        parameter_mask (_type_, optional): parameter mask to apply. Defaults to None.
        scale_factor (float, optional): square root of the variance scaling factor. Defaults to 2.0.
        diagonal (bool, optional): whether or not to use only the diagonal of the SVI posterior. Defaults to True.

    Returns:
        dist.MultivariateNormal: A `MultivariateNormal` distribution with rescaled covaraince.
    """
    fitted_posterior = guide.get_posterior(svi_result.params)
    parameter_mask = 1.0 if parameter_mask is None else parameter_mask
    # full covariance matrix
    C = fitted_posterior.covariance_matrix
    # covariance diagonal
    D = jnp.diag(fitted_posterior.covariance_matrix)
    # identity matrix
    I = jnp.eye(len(fitted_posterior.mean))
    if diagonal:
        rescaled_cov = D*I + parameter_mask*(scale_factor**2 - 1)*I*D
    else:
        rescaled_cov = C + parameter_mask*(scale_factor**2 - 1)*I*D
    return dist.MultivariateNormal(fitted_posterior.mean, rescaled_cov)