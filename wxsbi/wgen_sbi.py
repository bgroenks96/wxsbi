import jax
import jax.numpy as jnp

import torch
import torch.distributions as torchdist

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

import logging
logger = logging.getLogger(__name__)

from abc import ABC

from weathergen import WGEN

from jax2torch.jax2torch import jax2torch, j2t, t2j

from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior

from tqdm import tqdm

from .utils import *

class Simulator(ABC):
    def __init__(self, sim_func, prior=None, rng_seed=1234):
        self.sim_func = sim_func
        self.prior = prior
        self.default_prng = jax.random.PRNGKey(rng_seed)
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
        simulator,
        sbi_prior,
        sbi_posterior,
        calibration_posterior,
        obs_target,
        samples: dict,
        simulations: dict,
    ):
        self.simulator = simulator
        self.sbi_prior = sbi_prior
        self.sbi_posterior = sbi_posterior
        self.calibration_posterior = calibration_posterior
        self.obs_target = obs_target
        self.samples = samples
        self.simulations = simulations
    
    def with_target(self, obs_target, simulation_batch_size=None, rng_seed=1234, map_kwargs=dict()):
        """Resamples from the SBI posterior for an alternative choice of `obs_target` and returns a
        new `SBIResults` object with the updated posterior samples and simulations.

        Args:
            obs_target (_type_): new observable target to use
            simulation_batch_size (_type_, optional): batch size for simulations. Defaults to None.
            rng_seed (int, optional): random seed. Defaults to 1234.
            map_kwargs (_type_, optional): keyword arguments for MAP estimation. Defaults to dict().

        Returns:
            _type_: _description_
        """
        num_samples = self.samples["sbi_posterior"].shape[0]
        simulation_batch_size = num_samples if simulation_batch_size is None else simulation_batch_size
        (theta_post, x_post), (theta_map, x_map) = simulate_from_sbi_posterior(
            simulator,
            self.sbi_posterior,
            obs_target,
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
        return SBIResults(self.simulator, self.sbi_prior, self.sbi_posterior, self.calibration_posterior, obs_target, samples, simulations)

def simulator(
    wgen: WGEN,
    timestamps=None,
    predictors=None,
    initial_state=None,
    observable=None,
    parallel=True,
    rng_seed=0,
    **prior_kwargs,
):
    """Constructs a wrapper function `f(theta)` that invokes `simulate` with the given `observable`.
        The parameters `theta` are assumed to be in the unconstrained (transformed) sample space of the model.
        Returns the simulator function as well as the corresponding prior distribution.

    Args:
        observable (_type_, optional): observable function for `simulate`. Defaults to None.
        rng_seed (int, optional): random seed. Defaults to 0.

    Returns:
        tuple: simulator_func, prior
    """
    timestamps = wgen.timestamps if timestamps is None else timestamps
    predictors = wgen.predictors if predictors is None else predictors
    initial_state = (
        initial_state if initial_state is not None else wgen.initial_states[:, wgen.first_valid_idx, :, :]
    )
    prior = StochasticFunctionDistribution(
        wgen.prior,
        fn_args=(predictors, initial_state),
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
        predictive = Predictive(wgen.simulate, posterior_samples=params, parallel=parallel)
        preds = predictive(prng)
        # call osbervable function if defined
        if observable is not None:
            return observable(**preds)
        else:
            return preds

    return Simulator(simulator, prior, rng_seed)
    
def run_sbi(
    simulator: Simulator,
    obs_target,
    prior = None,
    calibration_posterior = None,
    num_samples = 1000,
    num_rounds = 1,
    simulations_per_round = 1000,
    simulation_batch_size = None,
    sbi_alg = SNPE,
    map_kwargs = dict(),
    rng_seed = 1234,
):
    """
    Run SBI on a weather generator to calibrate it to a set of target statistics.

    Args:
        simulator: A Simulator which maps from parameters to simulation ou tputs (summary statistics).
        prior: Prior (and initial proposal) distribution to use for SBI as numpyro distribution.
        observable_func: A function taking as input arrays of realizations of ts, precip, Tmin, Tavg, Tmax and outputting summary statistics.
        obs_target: A vector of target summary statistics.
        scale_factor: Scaling amount for the posterior to obtain the SBI prior.
        parameter_mask: A mask to keep certain parameters from being scaled up. Helps with the SBI fit.
        
    Returns:
        tuple[array,array | None]: outputs, observable
    sim, _ = wgen.simulator(observable=observable_func, rng_seed=1234)
    """
    # parameter checks
    simulation_batch_size = simulations_per_round if simulation_batch_size is None else simulation_batch_size
    assert simulation_batch_size <= simulations_per_round, "batch size must be <= number of simulations per round"
    assert num_rounds >= 1, "number of rounds must be >= 1"
    
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
        sbi_posterior = sbi_alg.build_posterior(density_estimator).set_default_x(j2t(obs_target.squeeze()))
        proposal = sbi_posterior
        
    (theta_post, x_post), (theta_map, x_map) = simulate_from_sbi_posterior(
        simulator,
        sbi_posterior,
        obs_target,
        num_samples,
        simulation_batch_size,
        rng_seed,
        map_kwargs,
    )
    samples["sbi_posterior"] = theta_post
    simulations["sbi_posterior"] = x_post
    samples["sbi_posterior_map"] = theta_map
    simulations["sbi_posterior_map"] = x_map
        
    return SBIResults(simulator, prior, sbi_posterior, calibration_posterior, obs_target, samples, simulations)

def simulate_from_sbi_posterior(
    simulator,
    sbi_posterior,
    obs_target,
    num_samples=1000,
    simulation_batch_size=None,
    rng_seed=1234,
    map_kwargs=dict(),
):    
    # reset torch RNG seed
    torch.manual_seed(rng_seed)
    theta_post = t2j(sbi_posterior.sample((num_samples,), x=j2t(obs_target)))
  
    logger.info(f"Running {num_samples} simulations for SBI posterior")
    prng = jax.random.PRNGKey(rng_seed)
    x_post = simulator(theta_post, batch_size=simulation_batch_size, prng=prng)
        
    # obtain MAP estimate
    logger.info(f"Finding MAP estimate")
    theta_map = t2j(sbi_posterior.set_default_x(j2t(obs_target)).map(**map_kwargs))
    
    # broadcast to num_samples and run simulations
    logger.info(f"Running {num_samples} simulations for SBI posterior MAP estimate")
    x_map = simulator(theta_map.reshape((1,-1))*jnp.ones((num_samples,1)), batch_size=simulation_batch_size, prng=prng)
    return (theta_post, x_post), (theta_map, x_map)

def get_rescaled_svi_posterior(
    guide,
    svi_result,
    parameter_mask = None,
    scale_factor = 2.0,
    diagonal = True,
):
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
