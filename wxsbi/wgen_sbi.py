import jax
import jax.numpy as jnp

import torch
import torch.distributions as torchdist

import numpyro
import numpyro.distributions as dist

import logging
logger = logging.getLogger(__name__)

from weathergen import WGEN

from jax2torch.jax2torch import jax2torch, j2t, t2j

from sbi.inference import SNPE, SNLE, simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator

from tqdm import tqdm

from .utils import Torch2NumPyroDistribution, check_if_list_in_string

def get_parameter_mask(guide, ignore_elems):
    return jnp.concat([jnp.zeros_like(x) if check_if_list_in_string(ignore_elems, k) else jnp.ones_like(x) for k, x in guide._init_locs.items()])

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
    
def run_sbi(
    wgen: WGEN,
    prior: dist.MultivariateNormal,
    observable_fn,
    obs_target,
    num_rounds = 1,
    simulations_per_round = 1000,
    simulation_batch_size = None,
    sbi_alg = SNPE,
):
    """
    Run SBI on a weather generator to calibrate it to a set of target statistics.

    Args:
        wgen: A WGEN object including a weather generator.
        prior: Prior to use for SBI as numpyro distribution; currently assumed to be a multivariate normal.
        observable_fn: A function taking as input arrays of realizations of ts, precip, Tmin, Tavg, Tmax and outputting summary statistics.
        obs_target: A vector of target summary statistics.
        scale_factor: Scaling amount for the posterior to obtain the SBI prior.
        parameter_mask: A mask to keep certain parameters from being scaled up. Helps with the SBI fit.
        
    Returns:
        tuple[array,array | None]: outputs, observable

    """
    # parameter checks
    simulation_batch_size = simulations_per_round if simulation_batch_size is None else simulation_batch_size
    assert simulation_batch_size <= simulations_per_round, "batch size must be <= number of simulations per round"
    assert num_rounds >= 1, "number of rounds must be >= 1"
    
    # Build simulator and SBI prior
    sim, _ = wgen.simulator(observable=observable_fn, rng_seed=1234)
    torchprior = torchdist.MultivariateNormal(j2t(prior.mean), j2t(prior.covariance_matrix))

    logger.info("Running SBI...")

    # Run SBI
    device = torchprior.mean.device if prior.mean.device.platform == "gpu" else "cpu"
    sbi_alg = sbi_alg(prior=torchprior, device=device)
    proposal = torchprior
    for i in range(num_rounds):
        logger.info(f"Starting round {i+1}/{num_rounds}")
        # Draw samples from proposal
        theta = proposal.sample((simulations_per_round,))
        # Run batched simulations
        logger.info(f"Running {simulations_per_round} simulations...")
        if simulation_batch_size < simulations_per_round:
            xs = []
            for i in tqdm(range(0, simulations_per_round, simulation_batch_size)):
                lo = i
                hi = min(i + simulation_batch_size, simulations_per_round)
                xs.append(j2t(sim(t2j(theta[lo:hi]))))
            x = torch.cat(xs, dim=0).squeeze(-1)
        else:
            x = j2t(sim(t2j(theta))).squeeze(-1)
        # Append simulations to estimtaor
        sbi_alg.append_simulations(theta, x, proposal, exclude_invalid_x=True)
        # Train estimator
        logger.info("Training estimator")
        density_estimator = sbi_alg.train()
        # Build posterior and update proposal
        sbi_posterior = sbi_alg.build_posterior(density_estimator).set_default_x(j2t(obs_target.squeeze()))
        proposal = sbi_posterior
        
    return Torch2NumPyroDistribution(sbi_posterior)
