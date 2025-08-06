import logging
import pickle

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import torch
import torch.distributions as torchdist
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal

logger = logging.getLogger(__name__)

from abc import ABC
from typing import Union

from jax2torch.jax2torch import j2t, t2j
from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior
from tqdm import tqdm

from weathergen.distributions import StochasticFunctionDistribution
from weathergen.types import AbstractTimeSeriesModel

from .summarizers import Summarizer
from .utils import *


class BatchSimulator(ABC):
    """Wrapper type for simulator functions that allows for batched evaluation."""

    def __init__(self, simulate_ts_func, summarizer, default_prng=jax.random.PRNGKey(0)):
        self.simulate_ts_func = simulate_ts_func
        self.summarizer = summarizer
        self.default_prng = default_prng
        super().__init__()

    def __call__(self, theta, batch_size=None, prng=None):
        """Alias for BatchSimulator.simulate."""
        return self.simulate(theta, batch_size, prng)

    def simulate(self, theta, batch_size=None, prng=None):
        """
        Simulates from the simulator and computes summary statistics for given parameter samples.
        Returns jnp.array with shape: [n_parameter_samples, n_summary_stats].

        Args:
            theta (jnp.array): Array of shape [n_samples, n_params] containing the weather generator
                parameter values for each simulation run.
            batch_size (int, optional): Number of parameter sets to simulate per batch. If None,
                all parameter sets are simulated in a single batch.
            prng (int or PRNGKey, optional): Random seed or JAX PRNGKey for simulation. If None,
                the object's default PRNG key is used.
        Returns:
            jnp.array: summary statistics from simulations, shape: [n_parameter_samples, n_summary_stats].
        """
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
                xs.append(self.summarizer(**self.simulate_ts_func(theta[lo:hi], prng)))
            x = jnp.concat(xs, axis=0)
        else:
            x = self.summarizer(**self.simulate_ts_func(theta, prng))
        return x

    def simulate_ts(self, theta, observables=None, batch_size=None, prng=None):
        """
        Simulates time series data from the simulator for given parameter samples.

        Args:
            theta (jnp.array): Array of shape [n_samples, n_params] containing the weather generator
                parameter values for each simulation run.
            observables (list of str, optional): List of observable names to return from the
                simulation output. If None, all available observables are returned.
            batch_size (int, optional): Number of parameter sets to simulate per batch. If None,
                all parameter sets are simulated in a single batch.
            prng (int or PRNGKey, optional): Random seed or JAX PRNGKey for simulation. If None,
                the object's default PRNG key is used.

        Returns:
            dict[str, jnp.array]: Dictionary mapping each observable name to its corresponding
            simulated data array. Each array has shape [n_samples, n_timesteps, 1].
        """
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
                if observables is not None:
                    xs.append(
                        {
                            key: val
                            for key, val in self.simulate_ts_func(theta[lo:hi], prng).items()
                            if key in observables
                        }
                    )
                else:
                    xs.append(**self.simulate_ts_func(theta[lo:hi], prng))
            x = {key: jnp.concat([d[key] for d in x], axis=0) for key in x[0].keys()}
        else:
            x = self.simulate_ts_func(theta, prng)
            if observables is not None:
                x = {key: val for key, val in x.items() if key in observables}
        return x


class SBIResults:
    def __init__(
        self,
        simulator: BatchSimulator,
        sbi_prior,
        sbi_posterior,
        calibration_posterior,
        summary_target,
        parameter_samples: dict,
        simulations: dict,
    ):
        self.simulator = simulator
        self.sbi_prior = sbi_prior
        self.sbi_posterior = sbi_posterior
        self.calibration_posterior = calibration_posterior
        self.summary_target = summary_target
        self.parameter_samples = parameter_samples
        self.simulations = simulations

    def with_target(
        self, summary_target, num_samples=None, simulation_batch_size=None, rng_seed=1234, map_kwargs=dict()
    ):
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
        if num_samples is None:
            num_samples = self.parameter_samples["sbi_posterior"].shape[0]
        simulation_batch_size = num_samples if simulation_batch_size is None else simulation_batch_size
        (theta_post, x_post), (theta_map, x_map) = _simulate_from_sbi_posterior(
            self.simulator,
            self.sbi_posterior,
            summary_target,
            return_ts=False,
            num_samples=num_samples,
            simulation_batch_size=simulation_batch_size,
            rng_seed=rng_seed,
            map_kwargs=map_kwargs,
        )
        parameter_samples = self.parameter_samples.copy()
        simulations = self.simulations.copy()
        parameter_samples["sbi_posterior"] = theta_post
        simulations["sbi_posterior"] = x_post
        parameter_samples["sbi_posterior_map"] = theta_map
        simulations["sbi_posterior_map"] = x_map
        return SBIResults(
            self.simulator,
            self.sbi_prior,
            self.sbi_posterior,
            self.calibration_posterior,
            summary_target,
            parameter_samples,
            simulations,
        )

    def simulate_ts(
        self, *which, observables=None, from_parameter_samples=True, batch_size=None, rng_seed=1234, num_samples=None
    ):
        """Simulates time series from the SBI results. Uses the sampled parameters, except if from_parameter_samples = False.

        Args:
            which (list, optional): from which parameter samples to simulate the time series.
                If not specified then all are used.
            observables (list of str, optional): List of observable names to return from the
                simulation output. If None, all available observables are returned.
            from_parameter_samples (bool): Whether to sample from stored parameter sets or
                sample newly from distributions.
            batch_size (int, optional): Number of parameter sets to simulate per batch. If None,
                all parameter sets are simulated in a single batch.
            rng_seed (int or PRNGKey, optional): Random seed or JAX PRNGKey for simulation.
            num_samples (int, optional): Only used if from_parameter_samples = False. Determines
                sample size of new simulations.
        Returns:
            dict[str, jnp.array]: Dictionary mapping each parameter sample type ("sbi_prior", "sbi_posterior", etc.)
            and each observable name to its corresponding simulated data array. Each array has shape [n_samples, n_timesteps, 1].
        """
        prng = jax.random.PRNGKey(rng_seed)

        if from_parameter_samples:
            if len(which) == 0:
                return {
                    key: self.simulator.simulate_ts(val, observables=observables, batch_size=batch_size, prng=prng)
                    for key, val in self.parameter_samples.items()
                }
            else:
                return {
                    key: self.simulator.simulate_ts(val, observables=observables, batch_size=batch_size, prng=prng)
                    for key, val in self.parameter_samples.items()
                    if key in which
                }
        else:
            """If which is empty (all are targets), we can make use of the resample function, else we need to sample from the individual model components."""
            if len(which) == 0:
                temp_resampled_sbi_results = self.resample(inplace=False, num_samples=num_samples)

                return {
                    key: self.simulator.simulate_ts(val, observables=observables, batch_size=batch_size, prng=prng)
                    for key, val in temp_resampled_sbi_results.parameter_samples.items()
                }
            else:
                return_dict = {}
                if "sbi_prior" in which:
                    _, ts, _ = self.simulate_from_sbi_prior(
                        return_ts=True,
                        num_samples=num_samples,
                        simulation_batch_size=batch_size,
                        rng_seed=rng_seed,
                    )

                    return_dict["sbi_prior"] = ts

                if "sbi_posterior" in which or "sbi_posterior_map" in which:
                    (_, ts_post, _), (_, ts_map, _) = self.simulate_from_sbi_posterior(
                        return_ts=True,
                        num_samples=num_samples,
                        simulation_batch_size=batch_size,
                        rng_seed=rng_seed,
                    )

                    if "sbi_posterior" in which:
                        return_dict["sbi_posterior"] = ts_post
                    if "sbi_posterior_map" in which:
                        return_dict["sbi_posterior_map"] = ts_map

                if "calibration_posterior" in which or "calibration_posterior_mean" in which:
                    (_, ts_cm, _), (_, ts_cp, _) = self.simulate_from_calibration_posterior(
                        return_ts=True,
                        num_samples=num_samples,
                        simulation_batch_size=batch_size,
                        rng_seed=rng_seed,
                    )

                    if "calibration_posterior" in which:
                        return_dict["calibration_posterior"] = ts_cm
                    if "calibration_posterior_mean" in which:
                        return_dict["calibration_posterior_mean"] = ts_cp

                return return_dict

    def resample(self, inplace=False, num_samples=None, simulation_batch_size=None, rng_seed=1234, map_kwargs=dict()):
        """Samples to regenerate parameter_samples and simulations in the object.

        Args:
            inplace (bool): Whether to replace in memory or to return a new oject.
            num_samples (int): Number of samples to generate.
            simulation_batch_size (int, optional). Simulation batch size.
            rng_seed (int, optional): Random seed, Defaults to 1234.
            map_kwargs (_type_, optional): _description_. Defaults to dict().
        Returns:
            dict[str, jnp.array]: Dictionary mapping each parameter sample type ("sbi_prior", "sbi_posterior", etc.)
            and each observable name to its corresponding simulated data array. Each array has shape [n_samples, n_timesteps, 1].
        """

        if num_samples is None:
            num_samples = self.parameter_samples["sbi_posterior"].shape[0]
        simulation_batch_size = num_samples if simulation_batch_size is None else simulation_batch_size

        parameter_samples = self.parameter_samples.copy()
        simulations = self.simulations.copy()

        # Generate calibration posterior samples/simulations
        if self.calibration_posterior is not None:
            (theta_cm, x_cm), (theta_cp, x_cp) = self.simulate_from_calibration_posterior(
                return_ts=False, num_samples=num_samples, simulation_batch_size=simulation_batch_size, rng_seed=rng_seed
            )
            # store results in dict(s)
            parameter_samples["calibration_posterior_mean"] = theta_cm
            simulations["calibration_posterior_mean"] = x_cm
            parameter_samples["calibration_posterior"] = theta_cp
            simulations["calibration_posterior"] = x_cp

        # Generate prior samples/simulations
        theta_prior, x_prior = self.simulate_from_sbi_prior(
            return_ts=False,
            num_samples=num_samples,
            simulation_batch_size=simulation_batch_size,
            rng_seed=rng_seed,
            map_kwargs=map_kwargs,
        )
        parameter_samples["sbi_prior"] = theta_prior
        simulations["sbi_prior"] = x_prior

        # Generate posterior samples / simulations
        (theta_post, x_post), (theta_map, x_map) = _simulate_from_sbi_posterior(
            self.simulator,
            self.sbi_posterior,
            self.summary_target,
            return_ts=False,
            num_samples=num_samples,
            simulation_batch_size=simulation_batch_size,
            rng_seed=rng_seed,
            map_kwargs=map_kwargs,
        )

        parameter_samples["sbi_posterior"] = theta_post
        simulations["sbi_posterior"] = x_post
        parameter_samples["sbi_posterior_map"] = theta_map
        simulations["sbi_posterior_map"] = x_map

        if inplace:
            self.parameter_samples = parameter_samples
            self.simulations = simulations
            return None
        else:
            return SBIResults(
                self.simulator,
                self.sbi_prior,
                self.sbi_posterior,
                self.calibration_posterior,
                self.summary_target,
                parameter_samples,
                simulations,
            )

    def simulate_from_calibration_posterior(
        self, return_ts=False, num_samples=1000, simulation_batch_size=None, rng_seed=1234
    ):
        """Convenience method for sampling from the given calibration posterior and corresponding calibration posterior predictive.

        Args:
            return_ts (bool): Whether to return time series, next to the parameter and simulation samples.
            num_samples (int): Number of samples to generate.
            simulation_batch_size (int, optional). Simulation batch size.
            rng_seed (int, optional): Random seed, Defaults to 1234.
            map_kwargs (_type_, optional): _description_. Defaults to dict().

        Returns:
            tuple: Tuple (sampled parameters, simulation_results) or tuple (sampled_parameters, timeseries, simulation_results) if return_ts = True.
        """
        # PRNG
        prng = jax.random.PRNGKey(rng_seed)

        if self.calibration_posterior is not None:
            # Posterior mean
            theta_cm = self.calibration_posterior.mean.reshape((1, -1)) * jnp.ones((num_samples, 1))
            # Full calibration posterior
            logger.info(f"Running {num_samples} simulations from calibration posterior")
            theta_cp = self.calibration_posterior.sample(prng, (num_samples,))

            if return_ts:
                ts_cm = self.simulator.simulate_ts(theta_cm, batch_size=simulation_batch_size, prng=prng)
                x_cm = self.simulator.summarizer(**ts_cm)

                ts_cp = self.simulator.simulate_ts(theta_cp, batch_size=simulation_batch_size, prng=prng)
                x_cp = self.simulator.summarizer(**ts_cp)

                return (theta_cm, ts_cm, x_cm), (theta_cp, ts_cp, x_cp)

            else:
                x_cm = self.simulator(theta_cm, batch_size=simulation_batch_size, prng=prng)
                x_cp = self.simulator(theta_cp, batch_size=simulation_batch_size, prng=prng)

                return (theta_cm, x_cm), (theta_cp, x_cp)
        else:
            raise ValueError("self.calibration_posterior is None")

    def simulate_from_sbi_prior(
        self,
        return_ts=False,
        num_samples=1000,
        simulation_batch_size=None,
        rng_seed=1234,
        map_kwargs=dict(),
    ):
        """Convenience method for sampling from the given SBI prior and corresponding prior predictive.

        Args:
            return_ts (bool): Whether to return time series, next to the parameter and simulation samples.
            num_samples (int): Number of samples to generate.
            simulation_batch_size (int, optional). Simulation batch size.
            rng_seed (int, optional): Random seed, Defaults to 1234.
            map_kwargs (_type_, optional): _description_. Defaults to dict().

        Returns:
            tuple: Tuple (sampled parameters, simulation_results) or tuple (sampled_parameters, timeseries, simulation_results) if return_ts = True.
        """
        prng = jax.random.PRNGKey(rng_seed)
        logger.info(f"Running {num_samples} simulations for SBI prior")
        theta_prior = self.sbi_prior.sample(prng, (num_samples,))
        if return_ts:
            ts_prior = self.simulator.simulate_ts(theta_prior, batch_size=simulation_batch_size, prng=prng)
            x_prior = self.simulator.summarizer(**ts_prior)
        else:
            x_prior = self.simulator(theta_prior, batch_size=simulation_batch_size, prng=prng)

        if return_ts:
            return theta_prior, ts_prior, x_prior
        else:
            return theta_prior, x_prior

    def simulate_from_sbi_posterior(
        self,
        summary_target=None,
        return_ts=False,
        num_samples=1000,
        simulation_batch_size=None,
        rng_seed=1234,
        map_kwargs=dict(),
    ):
        """Convenience method for sampling from the given SBI posterior and corresponding posterior predictive.

        Args:
            summary_target (jax.Array): Target summary statistics for the posterior. If none, then self.summary_target is taken.
            return_ts (bool): Whether to return time series, next to the parameter and simulation samples.
            num_samples (int): Number of samples to generate.
            simulation_batch_size (int, optional). Simulation batch size.
            rng_seed (int, optional): Random seed, Defaults to 1234.
            map_kwargs (_type_, optional): _description_. Defaults to dict().

        Returns:
            tuple: Tuple (sampled parameters, simulation_results) or tuple (sampled_parameters, timeseries, simulation_results) if return_ts = True.
        """
        # reset torch RNG seed
        torch.manual_seed(rng_seed)

        if summary_target is None:
            summary_target = self.summary_target

        return _simulate_from_sbi_posterior(
            simulator=self.simulator,
            sbi_posterior=self.sbi_posterior,
            summary_target=summary_target,
            return_ts=return_ts,
            num_samples=num_samples,
            simulation_batch_size=simulation_batch_size,
            rng_seed=rng_seed,
            map_kwargs=map_kwargs,
        )

    def to_file(self, filename):
        import dill

        with open(filename, "wb") as outp:
            dill.dump(self, outp)

    @classmethod
    def from_file(cls, filename):
        import dill

        with open(filename, "rb") as inp:
            obj = dill.load(inp)
        return obj


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

    def simulate_ts(_theta, prng=default_prng):
        theta = _theta if len(_theta.shape) > 1 else _theta.reshape((1, -1))
        # batch_size = theta.shape[0]
        params = {k: v for k, v in prior.constrain(theta, as_dict=True).items()}
        # run simulations using numpyro predictive
        predictive = Predictive(model.simulate, posterior_samples=params, parallel=parallel)
        preds = predictive(prng)
        # splat predicted variables into summary stats function
        return preds

    return BatchSimulator(simulate_ts, summarizer, default_prng)


def run_sbi(
    simulator: BatchSimulator,
    summary_target: jax.Array,
    prior: dist.Distribution = None,
    calibration_posterior: dist.Distribution = None,
    num_samples: int = 1000,
    num_rounds: int = 1,
    simulations_per_round: int = 1000,
    simulation_batch_size: int = None,
    sbi_alg=SNPE,
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
    parameter_samples = dict()
    simulations = dict()

    if calibration_posterior is not None:
        # Posterior mean
        logger.info(f"Running {num_samples} simulations for calibration posterior mean")
        theta_cm = calibration_posterior.mean.reshape((1, -1)) * jnp.ones((num_samples, 1))
        x_cm = simulator(theta_cm, batch_size=simulation_batch_size, prng=prng)
        # Full calibration posterior
        logger.info(f"Running {num_samples} simulations for full calibration posterior")
        theta_cp = calibration_posterior.sample(prng, (num_samples,))
        x_cp = simulator(theta_cp, batch_size=simulation_batch_size, prng=prng)
        # store results in dict(s)
        parameter_samples["calibration_posterior_mean"] = theta_cm
        simulations["calibration_posterior_mean"] = x_cm
        parameter_samples["calibration_posterior"] = theta_cp
        simulations["calibration_posterior"] = x_cp

    # Generate prior samples/simulations
    logger.info(f"Running {num_samples} simulations for SBI prior")
    prior = simulator.prior if prior is None else prior
    theta_prior = prior.sample(prng, (num_samples,))
    x_prior = simulator(theta_prior, batch_size=simulation_batch_size, prng=prng)
    parameter_samples["sbi_prior"] = theta_prior
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

    (theta_post, x_post), (theta_map, x_map) = _simulate_from_sbi_posterior(
        simulator,
        sbi_posterior,
        summary_target,
        return_ts=False,
        num_samples=num_samples,
        simulation_batch_size=simulation_batch_size,
        rng_seed=rng_seed,
        map_kwargs=map_kwargs,
    )
    parameter_samples["sbi_posterior"] = theta_post
    simulations["sbi_posterior"] = x_post
    parameter_samples["sbi_posterior_map"] = theta_map
    simulations["sbi_posterior_map"] = x_map

    return SBIResults(
        simulator, prior, sbi_posterior, calibration_posterior, summary_target, parameter_samples, simulations
    )


def _simulate_from_sbi_posterior(
    simulator,
    sbi_posterior,
    summary_target,
    return_ts=False,
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
        return_ts (bool): _description_
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
    if return_ts:
        ts_post = simulator.simulate_ts(theta_post, batch_size=simulation_batch_size, prng=prng)
        x_post = simulator.summarizer(**ts_post)
    else:
        x_post = simulator(theta_post, batch_size=simulation_batch_size, prng=prng)

    # obtain MAP estimate
    logger.info(f"Finding MAP estimate")
    theta_map = t2j(sbi_posterior.set_default_x(j2t(summary_target)).map(**map_kwargs))

    # broadcast to num_samples and run simulations
    logger.info(f"Running {num_samples} simulations for SBI posterior MAP estimate")
    if return_ts:
        ts_map = simulator.simulate_ts(
            theta_map.reshape((1, -1)) * jnp.ones((num_samples, 1)), batch_size=simulation_batch_size, prng=prng
        )
        x_map = simulator.summarizer(**ts_map)
    else:
        x_map = simulator(
            theta_map.reshape((1, -1)) * jnp.ones((num_samples, 1)), batch_size=simulation_batch_size, prng=prng
        )

    if return_ts:
        return (
            (theta_post, ts_post, x_post),
            (theta_map, ts_map, x_map),
        )
    else:
        return (theta_post, x_post), (theta_map, x_map)


def get_rescaled_svi_posterior(
    guide: AutoMultivariateNormal,
    svi_result,
    parameter_mask=None,
    scale_factor=2.0,
    diagonal=True,
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
        rescaled_cov = D * I + parameter_mask * (scale_factor**2 - 1) * I * D
    else:
        rescaled_cov = C + parameter_mask * (scale_factor**2 - 1) * I * D
    return dist.MultivariateNormal(fitted_posterior.mean, rescaled_cov)
