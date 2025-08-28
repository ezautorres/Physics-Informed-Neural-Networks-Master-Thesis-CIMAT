"""
mcmc_inference.py
-----------------
Bayesian Uncertainty Quantification (BUQ) with MCMC Sampling for PINN-Based
Models.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Implements Markov Chain Monte Carlo (MCMC) inference with Bayesian Uncertainty
Quantification (BUQ) for PDE-based forward models, such as Physics-Informed
Neural Networks (PINNs). The module provides both a class-based API (`MCMC`)
and a high-level function (`MCMCInference`) to run inference, save samples,
and compute posterior summaries.

The algorithm leverages the BUQ t-walk sampler to explore posterior parameter
distributions given observational data and a forward model. Support is included
for synthetic data generation, Gaussian likelihoods, CSV export of posterior
samples, and reproducible statistical summaries.

Key Components
--------------
- **Class `MCMC`**
  - Encapsulates the BUQ sampler for parameter inference.
  - Handles synthetic data generation, chain execution, and CSV storage.
  - Stores execution time and posterior samples after burn-in.

- **Function `MCMCInference`**
  - High-level interface for running or loading MCMC inference.
  - Checks for existing CSV files, otherwise launches a new chain.
  - Returns posterior samples and summary statistics.

- **Function `define_forward_map`**
  - Utility to evaluate the forward operator using either a trained PINN or
    the analytical solution, with automatic parameter handling.

Mathematical Model
------------------
Likelihood (Gaussian noise model):
$$
    p(u \mid \theta) \propto 
    \exp\!\left(-\tfrac{1}{2\sigma^{2}}\|F(\theta) - u\|^{2}\right),
$$
where $F(\theta)$ is the forward operator, $\theta$ the parameters, and $u$
the observed data.

Usage
-----
To run MCMC inference and save posterior samples:
>>> samples, stats = MCMCInference(
...     filename="posterior_samples.csv",
...     forward_map=forward_map,
...     data_x=data_x,
...     data_u=data_u,
...     par_names=["rho", "R"],
...     par_prior=[prior_rho, prior_R],
...     par_supp=[supp_rho, supp_R],
...     par_true=[3.2, 0.85],
...     sigma=1e-3,
...     n_iter=100000,
...     burn_in=10000,
...     SimData=True
... )

To evaluate solutions with a trained PINN or analytical function:
>>> y_pred = define_forward_map(
...     theta=[3.2, 0.85], t=np.linspace(0,1,100), pinn_instance=pinn
... )

Notes
-----
    - Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
    - Samples are exported as CSV for analysis in Python, R, or Julia.
    - The posterior summary includes means, variances, and credible intervals.
    - The t-walk BUQ algorithm ensures robustness for multimodal posterior landscapes.
"""
# Necessary libraries.
import os                                          # File paths.
import sys                                         # System functions.
import random                                      # Random numbers.
from typing import Callable                        # Type hints.
import numpy as np                                 # Arrays and math.
import torch                                       # Tensors and autograd.
from pytwalk import BUQ                            # Bayesian sampling (t-walk).
import time                                        # Time library.
from typing import Sequence                        # Type hints.
import pandas as pd                                # Data handling.
import scipy.stats as stats                        # Statistical functions.
np.set_printoptions(precision=17, suppress=False)  # NumPy printing precision.
np.random.seed(0)                                  # NumPy random seed.
random.seed(0)                                     # Python random seed.
torch.manual_seed(0)                               # PyTorch random seed.
torch.backends.cudnn.benchmark = False             # Disable CuDNN auto-tuner.
device = torch.device(                             # Select GPU if available.
    "cuda" if torch.cuda.is_available() else "cpu"
)  

# Project root and utils.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from utils import load_samples_from_csv, summarize_results  # Data I/O and summaries.

class MCMC:
    def __init__(
        self,
        forward_map: Callable,
        data_x: np.ndarray,
        data_u: np.ndarray,
        par_names: Sequence[str],
        par_prior: Sequence[Callable],
        par_supp: Sequence[Callable],
        par_true: Sequence[float],
        sigma: float = 1e-03,
        n_iter: int = 100_000,
        burn_in: int = 10_000,
        SimData: bool = True
    ):
        """
        Markov Chain Monte Carlo (MCMC) inference with Bayesian Uncertainty
        Quantification (BUQ).

        This class implements an MCMC sampler using the BUQ algorithm for 
        parameter inference in PDE-based forward models. It takes as input a 
        forward map, observed data, parameter priors, and supports generating 
        synthetic data when the ground-truth parameters are available. The 
        sampler runs a Metropolis-Hastings-like chain and stores posterior 
        samples for subsequent analysis.

        Parameters
        ----------
        forward_map : Callable
            Forward operator F(\theta) that maps parameters \theta to predicted
            data u.
        data_x : np.ndarray
            Independent variable values (e.g., spatial or temporal grid points).
        data_u : np.ndarray
            Observed data corresponding to `data_x`.
        par_names : Sequence[str]
            Names of the model parameters to be inferred.
        par_prior : Sequence[Callable]
            List of prior distribution functions, one per parameter.
        par_supp : Sequence[Callable]
            List of support functions (indicator functions) for parameter domains.
        par_true : Sequence[float]
            True parameter values (used for data simulation when `SimData=True`).
        sigma : float, default=1e-03
            Standard deviation of the Gaussian likelihood (measurement noise).
        n_iter : int, default=100_000
            Total number of MCMC iterations.
        burn_in : int, default=10_000
            Number of initial samples to discard (burn-in phase).
        SimData : bool, default=True
            If True, generate synthetic data using `par_true` before running MCMC.

        Attributes
        ----------
        buq : BUQ
            Instance of the BUQ sampler initialized with the given forward map
            and priors.
        execution_time : float
            Total runtime (in seconds) of the MCMC chain execution.
        samples : np.ndarray
            Array of posterior samples after burn-in, excluding auxiliary columns.
        """
        self.forward_map = forward_map
        self.data_x = data_x
        self.data_u = data_u
        self.par_names = par_names
        self.par_prior = par_prior
        self.par_supp = par_supp
        self.par_true = par_true
        self.sigma = sigma
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.SimData = SimData

    def run_mcmc(self):
        """
        Execute the MCMC inference routine. This method initializes a BUQ
        sampler with the forward map, prior distributions, and Gaussian
        likelihood model. If `SimData=True`, synthetic observations are
        generated from the ground-truth parameters before sampling. The
        sampler then runs a Markov Chain Monte Carlo routine with the
        specified number of iterations and burn-in period.

        The results, including execution time, are stored internally in the
        `buq` object and as the attribute `execution_time`.

        Notes
        -----
        - The likelihood is modeled as a Gaussian:
          .. math::
             p(u \mid \theta) \propto \exp\left(-\frac{1}{2\sigma^2}
             \|F(\theta) - u\|^2 \right).
        - The posterior distribution is sampled using a BUQ-based MCMC 
          scheme with proposal adaptation.
        - The chain includes the burn-in samples, but these are 
          typically discarded when analyzing results.

        See Also
        --------
        save_samples_to_csv : To store posterior samples on disk.
        """
        # Initialize BUQ sampler.
        self.buq = BUQ(
            q=len(self.par_names),
            F=self.forward_map,
            data=self.data_u,
            logdensity=stats.norm.logpdf,
            sigma=self.sigma,
            t=self.data_x,
            par_names=self.par_names,
            par_prior=self.par_prior,
            par_supp=self.par_supp,
            simdata=lambda n, loc, scale: stats.norm.rvs(
                size=n[0], loc=loc, scale=scale
            )
        )

        # Simulate data if required.
        if self.SimData:
            self.buq.SimData(x=np.array(self.par_true))

        # Run MCMC.
        start_time = time.time()
        self.buq.RunMCMC(T=self.n_iter, burn_in=self.burn_in)
        self.execution_time = time.time() - start_time

    def save_samples_to_csv(self, filename: str):
        """
        Save posterior samples to a CSV file. Extracts the MCMC samples
        (excluding the auxiliary column) after burn-in and stores them in
        a CSV file with parameter values and the total execution time. The
        file is written to the directory of the running script.

        Parameters
        ----------
        filename : str
            Name of the output CSV file where posterior samples will be saved.

        Output Format
        -------------
        - Each column corresponds to one inferred parameter 
          (named as `param_0`, `param_1`, ...).
        - An additional column `execution_time` stores the runtime (in seconds)
          of the full MCMC routine, replicated across all rows for convenience.

        Notes
        -----
        - Samples are taken from `self.buq.Output[self.burn_in:, :-1]`.
        - The resulting file can be directly imported into statistical analysis
          software (e.g., R, Python, Julia) for posterior analysis and visualization.
        """
        # Get the directory of the running script.
        script_dir = os.path.dirname(
            os.path.abspath(sys.modules["__main__"].__file__)
        )
        path = os.path.join(script_dir, filename)

        # Create DataFrame for samples.
        self.samples = self.buq.Output[self.burn_in:, :-1]
        columns = [
            f"param_{i}" for i in range(self.samples.shape[1])
        ] + ["execution_time"]

        # Add execution time (in seconds).
        df = pd.DataFrame(
            np.column_stack([
                self.samples,
                np.full((self.samples.shape[0], 1), self.execution_time)
            ]),
            columns=columns
        )
        df.to_csv(path, index=False)

def MCMCInference(
    filename: str,
    forward_map: Callable,
    data_x: np.ndarray,
    data_u: np.ndarray,
    par_names: Sequence[str],
    par_prior: Sequence[Callable],
    par_supp: Sequence[Callable],
    par_true: Sequence[float],
    sigma: float = 1e-03,
    n_iter: int = 100_000,
    burn_in: int = 10_000,
    SimData: bool = True
) -> tuple[dict, dict]:
    """
    Perform MCMC inference with Bayesian Uncertainty Quantification (BUQ).

    This function provides a high-level interface for running MCMC inference 
    on PDE-based models. It checks if posterior samples already exist in the 
    given CSV file; if so, it loads them, otherwise it initializes an `MCMC`
    sampler, runs the chain, saves samples, and computes posterior statistics.

    Parameters
    ----------
    filename : str
        Name of the CSV file used to store or load posterior samples.
    forward_map : Callable
        Forward operator F(θ) mapping parameters θ to predicted data u.
    data_x : np.ndarray
        Independent variable values (e.g., spatial or temporal grid points).
    data_u : np.ndarray
        Observed data corresponding to `data_x`.
    par_names : Sequence[str]
        Names of the model parameters to be inferred.
    par_prior : Sequence[Callable]
        List of prior distribution functions for each parameter.
    par_supp : Sequence[Callable]
        List of support (indicator) functions for parameter domains.
    par_true : Sequence[float]
        True parameter values (used for data simulation when `SimData=True`).
    sigma : float, default=1e-3
        Standard deviation of the Gaussian likelihood.
    n_iter : int, default=100_000
        Total number of MCMC iterations.
    burn_in : int, default=10_000
        Number of initial samples to discard.
    SimData : bool, default=True
        If True, synthetic data are generated from `par_true` prior to sampling.

    Returns
    -------
    samples : dict
        Dictionary containing posterior samples and execution time:
        - `"samples"` : np.ndarray of posterior draws.
        - `"execution_time"` : float with runtime in seconds.
    stats : dict
        Dictionary with summarized posterior statistics (e.g., means, credible
        intervals).

    Notes
    -----
    - If `filename` exists, stored samples are loaded instead of running MCMC.
    - Otherwise, a new chain is generated with the `MCMC` class and saved.
    - Posterior statistics are computed via `summarize_results`.

    See Also
    --------
    MCMC : Low-level class for running MCMC chains with BUQ.
    summarize_results : Utility function for posterior summaries.
    """
    # Check if samples already exist.
    if os.path.exists(filename):
        samples = load_samples_from_csv(filename)
    else:
        # Run MCMC inference.
        twalk = MCMC(
            forward_map=forward_map,
            data_x=data_x,
            data_u=data_u,
            par_names=par_names,
            par_prior=par_prior,
            par_supp=par_supp,
            par_true=par_true,
            sigma=sigma,
            n_iter=n_iter,
            burn_in=burn_in,
            SimData=SimData
        )
        twalk.run_mcmc()
        twalk.save_samples_to_csv(filename=filename)
        samples = {
            "samples": twalk.samples,
            "execution_time": twalk.execution_time,
        }

    # Print Summarized results.
    stats = summarize_results(samples=samples, par_true=par_true)

    return samples, stats

def define_forward_map(
    theta: np.ndarray,
    t: np.ndarray,
    pinn_instance: Callable,
    analytic: bool = False
) -> np.ndarray:
    """
    Evaluates either the PINN-predicted solution or the analytical solution
    for a given set of base input coordinates and physical parameters.

    Parameters
    ----------
    theta : np.ndarray
        Array of shape (P,) containing the parameter values to be appended to
        each input row in the same order they appear.
    t : np.ndarray
        Array of shape (N,) or (N, D) containing the base input coordinates
        (e.g., spatial position, time, etc.) without parameters.
    pinn_instance : object
        Trained PINN model instance, which must implement the methods
        `.pinn(torch.Tensor)` and `.analytical_solution(torch.Tensor)`.
    analytic : bool, optional
        If True, evaluates the analytical solution; if False, evaluates the
        PINN-predicted solution. Default is False.

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing the evaluated solution.
    """    
    # Convert inputs to appropriate types.
    t = np.asarray(t, dtype=np.float32)
    t = t.reshape(-1, 1) if t.ndim == 1 else t

    # Reshape theta to ensure it is a 2D array with one row.
    theta = np.asarray(theta, dtype=np.float32).reshape(1, -1)
    theta_cols = np.repeat(theta, t.shape[0], axis=0)

    # Create the input tensor for the forward map and evaluate the solution.
    X = np.column_stack((t, theta_cols))
    eval_fn = pinn_instance.analytical_solution if analytic else pinn_instance.pinn

    return eval_fn(torch.tensor(X)).detach().cpu().numpy().reshape(-1)