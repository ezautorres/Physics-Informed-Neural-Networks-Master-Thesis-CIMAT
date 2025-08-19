# Necessary libraries.
import os                                # File paths.
import sys                               # System functions.
import time                              # Time utilities.
import random                            # Random numbers.
from typing import Callable, Sequence    # Type hints.
import numpy as np                       # Arrays and math.
import pandas as pd                      # Data handling.
import torch                             # Tensors and autograd.
import scipy.stats as stats              # Probability and stats.
from pytwalk import BUQ                  # Bayesian sampling (t-walk).
np.random.seed(0)                        # NumPy random seed.
random.seed(0)                           # Python random seed.
torch.manual_seed(0)                     # PyTorch random seed.
torch.backends.cudnn.benchmark = False   # Disable CuDNN auto-tuner.

from utils import load_samples_from_csv, summarize_results  # Data I/O and summaries.
sys.path.append(                                            # Add project root to path.
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)                      

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
        sigma: float = 1e-3,
        n_iter: int = 100_000,
        burn_in: int = 10_000,
        SimData: bool = True
    ):
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
        Run the MCMC inference using the BUQ algorithm.
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
        Save MCMC samples to a CSV file in the directory of the running script,
        including execution time in the last row.

        Parameters
        ----------
        filename : str
            The name of the file to save the samples to.
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
            np.column_stack(
                self.samples,
                np.full((self.samples.shape[0], 1), self.execution_time)
            ),
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
    sigma: float = 1e-3,
    n_iter: int = 100_000,
    burn_in: int = 10_000,
    SimData: bool = True
) -> tuple[dict, dict]:
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
    Evaluates either the PINN-predicted solution or the analytical solution for
    a given set of base input coordinates and physical parameters.

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
    t = np.asarray(t, dtype = np.float32)     
    t = t.reshape(-1,1) if t.ndim == 1 else t 
    
    # Reshape theta to ensure it is a 2D array with one row.
    theta = np.asarray(theta, dtype = np.float32).reshape(1, -1)
    theta_cols = np.repeat(theta, t.shape[0], axis = 0)

    # Create the input tensor for the forward map and evaluate the solution.
    X = np.column_stack((t, theta_cols))
    eval_fn = pinn_instance.analytical_solution if analytic else pinn_instance.pinn

    return eval_fn(torch.tensor(X)).detach().cpu().numpy().reshape(-1)