import torch                                                                       # Import PyTorch.         
import numpy as np                                                                 # Import NumPy.                 
import pandas as pd                                                                # Import pandas for data handling.           
import random                                                                      # Import random module.   
import sys, os                                                                     # Import sys and os modules.
import time                                                                        # Import time module for timing.
import scipy.stats as stats                                                        # Import scipy.stats for statistical functions.
from pytwalk import BUQ                                                            # Import the BUQ class from the pytwalk package.
from typing import Callable                                                        # For type hinting.
np.random.seed(0)                                                                  # For reproducibility.
random.seed(0)                                                                     # For reproducibility.
torch.manual_seed(0)                                                               # For reproducibility.
torch.backends.cudnn.benchmark = False                                             # For reproducibility.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # Add the parent directory to the path.
from utils import load_samples_from_csv, summarize_results                         # Import utility functions.

class MCMC:
    def __init__(
            self, forward_map: Callable, data_x: np.ndarray, data_u: np.ndarray, par_names: str,
            par_prior: Callable, par_supp: tuple, par_true: float, sigma: float = 0.01, 
            n_iter: int = 100000, burn_in: int = 10000,):
        """
        Initialize the MCMC inference class.

        Parameters
        ----------
        forward_map : Callable
            Forward map to simulate data.
        data_x : np.ndarray
            Input locations.
        data_u : np.ndarray
            Observed data.
        par_names : list
            Names of the parameters to infer.
        par_prior : list
            List of prior distributions for the parameters.
        par_supp : list
            List of support functions for the parameters.
        par_true : float
            True parameter value (for computing simulated data).
        sigma : float
            Noise standard deviation.
        n_iter : int
            Total number of MCMC iterations.
        burn_in : int
            Number of burn-in iterations.   
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

    def run_mcmc(self):
        """
        Run the MCMC inference using the BUQ algorithm.
        """
        self.buq = BUQ(
            q          = len(self.par_names),
            F          = self.forward_map,
            data       = self.data_u,
            logdensity = stats.norm.logpdf,
            sigma      = self.sigma,
            t          = self.data_x,
            par_names  = self.par_names,
            par_prior  = self.par_prior,
            par_supp   = self.par_supp,
            simdata    = lambda n, loc, scale: stats.norm.rvs(size = n[0], loc = loc, scale = scale)
        )

        self.buq.SimData(x = np.array([self.par_true]))
        
        start_time = time.time()
        self.buq.RunMCMC(T = self.n_iter, burn_in = self.burn_in)
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
        script_dir = os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__))
        path = os.path.join(script_dir, filename)

        self.samples = self.buq.Output[self.burn_in:, :-1]
        columns = [f"param_{i}" for i in range(self.samples.shape[1])] + ["execution_time"]

        # Add execution time (in seconds)
        df = pd.DataFrame(
            np.column_stack((self.samples, np.full((self.samples.shape[0], 1), self.execution_time))),
            columns = columns
        )

        df.to_csv(path, index = False)

def MCMCInference(
        filename: str, forward_map: Callable, data_x: np.ndarray, data_u: np.ndarray, par_names: str,
        par_prior: Callable, par_supp: tuple, par_true: float, sigma: float = 0.01, n_iter: int = 100000,
        burn_in: int = 10000,
):
    """
    Run MCMC inference or load samples from file if available.

    Parameters
    ----------
    filename : str
        Path to the CSV file where samples are saved/loaded.
    forward_map : Callable
        Forward map to simulate data.
    data_x : np.ndarray
        Input locations.
    data_u : np.ndarray
        Observed data.
    par_names : list
        Names of the parameters to infer.
    par_prior : list
        List of prior distributions for the parameters.
    par_supp : list
        List of support functions for the parameters.
    par_true : float
        True parameter value (for computing simulated data).
    sigma : float
        Noise standard deviation.
    n_iter : int
        Total number of MCMC iterations.
    burn_in : int
        Number of burn-in iterations.

    Returns
    -------
    samples : np.ndarray
        The MCMC samples (after burn-in).
    stats : dict
        Dictionary with summary statistics.
    """
    if os.path.exists(filename):
        samples = load_samples_from_csv(filename)
    else:
        twalk = MCMC(
            forward_map = forward_map,
            data_x      = data_x,
            data_u      = data_u,
            par_names   = par_names,
            par_prior   = par_prior,
            par_supp    = par_supp,
            par_true    = par_true,
            sigma       = sigma,
            n_iter      = n_iter,
            burn_in     = burn_in,
        )
        twalk.run_mcmc()
        twalk.save_samples_to_csv(filename = filename)
        samples = twalk.samples

    stats = summarize_results(samples = samples, par_true = par_true)

    return samples, stats
    