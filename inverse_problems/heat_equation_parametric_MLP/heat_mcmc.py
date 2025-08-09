"""
heat_run_mcmc.py
-------------------------------------
Bayesian parameter inference for the 1D Heat Equation using Physics-Informed Neural Networks (PINNs) 
and analytical solutions as forward models.

Author: Ezau Faridh Torres Torres.
Date: 7 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script performs Bayesian inference to estimate unknown physical parameters of the 1D Heat Equation,
leveraging both a trained Physics-Informed Neural Network (PINN) and the corresponding analytical solution 
as forward models. The problem setup assumes a parametric formulation where fixed parameters (e.g., mode 
number `n`) and unknown parameters (e.g., thermal diffusivity `⍺`) are incorporated into the forward map.

The workflow includes:
    - Loading a pre-trained parametric PINN model for the heat equation.
    - Defining the prior distribution and support for the parameters of interest.
    - Generating synthetic measurement data with optional Gaussian noise.
    - Defining forward maps for both analytical and PINN-based solutions.
    - Running Markov Chain Monte Carlo (MCMC) sampling using the t-walk algorithm to estimate posterior distributions.
    - Saving MCMC samples to CSV files for reproducibility.
    - Plotting joint posterior distributions for direct comparison between analytical and PINN-based inference.

Functions and Utilities
-----------------------
generate_synthetic_data :
    Generates noisy synthetic data points given spatial/temporal domains, fixed parameters, 
    true parameters, and a forward model.
define_forward_map :
    Wraps a PINN instance or analytical solution into a callable function for MCMC evaluation.
MCMCInference :
    Performs Bayesian sampling using the t-walk algorithm, returning posterior samples and statistics.
plot_joint_posteriors :
    Generates comparative posterior plots for two sets of MCMC results.

Usage
-----
>>> # Generate synthetic data from the analytical solution
>>> data_x, data_u_exact, data_u = generate_synthetic_data(
...     dim1_min=0, dim1_max=L, dim2_min=0, dim2_max=T, n_points=20,
...     pinn_instance=heat_pinn, fixed_params=[n], par_true=[par_true], sigma=0.01
... )
>>> # Define forward maps
>>> analytical_forward_map = lambda theta, t: define_forward_map(theta, t, pinn_instance=heat_pinn, analytic=True)
>>> pinn_forward_map = lambda theta, t: define_forward_map(theta, t, pinn_instance=heat_pinn, analytic=False)
>>> # Run MCMC inference
>>> samples_analytical, stats_analytical = MCMCInference(
...     filename="samples_analytical.csv", forward_map=analytical_forward_map,
...     data_x=data_x, data_u=data_u, par_names=[r"$\\alpha$"], par_prior=par_prior,
...     par_supp=par_supp, par_true=par_true, sigma=0.01, n_iter=100000, burn_in=10000
... )

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep 
    learning framework for solving forward and inverse problems involving nonlinear partial differential 
    equations. *Journal of Computational Physics*, 378, 686-707.
- Christen, J. A., & Fox, C. (2010). A general purpose sampling algorithm for continuous distributions 
    (the t-walk). *Bayesian Analysis*, 5(2), 263-281.
- https://github.com/carlosrtf/twalk
"""
import numpy as np                                                                 # Numpy library.
import torch                                                                       # Import PyTorch
import sys, os                                                                     # Import sys and os modules.
import random                                                                      # Random module for reproducibility.
import scipy.stats as stats                                                        # Import scipy.stats for statistical functions.
np.set_printoptions(precision = 17, suppress = False)                              # Set print options for numpy.
np.random.seed(0)                                                                  # Seed for reproducibility.
random.seed(0)                                                                     # Seed for reproducibility.
torch.manual_seed(0)                                                               # Seed for reproducibility.
torch.backends.cudnn.benchmark = False                                             # Reproducibility for CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              # Set device to GPU if available, else CPU.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # Add the parent directory to the path.
from inference.mcmc import MCMCInference                                           # Import MCMC inference class and utility function.
from inference.mcmc import define_forward_map                                      # Import forward map definition utility.
from utils import get_model_info, load_full_model                                  # Import utility functions.
from plotting import plot_joint_posteriors                                         # Import plotting function.
from sampling import generate_synthetic_data                                       # Import synthetic data generation utility.
from inverse_problems.heat_equation_parametric_MLP.heat_equation_parametric_MLP import HeatEquationPinn # Import the PINN class for heat equation parameter inference.

# ------------------------------------------------------------------------------------------------------
# Load the trained PINN model for parameter inference.
# ------------------------------------------------------------------------------------------------------
checkpoint_filename = "heat_parametric_MLP.pth"
heat_pinn = load_full_model(
    checkpoint_path = os.path.join("trained_models", checkpoint_filename),
    model_class     = HeatEquationPinn)
get_model_info(checkpoint_filename) # Print model information.

# ------------------------------------------------------------------------------------------------------
# Parameters for MCMC inference.
# ------------------------------------------------------------------------------------------------------ 
L, T, n   = 2.0, 2.0, 5               # Length of the domain in x, time, and fixed n value.
par_true  = 0.05                      # True value of the parameters to be inferred.
par_names = [r"$\alpha$"]             # Name of the parameters to be inferred.
par_prior = [stats.uniform(0, 0.1)]   # Prior distribution for the parameters.
par_supp  = [lambda a: 0 <= a <= 0.1] # Support function for the prior distributions.
sigma     = 0.01                      # Standard deviation for the noise in the data.
n_iter    = 500000                    # Number of MCMC iterations.
burn_in   = int(0.1 * n_iter)         # Burn-in period.

# ------------------------------------------------------------------------------------------------------
# Synthetic data generation and forward map definition.
# ------------------------------------------------------------------------------------------------------
n_points = 20  # Number of data points to generate.
data_x, data_u_exact, data_u = generate_synthetic_data(
    dim1_min = 0, dim1_max = L, dim2_min = 0, dim2_max = T, n_points = n_points, pinn_instance = heat_pinn,
    fixed_params = [n], par_true = [par_true], sigma = sigma
    )

# Define the forward maps for the analytical and PINN solutions.
analytical_forward_map = lambda theta, t: define_forward_map( # Analytical forward map.
    theta, t, pinn_instance = heat_pinn, analytic = True)
pinn_forward_map       = lambda theta, t: define_forward_map( # PINN forward map.
    theta, t, pinn_instance = heat_pinn, analytic = False)

# ------------------------------------------------------------------------------------------------------
# File paths for saving/loading MCMC samples.
# ------------------------------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__)) # Directory of the running script.
analytical_csv_path = os.path.join(script_dir, "samples_analytical.csv")        # CSV file for analytical samples.
pinn_csv_path = os.path.join(script_dir, "samples_pinn.csv")                    # CSV file for PINN samples.

# ------------------------------------------------------------------------------------------------------
# Run MCMC inference 
# ------------------------------------------------------------------------------------------------------
print("\n" + "─"*60 + "\nAnalytical Forward Map\n" + "─"*60)
samples_analytical, stats_analytical = MCMCInference(
    filename    = analytical_csv_path,
    forward_map = analytical_forward_map,
    data_x      = data_x,
    data_u      = data_u,
    par_names   = par_names,
    par_prior   = par_prior,
    par_supp    = par_supp,
    par_true    = par_true,
    sigma       = sigma,
    n_iter      = n_iter,
    burn_in     = burn_in,
    SimData     = False
)
print("\n" + "─"*60 + "\nPINN Forward Map\n" + "─"*60)
samples_pinn, stats_pinn = MCMCInference(
    filename    = pinn_csv_path,
    forward_map = pinn_forward_map,
    data_x      = data_x,
    data_u      = data_u,
    par_names   = par_names,
    par_prior   = par_prior,
    par_supp    = par_supp,
    par_true    = par_true,
    sigma       = sigma,
    n_iter      = n_iter,
    burn_in     = burn_in,
    SimData     = False
)

# ------------------------------------------------------------------------------------------------------
# Plot joint posterior distributions.
# ------------------------------------------------------------------------------------------------------
plot_joint_posteriors(
    samples1  = samples_analytical["samples"],
    samples2  = samples_pinn["samples"],
    par_true  = par_true,
    par_names = r"$\alpha$",
    bins      = 30,
    filename  = "posterior_comparison.pdf",
    param_idx = 0
)