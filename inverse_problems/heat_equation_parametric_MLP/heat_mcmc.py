"""
heat_mcmc.py
--------------------
MCMC-Based Bayesian Inference for the Parametric Heat Equation Using PINNs.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Performs Bayesian Uncertainty Quantification (BUQ) via Markov Chain Monte Carlo
(MCMC) for the 1D heat equation with thermal diffusivity $\alpha$ as the unknown
parameter. The forward model is provided by a trained Physics-Informed Neural
Network (PINN) and, for comparison, by the analytical solution.

The script generates synthetic noisy observations, defines both analytical and
PINN-based forward maps, runs MCMC inference using the BUQ sampler, and compares
posterior distributions of the parameter $\alpha$ between both models.

Governing PDE:
$$
    u_{t}(x,t) - \alpha u_{xx}(x,t) = 0, \quad (x,t)\in(0,L)\times(0,T).
$$

Boundary conditions:
$$
    u(0,t) = u(L,t) = 0, \quad t\in[0,T].
$$

Initial condition:
$$
    u(x,0) = \sin\!\left(frac{n\pi x}{L}\right), \quad x\in[0,L].
$$

Analytical solution:
$$
    u(x,t) = \sin\left(frac{n\pi x}{L}\right)
             \exp\left(-frac{\alpha n^{2}\pi^{2}}{L^{2}}t\right).
$$

Implementation
--------------
- Loads a trained `HeatEquationPinn` model from checkpoint.
- Generates synthetic noisy data from the analytical solution.
- Defines forward maps for:
  - **Analytical solution** (baseline inference).
  - **PINN-based prediction** (surrogate inference).
- Runs MCMC inference with Gaussian likelihood for both maps:
  - Saves posterior samples in CSV files.
  - Computes posterior summaries (means, variances, credible intervals).
- Produces posterior comparison plots between analytical and PINN solutions.

Usage
-----
To run inference:
    $ python heat_mcmc.py

The script will:
1. Load the trained PINN (`heat_parametric_MLP.pth`).
2. Generate synthetic data with noise.
3. Run MCMC inference using analytical and PINN forward maps.
4. Save samples in CSV files (`samples_analytical.csv`, `samples_pinn.csv`).
5. Plot and save posterior comparison (`posterior_comparison.pdf`).

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- The BUQ MCMC sampler is implemented with the t-walk algorithm.
- Posterior inference is performed for a single parameter: thermal diffusivity $\alpha$.
- Outputs include posterior samples, statistics, and visual comparisons.
"""
# Necessary libraries.
import os                                          # File paths.
import sys                                         # System functions.
import random                                      # Random numbers.
import numpy as np                                 # Arrays and math.
import torch                                       # Tensors and autograd.
import scipy.stats as stats                        # Statistical distributions.
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

# Project imports.
from utils import load_full_model, get_model_info
from sampling import generate_synthetic_data_on_square
from inference.mcmc import MCMCInference, define_forward_map
from plotting import plot_joint_posteriors
from inverse_problems.heat_equation_parametric_MLP.heat_equation_parametric_MLP import HeatParametricPinn

# ----------------------------------------------------------------------------------
# Load the trained PINN model for parameter inference.
# ----------------------------------------------------------------------------------
checkpoint_filename = "heat_parametric_MLP.pth"
heat_pinn = load_full_model(
    checkpoint_path=os.path.join("trained_models", checkpoint_filename),
    model_class=HeatParametricPinn
)
get_model_info(checkpoint_filename) # Print model information.

# ----------------------------------------------------------------------------------
# Parameters for MCMC inference.
# ---------------------------------------------------------------------------------- 
L, T, n = 2, 2, 5  # Length of the domain in x, time, and fixed n value.
par_true = [0.05]  # True value to be inferred.
par_names = [r"$\alpha$"]  # Name of the parameters to be inferred.
par_prior = [stats.uniform(loc=0, scale=0.1)]  # Prior distribution ⍺ ~ U(0, 0.1).
par_supp = [lambda a: 0 <= a <= 0.1]  # Support function for the priors.
sigma = 0.01  # Standard deviation for the noise.
n_iter = 500000  # Iterations.
burn_in = int(0.1 * n_iter)  # Burn-in.

# ----------------------------------------------------------------------------------
# Synthetic data generation and forward map definition.
# ----------------------------------------------------------------------------------
n_points = 20  # Number of data points to generate.
data_x, data_u_exact, data_u = generate_synthetic_data_on_square(
    dim1_min=0,
    dim1_max=L,
    dim2_min=0,
    dim2_max=T,
    n_points=n_points,
    pinn_instance=heat_pinn,
    fixed_params=[n],
    par_true=par_true,
    sigma=sigma
)

# Define the forward maps for the analytical and PINN solutions.
analytical_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=heat_pinn, analytic=True
)
pinn_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=heat_pinn, analytic=False
)

# ----------------------------------------------------------------------------------
# File paths for saving/loading MCMC samples.
# ----------------------------------------------------------------------------------
# Directory of the running script.
script_dir = os.path.dirname(
    os.path.abspath(sys.modules["__main__"].__file__)
)

# CSV file for analytical samples.
analytical_csv_path = os.path.join(script_dir, "samples_analytical.csv")

# CSV file for PINN samples.
pinn_csv_path = os.path.join(script_dir, "samples_pinn.csv")

# ----------------------------------------------------------------------------------
# Run MCMC inference
# ----------------------------------------------------------------------------------
print("\n" + "─"*60 + "\nAnalytical Forward Map\n" + "─"*60)
samples_analytical, stats_analytical = MCMCInference(
    filename=analytical_csv_path,
    forward_map=analytical_forward_map,
    data_x=data_x,
    data_u=data_u,
    par_names=par_names,
    par_prior=par_prior,
    par_supp=par_supp,
    par_true=par_true,
    sigma=sigma,
    n_iter=n_iter,
    burn_in=burn_in,
    SimData=False
)
print("\n" + "─"*60 + "\nPINN Forward Map\n" + "─"*60)
samples_pinn, stats_pinn = MCMCInference(
    filename=pinn_csv_path,
    forward_map=pinn_forward_map,
    data_x=data_x,
    data_u=data_u,
    par_names=par_names,
    par_prior=par_prior,
    par_supp=par_supp,
    par_true=par_true,
    sigma=sigma,
    n_iter=n_iter,
    burn_in=burn_in,
    SimData=False
)

# ----------------------------------------------------------------------------------
# Plot joint posterior distributions.
# ----------------------------------------------------------------------------------
plot_joint_posteriors(
    samples1=samples_analytical["samples"],
    samples2=samples_pinn["samples"],
    par_true=par_true,
    par_names=r"$\alpha$",
    bins=30,
    filename="posterior_comparison.png",
    param_idx=0
)