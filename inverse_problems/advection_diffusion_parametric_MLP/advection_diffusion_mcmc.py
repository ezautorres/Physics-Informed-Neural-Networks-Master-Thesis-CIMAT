"""
advection_diffusion_mcmc.py
---------------------------
Bayesian Inference with MCMC for the 1D Advection-Diffusion Equation using PINNs.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Performs Bayesian parameter inference for the 1D advection-diffusion equation
using Markov Chain Monte Carlo (MCMC) with Bayesian Uncertainty Quantification
(BUQ). Both the analytical solution and a trained Physics-Informed Neural 
Network (PINN) surrogate are used as forward models for comparison.

The PDE is:
$$
    u_{t}(x,t) - \alpha u_{xx}(x,t) + \beta u_{x}(x,t) = 0, 
    \quad (x,t)\in(0,L)\times(0,T).
$$

Boundary conditions:
$$
    u(0,t) = u(L,t) = 0, \quad t\in[0,T].
$$

Initial condition:
$$
    u(x,0) = \sin\left(\frac{n \pi x}{L}\right) 
             \exp\left(\frac{\beta x}{2\alpha}\right), \quad x\in[0,L].
$$

Analytical solution:
$$
    u(x,t) = \sin\left(\frac{n \pi x}{L}\right)
             \exp\left(-t\Big(\tfrac{\alpha n^{2}\pi^{2}}{L^{2}}
             + \tfrac{\beta^{2}}{4\alpha}\Big)\right)
             \exp\left(\tfrac{\beta x}{2\alpha}\right).
$$

Implementation
--------------
- Loads a pre-trained `AdvectionDiffusionPinn` model.
- Synthetic data are generated from the analytical solution with Gaussian noise.
- Two forward maps are defined:
  - Analytical closed-form solution.
  - PINN surrogate model prediction.
- Runs MCMC via the `MCMCInference` function:
  - Priors: $\alpha \sim U(0.02,0.12)$, $\beta \sim U(-0.15,0.10)$.
  - Likelihood: Gaussian with $\sigma=0.01$.
  - True parameters: $(\alpha, \beta) = (0.06, -0.05)$.
- Posterior samples are stored as CSV and reused if available.

Visualization
-------------
- Joint posterior histograms comparing analytical vs PINN inference for 
  each parameter.
- Corner plot for joint posterior comparison.

Usage
-----
To run the inference:
    $ python advection_diffusion_mcmc.py

Example output files:
- `"samples_analytical.csv"` : Posterior samples using analytical solution.
- `"samples_pinn.csv"`       : Posterior samples using PINN surrogate.
- `"posterior_alpha.png"`    : Posterior distribution of $\alpha$.
- `"posterior_beta.png"`     : Posterior distribution of $\beta$.
- `"corner_comparison.png"`  : Joint posterior corner plot.

Notes
-----
- Reproducibility ensured via fixed random seeds (NumPy, Python, PyTorch).
- Uses **t-walk MCMC** (`BUQ` sampler) for posterior sampling.
- The PINN surrogate allows inference when the analytical solution is unknown.
- The comparison highlights how close the PINN-based posterior is to the
  analytical posterior.
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
from plotting import plot_joint_posteriors, plot_corner_comparison
from inverse_problems.advection_diffusion_parametric_MLP.advection_diffusion_parametric_MLP import AdvectionDiffusionPinn # Import the PINN class.

# ----------------------------------------------------------------------------------
# Load the trained PINN model for parameter inference.
# ----------------------------------------------------------------------------------
checkpoint_filename = "advection_diffusion_parametric_MLP.pth"
advection_diffusion_pinn = load_full_model(
    checkpoint_path=os.path.join("trained_models", checkpoint_filename),
    model_class=AdvectionDiffusionPinn
)
get_model_info(checkpoint_filename) # Print model information.

# ----------------------------------------------------------------------------------
# Parameters for MCMC inference.
# ----------------------------------------------------------------------------------
L, T, n = 1, 1, 2  # Length of the domain in x, time, and fixed n value.
par_true = [0.06, -0.05]  # True values to be inferred.
par_names = [r"$\alpha$", r"$\beta$"]  # Name of the parameters to be inferred.
par_prior = [
    stats.uniform(loc=0.02, scale=0.1),  # ⍺ ~ U(0.02, 0.12).
    stats.uniform(loc=-0.15, scale=0.25)  # β ~ U(-0.15, 0.10)].
]
par_supp = [
    lambda a: 0.02 <= a <= 0.12,  # Support for ⍺.
    lambda b: -0.15 <= b <= 0.1  # Support for β.
]
sigma = 0.01  # Standard deviation for the noise.
n_iter = 500000  # Iterations.
burn_in = int(0.1 * n_iter)  # Burn-in.

# ----------------------------------------------------------------------------------
# Synthetic data generation and forward map definition.
# ----------------------------------------------------------------------------------
n_points = 100  # Number of data points to generate.
data_x, data_u_exact, data_u = generate_synthetic_data_on_square(
    dim1_min=0,
    dim1_max=L,
    dim2_min=0,
    dim2_max=T,
    n_points=n_points,
    pinn_instance=advection_diffusion_pinn,
    fixed_params=[n],
    par_true=par_true,
    sigma=sigma
)

# Define the forward maps for the analytical and PINN solutions.
analytical_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=advection_diffusion_pinn, analytic=True
)
pinn_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=advection_diffusion_pinn, analytic=False
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
# For α.
plot_joint_posteriors(
    samples1=samples_analytical["samples"],
    samples2=samples_pinn["samples"],
    par_true=par_true,
    par_names=par_names,
    bins=30,
    filename="posterior_alpha.png",
    param_idx=0
)

# For β.
plot_joint_posteriors(
    samples1=samples_analytical["samples"],
    samples2=samples_pinn["samples"],
    par_true=par_true,
    par_names=par_names,
    bins=30,
    filename="posterior_beta.png",
    param_idx=1
)

# ----------------------------------------------------------------------------------
# Plot Corner.
# ----------------------------------------------------------------------------------
plot_corner_comparison(
    samples_analytical=samples_analytical["samples"],
    samples_pinn=samples_pinn["samples"],
    par_names=par_names,
    par_true=par_true,               # lista
    bins=30,
    filename="corner_comparison.png",
)