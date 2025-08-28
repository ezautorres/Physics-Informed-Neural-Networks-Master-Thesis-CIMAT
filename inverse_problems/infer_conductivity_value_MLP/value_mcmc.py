"""
value_mcmc.py
--------------------------
Bayesian Inference with MCMC for the Conductivity Problem in the Unit Disk
using PINNs.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Performs Bayesian parameter inference for the conductivity problem in the
unit disk using Markov Chain Monte Carlo (MCMC) with Bayesian Uncertainty
Quantification (BUQ). Both the analytical solution and a trained 
Physics-Informed Neural Network (PINN) surrogate are used as forward models 
for comparison.

The PDE is:
$$
    \nabla \cdot (\lambda(x,y; R, \rho) \nabla u(x,y)) = 0, 
    \quad (x,y)\in \Omega \subset \mathbb{R}^2,
$$
with piecewise conductivity:
$$
    \lambda(r) = 
    \begin{cases}
        1 + \rho, & r < R, \\
        1, & r \geq R,
    \end{cases}
$$
where $r = \sqrt{x^2+y^2}$.

Boundary condition:
$$
    \lambda \frac{\partial u}{\partial n} = \cos(4\theta),
    \quad (x,y)\in \partial\Omega,
$$
with $\theta = \arctan(y/x)$.

Analytical solution (polar form):
$$
    u(r,\theta) =
    \begin{cases}
        2(b+c)\,(r/R)^4 \cos(4\theta), & r < R, \\
        2\big(b\,(r/R)^4 + c\,(r/R)^{-4}\big)\cos(4\theta), & r \geq R,
    \end{cases}
$$
with coefficients $b, c$ depending on $\rho, R$.

Implementation
--------------
- Loads a pre-trained `InferringConductivityValue` PINN model.
- Synthetic boundary data are generated with Gaussian noise.
- Two forward maps are defined:
  - Analytical closed-form solution.
  - PINN surrogate model prediction.
- Runs MCMC via the `MCMCInference` function:
  - Prior: $\rho \sim U(0,10)$.
  - Likelihood: Gaussian with $\sigma=0.01$.
  - True parameter: $\rho = 3.2$, fixed $R=0.85$.
- Posterior samples are stored as CSV and reused if available.

Visualization
-------------
- Posterior distribution of $\rho$ (analytical vs PINN).
- Joint posterior plot for comparison.

Usage
-----
To run the inference:
    $ python value_mcmc.py

Example output files:
- `"samples_analytical.csv"` : Posterior samples using analytical solution.
- `"samples_pinn.csv"`       : Posterior samples using PINN surrogate.
- `"posterior_comparison.png"` : Posterior distribution of $\rho$.

Notes
-----
- Reproducibility ensured via fixed random seeds (NumPy, Python, PyTorch).
- Uses **t-walk MCMC** (`BUQ` sampler) for posterior sampling.
- PINN surrogate enables parameter inference in cases where the analytical
  solution is unavailable.
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
from sampling import generate_synthetic_data_on_circle_boundary
from inference.mcmc import MCMCInference, define_forward_map
from plotting import plot_joint_posteriors
from inverse_problems.infer_conductivity_value_MLP.infer_conductivity_value_MLP import InferringConductivityValue

# ----------------------------------------------------------------------------------
# Load the trained PINN model for parameter inference.
# ----------------------------------------------------------------------------------
checkpoint_filename = "infer_conductivity_value_MLP.pth"
infer_rho_pinn = load_full_model(
    checkpoint_path=os.path.join("trained_models", checkpoint_filename),
    model_class=InferringConductivityValue
)
get_model_info(checkpoint_filename)  # Print model information.

# ----------------------------------------------------------------------------------
# Parameters for MCMC inference.
# ----------------------------------------------------------------------------------
R = [0.85]  # Radius of the circular domain.
par_true = [3.2]  # True value to be inferred.
par_names = [r"$\rho$"]  # Name of the parameters to be inferred.
par_prior = [stats.uniform(0, 10)]  # Prior distribution for 𝞺 ~ U(0, 10).
par_supp = [lambda p: 0 <= p <= 10]  # Support function for the prior.
sigma = 0.01  # Standard deviation for the noise.
n_iter = 500000  # Iterations.
burn_in = int(0.1 * n_iter)  # Burn-in.

# ----------------------------------------------------------------------------------
# Synthetic data generation and forward map definition.
# ----------------------------------------------------------------------------------
n_points = 20  # Number of data points to generate.
data_x, data_u_exact, data_u = generate_synthetic_data_on_circle_boundary(
    center=(0.0, 0.0),
    radius=1.0,
    n_points=n_points,
    pinn_instance=infer_rho_pinn,
    fixed_params=R,
    par_true=par_true,
    sigma=sigma
)

# Define the forward maps for the analytical and PINN solutions.
analytical_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=infer_rho_pinn, analytic=True
)
pinn_forward_map = lambda theta, t: define_forward_map(
    theta, t, pinn_instance=infer_rho_pinn, analytic=False
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
    burn_in=burn_in
)

# ----------------------------------------------------------------------------------
# Plot joint posterior distributions.
# ----------------------------------------------------------------------------------
plot_joint_posteriors(
    samples1=samples_analytical["samples"],
    samples2=samples_pinn["samples"],
    par_true=par_true,
    par_names=par_names,
    bins=30,
    filename="posterior_comparison.png",
    param_idx=0,
)