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
R = 0.85  # Radius of the circular domain.
par_true = [3.2]  # True value to be inferred.
par_names = [r"$\rho$"]  # Name of the parameters to be inferred.
par_prior = [stats.uniform(0,10)]  # Prior distribution 𝞺 ~ U(0, 10).
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
    fixed_params=[R],
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