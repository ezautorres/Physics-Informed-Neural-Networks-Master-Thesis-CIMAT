import os, sys                                                                     # For file and directory operations.
import numpy as np                                                                 # Import NumPy.
import torch                                                                       # Import PyTorch.
import scipy.stats as stats                                                        # Import scipy.stats for statistical functions.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # Add the parent directory to the path.
from inference.mcmc import MCMCInference                                           # Import MCMC inference class and utility function.
from utils import get_model_info, load_full_model                                  # Import utility functions.
from plotting import plot_joint_posteriors                                         # Import plotting function.
from inverse_problems.infer_conductivity_value_MLP.infer_conductivity_value_MLP import InferringConductivityValue

# ------------------------------------------------------------------------------------------------------
# Load the trained PINN model for conductivity inference.
# ------------------------------------------------------------------------------------------------------
checkpoint_filename = "infer_conductivity_value_MLP.pth"
infer_rho_pinn = load_full_model(
    checkpoint_path = os.path.join("trained_models", checkpoint_filename),
    model_class     = InferringConductivityValue)
#get_model_info(checkpoint_filename) # Print model information.

# ------------------------------------------------------------------------------------------------------
# Parameters and data for MCMC inference.
# ------------------------------------------------------------------------------------------------------ 
R         = 0.85                         # Additional parameter for the forward maps.
par_true  = 3.2                          # True value of the parameters to be inferred.
par_names = ["rho"]                      # Name of the parameters to be inferred.
par_prior = [stats.uniform(0,10)]        # Prior distribution for the parameters.
par_supp  = [lambda rho: 0 <= rho <= 10] # Support function for the prior distributions.
sigma     = 0.01                         # Standard deviation for the noise in the data.
n_iter    = 100000                       # Number of MCMC iterations.
burn_in   = 10000                        # Number of burn-in iterations.

# Observed data points (x,u) for the inference.
data_x = np.array([
    (np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, 10, endpoint = False)
    ])
data_u = np.array([
    0.16847560805179593, -0.13965275243926256, 0.0683518569802451, 0.06133918738120912, -0.1667469200806133,
    0.18860308992045097, -0.14238846110196404, 0.0724993834591109, 0.05466561651529539, -0.14466339803863262
    ])

# ------------------------------------------------------------------------------------------------------
# Define the forward maps for the analytical and PINN solutions.
# ------------------------------------------------------------------------------------------------------
def analytical_forward_map(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Forward map for the analytical solution.

    Parameters
    ----------
    theta : np.ndarray
        Parameter values to be inferred.
    t : np.ndarray
        Time points at which the solution is evaluated.
    """
    rho_val = torch.full((len(t),), theta[0], dtype = torch.float32)                 # Conductivity value.
    R_val   = torch.full((len(t),), R, dtype = torch.float32)                        # Additional parameter.
    X = torch.column_stack((torch.tensor(t, dtype = torch.float32), R_val, rho_val)) # Create input tensor for the forward map.

    return infer_rho_pinn.analytical_solution(X).detach().numpy().flatten()

def pinn_forward_map(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Forward map for the PINN solution.

    Parameters
    ----------
    theta : np.ndarray
        Parameter values to be inferred.
    t : np.ndarray
        Time points at which the solution is evaluated.
    """
    rho_val = torch.full((len(t),), theta[0], dtype = torch.float32)                 # Conductivity value.
    R_val   = torch.full((len(t),), R, dtype = torch.float32)                        # Additional parameter.
    X = torch.column_stack((torch.tensor(t, dtype = torch.float32), R_val, rho_val)) # Create input tensor for the forward map.

    return infer_rho_pinn.pinn(X).detach().numpy().flatten()

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
    burn_in     = burn_in
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
    burn_in     = burn_in
)

# ------------------------------------------------------------------------------------------------------
# Plot joint posterior distributions.
# ------------------------------------------------------------------------------------------------------
plot_joint_posteriors(
    samples1  = samples_analytical["samples"],
    samples2  = samples_pinn["samples"],
    par_true  = par_true,
    par_names = r"$\rho$",
    bins      = 30,
    filename  = "posterior_comparison.pdf"
)