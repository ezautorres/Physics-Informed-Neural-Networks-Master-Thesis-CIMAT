
"""
wave_highfreq_MLP.py
-------------------------------
Instance of a Physics-Informed Neural Network (PINN) applied to a linear wave equation with high-frequency
components.

Author: Ezau Faridh Torres Torres.
Date: 6 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves a linear wave equation using a Physics-Informed Neural Network (PINN).
The problem is defined on the spatio-temporal domain [x,t] ∈ [0,1] × [0,1], subject to
Dirichlet boundary conditions and an initial condition.

The PDE to be solved is:
    ∂²u/∂t² - 100 ∂²u/∂x² = 0.

Subject to:
    - Initial condition: u(x,0) = sin(πx) + sin(2πx), ∂u/∂t(x,0) = 0.
    - Boundary conditions: u(0,t) = u(1,t) = 0.

The exact analytical solution is:
    u(x,t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt).

This implementation includes:
    - A custom class 'WaveHighFreqPinn' inheriting from a general PINN base class.
    - The definition of the PDE residual, initial and boundary loss terms.
    - Training, model saving, and visual comparison with the analytical solution.

Usage
-----
Run the script directly. It will train the PINN and generate plots for loss convergence and solution comparison.
"""
import numpy as np                                                                   # Numpy library.
import torch                                                                         # Import PyTorch
from typing import Callable                                                          # Type hinting.
import sys, os                                                                       # Import sys and os modules.
import random                                                                        # Random module for reproducibility.
np.set_printoptions(precision = 17, suppress = False)                                # Set print options for numpy.
np.random.seed(0)                                                                    # Seed for reproducibility.
random.seed(0)                                                                       # Seed for reproducibility.
torch.manual_seed(0)                                                                 # Seed for reproducibility.
torch.backends.cudnn.benchmark = False                                               # Reproducibility for CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                # Set device to GPU if available, else CPU.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # Add the parent directory to the path.
from pinn_base import PinnBase                                                       # Import the base class for PINNs.
from plotting import plot_loss, plot_solution_square, plot_comparison_contour_square # Plotting functions.
from utils import get_model_info                                                     # Utility function to get model information.


class WaveHighFreqPinn(PinnBase):
    def __init__(self, **params):
        super(WaveHighFreqPinn, self).__init__(**params)
        self.c = 10.0  # Wave speed.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        x = X[:,0]
        t = X[:,1]
        return torch.sin(np.pi * x) * torch.cos(self.c * np.pi * t) + torch.sin(2 * np.pi * x) * torch.cos(2 * self.c * np.pi * t)

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss using a fully vectorized implementation.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(x,t)$.
        X : torch.Tensor
            Input tensor of shape (N_total, 2), containing collocation, boundary, and initial points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss.
        """
        # Define the weights for the different loss components.
        lb_pde = 5 # λ_pde.
        lb_ic  = 1 # λ_ic.
        lb_bc  = 1 # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde   = self.domain_kwargs["interiorSize"] # Number of PDE collocation points.
        N_bc_l  = self.domain_kwargs["dim1_minSize"] # Number of left boundary points.
        N_bc_r  = self.domain_kwargs["dim1_maxSize"] # Number of right boundary points.
        N_ic    = self.domain_kwargs["dim2_minSize"] # Number of initial condition points.

        # Split the input tensor X into the different regions.
        X_pde = X[0:N_pde]                          # PDE collocation points.
        X_bcl = X[N_pde:N_pde+N_bc_l]               # Left boundary points.
        X_bcr = X[N_pde+N_bc_l:N_pde+N_bc_l+N_bc_r] # Right boundary points.
        X_ic  = X[-N_ic:]                           # Initial condition points.

        # -----------------------------------------------------------------------------------------------
        # PDE loss: N[u] = f => u_tt - c² u_xx = 0.
        # -----------------------------------------------------------------------------------------------
        u_pde = net(X_pde)                                                                                        # Compute the model output for the PDE points.
        grad_u = torch.autograd.grad(u_pde, X_pde, grad_outputs = torch.ones_like(u_pde), create_graph = True)[0] # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂y.
        u_x, u_t = grad_u[:,0], grad_u[:,1]                                                                       # ∂u/∂x, ∂u/∂t
        u_xx = torch.autograd.grad(u_x, X_pde, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:,0]  # ∂²u/∂x².
        u_tt = torch.autograd.grad(u_t, X_pde, grad_outputs = torch.ones_like(u_t), create_graph = True)[0][:,1]  # ∂²u/∂t².

        loss_pde = torch.mean((u_tt - self.c**2 * u_xx)**2) # Compute the PDE residual loss.

        # -----------------------------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => u(0,t) = 0, u(1,t) = 0.
        # -----------------------------------------------------------------------------------------------
        loss_bc = torch.mean(net(X_bcl)**2) + torch.mean(net(X_bcr)**2)

        # -----------------------------------------------------------------------------------------------
        # Initial condition loss: u0(x) = sin(pi x) + sin(2pi x), ∂u/∂t(x,0) = 0.
        # -----------------------------------------------------------------------------------------------
        u_ic = net(X_ic)                                                                                        # Compute the model output for the initial condition points.               
        grad_ic = torch.autograd.grad(u_ic, X_ic, grad_outputs = torch.ones_like(u_ic), create_graph = True)[0] # ∇u, grad_ic[:,0] = ∂u/∂x, grad_ic[:,1] = ∂u/∂t.
        u_t_ic = grad_ic[:,1]                                      # ∂u/∂t at initial condition points.
        x_ic = X_ic[:,0]                                           # Extract x values from initial condition points.
        u0 = torch.sin(np.pi * x_ic) + torch.sin(2 * np.pi * x_ic) # Initial condition function u0(x).

        loss_ic = torch.mean((u_ic.squeeze() - u0)**2 + u_t_ic**2) # Compute the initial condition loss.

        return lb_pde * loss_pde + lb_ic * loss_ic + lb_bc * loss_bc

# =======================================================================================================
# Main function.
# =======================================================================================================
if __name__ == "__main__":

    from architectures import MLP              # Import the MLP architecture for the PINN.
    from sampling import sample_square_uniform # Sampling function for uniform sampling in a square domain.
    
    # ---------------------------------------------------------------------------------------------------
    # Domain and model parameters.
    # ---------------------------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'dim1_min'      : 0.,
        'dim1_max'      : 1.,
        'dim2_min'      : 0.,
        'dim2_max'      : 1.,
        # Collocation points.
        'interiorSize'  : 1500,
        'dim1_minSize'  : 100,
        'dim1_maxSize'  : 100,
        'dim2_minSize'  : 500,
        'dim2_maxSize'  : 0,
        'valSize'       : 700,
        # Parameters for the PINN.
        'fixed_params'  : None,
        'param_domains' : None,
        # Observed data.
        'data_x'        : None,
        'data_u'        : None,
    }

    # ---------------------------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ---------------------------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize'  : 2,       # Because we do not have parameters.
        'hidden_lys' : [100]*6, # Hidden layers of the MLP.
        'outputSize' : 1        # Output size.
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr'               : 1,             # Learning rate.
        'max_iter'         : 32,            # Maximum number of iterations.
        'tolerance_grad'   : 1e-09,         # Tolerance for the gradient.
        'tolerance_change' : 1e-09,         # Tolerance for the change in the loss.
        'history_size'     : 100,           # History size for the optimizer.
        'line_search_fn'   : "strong_wolfe" # Line search function for the optimizer.
    }

    checkpoint_filename = "wave_highfreq_MLP.pth"
    wave_pinn = WaveHighFreqPinn(
        model_class         = MLP,                   # Model class for the PINN.
        model_kwargs        = model_kwargs,          # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,         # Domain parameters.
        optimizer_class     = optimizer_class,       # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,      # Optimizer parameters.
        epochs              = 10000,                 # Number of epochs for training.
        patience            = 100,                   # Patience for early stopping.
        sampling_fn         = sample_square_uniform, # Sampling function.
        checkpoint_filename = checkpoint_filename,   # Filename for the checkpoints.
    )

    # Train the model.
    #wave_pinn.train()

    # Load the complete model.
    wave_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)     # Print model information.

    # Plot the loss and the solution.
    plot_loss(
        model_instance = wave_pinn,
        filename       = "loss_plot.png"
    )

    # Plot the loss and the solution.
    wave_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = wave_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.png",
        time_dependent = True,
        adjust_zlim    = True
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance = wave_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.png",
        time_dependent = True,
        levels         = 10,
        fix = True, # Fix the levels for the contour plot.
    )