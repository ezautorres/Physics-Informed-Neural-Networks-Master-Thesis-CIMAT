"""
poisson_pinn.py
---------------
Instance of a Physics-Informed Neural Network (PINN) applied to the Poisson Equation in 2D.

Author: Ezau Faridh Torres Torres.
Date: 21 January 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves the Poisson equation using a Physics-Informed Neural Network (PINN). The Poisson
equation is defined in the unit square domain [0,1]x[0,1] with homogeneous Dirichlet boundary conditions.

The PDE to be solved is:
    $\Delta u = -2\pi^2 \sin(\pi x) sin(\pi y),  for $(x,y) \in (0,1)x(0,1)$.

Subject to boundary conditions:
    $u(x,0) = 0$,   $u(x,1) = 0$,  
    $u(0,y) = 0$,   $u(1,y) = 0$

The exact analytical solution is:
    $u(x,y) = sin(\pi x) sin(\pi y)$

This implementation includes:
    - A custom class 'PoissonPinn' inheriting from a general PINN base class.
    - The definition of the physical loss term and boundary condition loss.
    - Training and visualization routines.

Usage
-----
Run the script directly to:
    - Instantiate and train the PINN for the Poisson problem.
    - Save and load checkpoints.
    - Visualize the loss, solution, and comparison with the analytical solution.
"""
import numpy as np                                                                   # Numpy library.
import torch                                                                         # Import PyTorch
from typing import Callable                                                          # Type hinting.
import sys, os                                                                       # Import sys and os modules.
import random                                                                        # Random library.
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

class PoissonPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the PoissonPinn instance using the configuration dictionary passed to the base class.
        
        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including model configuration,
            optimizer settings, and domain sampling specifications.
        """
        super(PoissonPinn, self).__init__(**params) # Initialize the PINN with parameters from the base class.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution u(x,y) = sin(\pi x) sin(\pi y) evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point (x,y) in the domain.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at each input point.
        """
        return torch.sin(torch.pi * X[:,0]) * torch.sin(torch.pi * X[:,1])

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss and the boundary
        condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(x,y)$.
        X : torch.Tensor
            Tensor of input points, where the first N_pde entries correspond to interior domain points
            and the rest to boundary points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current batch.

        Notes
        -----
        In this example, initial and additional conditions are not used, so the loss is computed only
        from the PDE residual and boundary conditions.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0 # λ_pde
        lb_bc  = 1.0 # λ_bc

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"] # Number of interior points for the PDE.
        
        # Split the input tensor X into the different regions.
        X_pde = X[0:N_pde] # Points in the interior for the PDE.
        X_bc  = X[N_pde:]  # Points for the boundary conditions.

        # -----------------------------------------------------------------------------------------------
        # PDE loss: 𝚫u = -2π²sin(πx)sin(πy). 
        # -----------------------------------------------------------------------------------------------
        u_pde = net(X_pde)                                                                                        # Compute the model output for the PDE points.
        grad_u = torch.autograd.grad(u_pde, X_pde, grad_outputs = torch.ones_like(u_pde), create_graph = True)[0] # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂y.
        u_x, u_y = grad_u[:,0], grad_u[:,1]                                                                       # ∂u/∂x, ∂u/∂y.
        u_xx = torch.autograd.grad(u_x, X_pde, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:,0]  # ∂²u/∂x².
        u_yy = torch.autograd.grad(u_y, X_pde, grad_outputs = torch.ones_like(u_y), create_graph = True)[0][:,1]  # ∂²u/∂y².
        source = -2 * torch.pi**2 * torch.sin(torch.pi * X_pde[:,0]) * torch.sin(torch.pi * X_pde[:,1])           # Source term: -2π²sin(πx)sin(πy).
        
        loss_pde = torch.mean((u_xx + u_yy - source)**2) 

        # -----------------------------------------------------------------------------------------------
        # Boundary condition loss: u(x,0) = 0, u(x,1) = 0, u(0,y) = 0, u(1,y) = 0.
        # -----------------------------------------------------------------------------------------------
        loss_bc = torch.mean(net(X_bc)**2) # Compute the boundary condition loss.

        return lb_pde * loss_pde + lb_bc * loss_bc

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
        'interiorSize'  : 500,
        'dim1_minSize'  : 2000,
        'dim1_maxSize'  : 2000,
        'dim2_minSize'  : 2000,
        'dim2_maxSize'  : 2000,
        'valSize'       : 2000,
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
        'inputSize'  : 2,               # Because we do not have parameters.
        'hidden_lys' : [100, 100, 100], # Hidden layers of the MLP.
        'outputSize' : 1                # Output size of the MLP (1 for the Poisson equation).
    }
    
    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr'               : 1,             # Learning rate.
        'max_iter'         : 100,           # Maximum number of iterations.
        'tolerance_grad'   : 1e-09,         # Tolerance for the gradient.
        'tolerance_change' : 1e-09,         # Tolerance for the change in the loss.
        'history_size'     : 100,           # History size for the optimizer.
        'line_search_fn'   : "strong_wolfe" # Line search function for the optimizer.
    }

    checkpoint_filename = 'poisson_MLP.pth'
    poisson_pinn = PoissonPinn(
        model_class         = MLP,                   # Model class for the PINN.
        model_kwargs        = model_kwargs,          # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,         # Domain parameters.
        optimizer_class     = optimizer_class,       # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,      # Optimizer parameters.
        epochs              = 150,                   # Number of epochs for training.
        patience            = 10,                    # Patience for early stopping.
        sampling_fn         = sample_square_uniform, # Sampling function.
        checkpoint_filename = checkpoint_filename,   # Filename for the checkpoints.
    )

    # Train the model.
    #poisson_pinn.train()

    # Load the complete model.
    poisson_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)        # Print model information.
    
    # Plot the loss and the solution.
    plot_loss(
        model_instance = poisson_pinn,
        filename       = "loss_plot.pdf"
    )

    # Plot the solution with the best model.
    poisson_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = poisson_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.pdf"
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance = poisson_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.pdf"
    )