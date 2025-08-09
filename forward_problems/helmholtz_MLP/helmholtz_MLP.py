"""
helmholtz_nonhomogeneous_MLP.py
---------------------------------
Instance of a Physics-Informed Neural Network (PINN) applied to a Nonhomogeneous Helmholtz Equation in 2D.

Author: Ezau Faridh Torres Torres.
Date: 06 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves a nonhomogeneous Helmholtz equation using a Physics-Informed Neural Network (PINN).
The problem is defined on the square domain (x,y) \in [0,1]^2, subject to homogeneous Dirichlet boundary
conditions.

The PDE to be solved is:
    -u_{xx} - u_{yy} - k_0^2 u = f(x,y)
where the source term is:
    f(x,y) = k_0^2 \sin(k_0 x) \sin(k_0 y)

Subject to:
    - Boundary conditions: u(x,y) = 0 for (x,y) \in \partial \Omega.

The exact analytical solution is:
    u(x,y) = \sin(k_0 x) \sin(k_0 y).

This implementation includes:
    - A custom class 'HelmholtzNonhomogeneousPinn' inheriting from a general PINN base class.
    - The definition of the PDE residual and boundary loss terms.
    - Training, model saving, and visual comparison with the analytical solution.

Usage
-----
Run the script directly to:
    - Instantiate and train the PINN for the 2D Helmholtz problem.
    - Save and load checkpoints.
    - Visualize the loss, the predicted solution, and the comparison with the exact solution.
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
class HelmholtzNonhomogeneousPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the HelmholtzNonhomogeneousPinn instance using the configuration dictionary passed to
        the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including model configuration, optimizer
            settings, and domain sampling specifications.
        """
        super(HelmholtzNonhomogeneousPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution u(x,y;k) = sin(kx) sin(ky) evaluated at input points X, k must
        be a multiple of pi (default k0 = pi).

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point (x,y) in the domain and the 
            third column contains the wave number k.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at each input point.
        """
        k = X[:,2]
        return torch.sin(k * X[:,0]) * torch.sin(k * X[:,1])

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss and the
        boundary condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(x,y;k)$.
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
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc  = len(X) - N_pde

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde),   # Interior points [0,1]x[0,1].
            torch.ones(N_bc) * 2 # Boundary points (x,y)ϵ ∂Ω.
        )).to(X.device)

        # Loss components.
        loss_pde = torch.tensor(0.0, device = X.device) # PDE loss.
        loss_bc  = torch.tensor(0.0, device = X.device) # Boundary condition loss.

        for i in range(len(X)):
            x = X[i,:].unsqueeze(0).requires_grad_(True) # Input point (x,y) for the current sample.
            region = int(indicators[i].item())           # Determine the region based on the indicator.
            u = net(x)                                   # Output of the neural network for the current input point.

            # -----------------------------------------------------------------------------------------------
            # PDE loss: N[u] = f => -𝚫u - ku = k²sin(kx)sin(ky).
            # -----------------------------------------------------------------------------------------------
            if region == 1:
                grad_u = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0]        # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂y.
                u_x, u_y  = grad_u[:,0], grad_u[:,1]                                                                 # ∂u/∂x, ∂u/∂y.
                u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:,0] # ∂²u/∂x².
                u_yy = torch.autograd.grad(u_y, x, grad_outputs = torch.ones_like(u_y), create_graph = True)[0][:,1] # ∂²u/∂y².
                k = x[0,2]                                                                                           # Extract the wave number k from the input point.
                f = k**2 * torch.sin(k * x[0,0]) * torch.sin(k * x[0,1])                                             # Source term: k²sin(kx)sin(ky).

                loss_pde += (-u_xx - u_yy - k**2 * u - f).pow(2).squeeze() # Compute the PDE residual loss.

        # -----------------------------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0.
        # -----------------------------------------------------------------------------------------------
            elif region == 2:
                loss_bc += u.pow(2).squeeze() # Compute the boundary condition loss.

        # Normalize each term.
        loss_pde /= N_pde
        loss_bc  /= N_bc

        return lb_pde * loss_pde + lb_bc * loss_bc

# =======================================================================================================
# Main function.
# =======================================================================================================
if __name__ == "__main__":

    from architectures import MLP              # Import the MLP architecture for the PINN.
    from sampling import sample_square_uniform # Sampling function for uniform sampling in a square domain.
    
    k = 3*torch.pi # Wave number (multiple of pi).

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
        'interiorSize'  : 2000,
        'dim1_minSize'  : 500,
        'dim1_maxSize'  : 500,
        'dim2_minSize'  : 500,
        'dim2_maxSize'  : 500,
        'valSize'       : 500,
        # Parameters for the PINN.
        'fixed_params'  : [k],
        'param_domains' : None,
        # Observed data.
        'data_x'        : None,
        'data_u'        : None,
    }

    # ---------------------------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ---------------------------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize'  : 3,                  # We have 2 spatial dimensions (x,y) and 1 parameter (k).
        'hidden_lys' : [100, 150, 75, 50], # Hidden layers with specified sizes.
        'outputSize' : 1                   # Output size is 1 (the solution u).     
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

    checkpoint_filename = 'helmholtz_nonhomogeneous_MLP.pth'
    helmholtz_pinn = HelmholtzNonhomogeneousPinn(
        model_class         = MLP,                   # Model class for the PINN.
        model_kwargs        = model_kwargs,          # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,         # Domain parameters.
        optimizer_class     = optimizer_class,       # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,      # Optimizer parameters.
        epochs              = 10,                   # Number of epochs for training.
        patience            = 5,                     # Patience for early stopping.
        sampling_fn         = sample_square_uniform, # Sampling function.
        checkpoint_filename = checkpoint_filename,   # Filename for the checkpoints.
    )

    # Train the model.
    #helmholtz_pinn.train()

    # Load the complete model.
    helmholtz_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)          # Print model information.

    # Plot the loss and the solution.
    plot_loss(
        model_instance = helmholtz_pinn,
        filename       = "loss_plot.png"
    )

    # Plot the solution with the best model.
    helmholtz_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = helmholtz_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.png",
        parameters     = [k]
    )

    # Plot the comparison with the analytical solution.
    plot_comparison_contour_square(
        model_instance = helmholtz_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.png",
        parameters     = [k]
    )