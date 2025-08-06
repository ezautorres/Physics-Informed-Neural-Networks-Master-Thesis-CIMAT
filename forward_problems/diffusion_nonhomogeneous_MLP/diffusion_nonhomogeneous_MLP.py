"""
diffusion_nonhomogeneous_MLP.py
-------------------------------
Instance of a Physics-Informed Neural Network (PINN) applied to a Nonhomogeneous Diffusion Equation in 1D.

Author: Ezau Faridh Torres Torres.
Date: 21 January 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves a nonhomogeneous diffusion equation using a Physics-Informed Neural Network (PINN). 
The problem is defined on the spatio-temporal domain \([xt, t] \in [-1,1] \times [0,1]\), subject to 
homogeneous Dirichlet boundary conditions and an initial condition.

The PDE to be solved is:
    \[
    \frac{\partial y}{\partial t} = \frac{\partial^2 y}{\partial xt^2} - e^{-t}(\sin(\pi xt) - \pi^2 \sin(\pi xt))
    \]

Subject to:
    - Initial condition: \( y(xt,0) = \sin(\pi xt) \)
    - Boundary conditions: \( y(-1,t) = y(1,t) = 0 \)

The exact analytical solution is:
    \[
    y(xt,t) = e^{-t} \sin(\pi xt)
    \]

This implementation includes:
    - A custom class 'DiffusionNonhomogeneousPinn' inheriting from a general PINN base class.
    - The definition of the PDE residual, initial and boundary loss terms.
    - Training, model saving, and visual comparison with the analytical solution.

Usage
-----
Run the script directly to:
    - Instantiate and train the PINN for the 1D diffusion problem.
    - Save and load checkpoints.
    - Visualize the loss, the predicted solution, and the comparison with the exact solution.
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

class DiffusionNonhomogeneousPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the DiffusionNonhomogeneousPinn instance using the configuration dictionary passed to
        the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including model configuration,
            optimizer settings, and domain sampling specifications.
        """
        super(DiffusionNonhomogeneousPinn, self).__init__(**params) # Initialize the PINN with parameters from the base class.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution u(x,t) = \exp{-t} sin(\pi xt) evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point (x,t) in the domain.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at each input point.
        """
        return torch.exp(-X[:,1]) * torch.sin(torch.pi * X[:,0])
    
    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss and the boundary
        condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(xt,t)$.
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
        lb_pde = 1.0 # lb_pde.
        lb_ic  = 1.0 # λ_ic.
        lb_bc  = 1.0 # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde  = self.domain_kwargs["interiorSize"]
        N_bc_l = self.domain_kwargs["dim1_minSize"]
        N_bc_r = self.domain_kwargs["dim1_maxSize"]
        N_ic   = self.domain_kwargs["dim2_minSize"]
        N_total = N_pde + N_bc_l + N_bc_r + N_ic

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde),      # Interior points [-1,1]xt[0,1].
            torch.ones(N_bc_l) * 2, # Boundary points at xt = -1.
            torch.ones(N_bc_r) * 3, # Boundary points at xt = 1.
            torch.ones(N_ic)   * 4  # Initial condition points at t = 0.
        )).to(X.device)

        # Loss components.
        loss_pde = torch.tensor(0.0, device = X.device) # PDE loss.
        loss_ic  = torch.tensor(0.0, device = X.device) # Initial condition loss.
        loss_bc  = torch.tensor(0.0, device = X.device) # Boundary condition loss.

        for i in range(N_total):
            xt = X[i,:].unsqueeze(0).requires_grad_(True) # Input point (x,t) for the current sample.
            region = int(indicators[i].item())            # Determine the region based on the indicator.
            u = net(xt)                                   # Output of the neural network for the current input point.

            # -----------------------------------------------------------------------------------------------
            # PDE loss: N[u] = f => u_{t} - u_{xx} = -\exp{-t} sin(πx) (1 - π²).
            # -----------------------------------------------------------------------------------------------
            if region == 1:
                grad_u = torch.autograd.grad(u, xt, grad_outputs = torch.ones_like(u), create_graph = True)[0]        # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂t.
                u_x, u_t = grad_u[:,0], grad_u[:,1]                                                                   # ∂u/∂x, ∂u/∂t.
                u_xx = torch.autograd.grad(u_x, xt, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:,0] # ∂²u/∂x².
                f = - torch.exp(-xt[0,1]) * torch.sin(torch.pi * xt[0,0]) * (1 - torch.pi**2)                         # Source term: -exp(-t)sin(πx)(1 - π²).
                
                loss_pde += (u_t - u_xx - f).pow(2).squeeze()

            # -----------------------------------------------------------------------------------------------
            # Boundary condition loss: B[u] = g => u(-1,t) = u(1,t) = 0.
            # -----------------------------------------------------------------------------------------------
            elif region in [2,3]:
                loss_bc += u.pow(2).squeeze() # Compute the boundary condition loss.

            # -----------------------------------------------------------------------------------------------
            # Initial condition loss: u(xt,0) = sin(πx).
            # -----------------------------------------------------------------------------------------------
            elif region == 4:
                u_0 = torch.sin(torch.pi * xt[0,0])   # Initial condition at t = 0.
                loss_ic += (u - u_0).pow(2).squeeze() # Compute the initial condition loss.

        # Normalize each term.
        loss_pde /= N_pde
        loss_ic  /= N_ic
        loss_bc  /= (N_bc_l + N_bc_r)

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
        'dim1_min'      : -1.,
        'dim1_max'      : 1.,
        'dim2_min'      : 0.,
        'dim2_max'      : 1.,
        # Collocation points.
        'interiorSize'  : 200,
        'dim1_minSize'  : 100,
        'dim1_maxSize'  : 100,
        'dim2_minSize'  : 100,
        'dim2_maxSize'  : 0,
        'valSize'       : 180,
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
        'inputSize'  : 2,              # Because we do not have parameters.
        'hidden_lys' : [50, 200, 100], # Hidden layers of the MLP.
        'outputSize' : 1               # Output size of the MLP.
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

    checkpoint_filename = 'diffusion_nonhomogeneous_MLP.pth'
    diffusion_pinn = DiffusionNonhomogeneousPinn(
        model_class         = MLP,                   # Model class for the PINN.
        model_kwargs        = model_kwargs,          # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,         # Domain parameters.
        optimizer_class     = optimizer_class,       # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,      # Optimizer parameters.
        epochs              = 50,                    # Number of epochs for training.
        patience            = 10,                    # Patience for early stopping.
        sampling_fn         = sample_square_uniform, # Sampling function.
        checkpoint_filename = checkpoint_filename,   # Filename for the checkpoints.
    )

    # Train the model.
    #diffusion_pinn.train()

    # Load the complete model.
    diffusion_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)        # Print model information.
    
    # Plot the loss and the solution.
    plot_loss(
        model_instance = diffusion_pinn,
        filename       = "loss_plot.png"
    )

    # Plot the solution with the best model.
    diffusion_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = diffusion_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.png",
        time_dependent = True,
        adjust_zlim    = True
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance = diffusion_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.png",
        time_dependent = True 
    )