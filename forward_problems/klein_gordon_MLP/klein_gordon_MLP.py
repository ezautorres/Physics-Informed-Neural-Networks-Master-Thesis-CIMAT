"""
klein_gordon_nonlinear_MLP.py
-------------------------------
Instance of a Physics-Informed Neural Network (PINN) applied to a Nonlinear Klein-Gordon Equation in 1D.

Author: Ezau Faridh Torres Torres.
Date: 6 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves a nonlinear Klein-Gordon equation using a Physics-Informed Neural Network (PINN). 
The problem is defined on the spatio-temporal domain [x,t] \in [-1,1] \times [0,1], subject to 
Dirichlet boundary conditions and an initial condition.

The PDE to be solved is:
    \frac{\partial^2 u}{\partial t^2} + \alpha \frac{\partial^2 u}{\partial x^2} + \beta u + \gamma u^k = -x \cos(t) + x^2 \cos^2(t).

Subject to:
    - Initial condition: u(x,0) = x, ∂u/∂t(x,0) = 0.
    - Boundary conditions: u(-1,t) = -cos(t), u(1,t) = cos(t).

where \alpha = -1, beta = 0, gamma = 1, k = 2. The exact analytical solution is:
    u(x,t) = x cos(t).

This implementation includes:
    - A custom class 'KleinGordonPinn' inheriting from a general PINN base class.
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

class KleinGordonPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the KleinGordonPinn instance using the configuration dictionary passed to
        the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including model configuration,
            optimizer settings, and domain sampling specifications.
        """
        super(KleinGordonPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution u(x,t) = x cos(t) evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point (x,t) in the domain.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at each input point.
        """
        return X[:,0] * torch.cos(X[:,1])

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss, the initial
        and the boundary condition loss.

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
        In this example, additional conditions are not used, so the loss is computed
        from the PDE residual, initial and boundary conditions.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0 # lb_pde.
        lb_ic  = 1.5 # λ_ic.
        lb_bc  = 1.5 # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde   = self.domain_kwargs["interiorSize"]
        N_bc_l  = self.domain_kwargs["dim1_minSize"]
        N_bc_r  = self.domain_kwargs["dim1_maxSize"]
        N_ic    = self.domain_kwargs["dim2_minSize"]
        N_total = N_pde + N_bc_l + N_bc_r + N_ic

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde),      # Interior points [-1,1]x[0,2].
            torch.ones(N_bc_l) * 2, # Boundary points at x = -1.
            torch.ones(N_bc_r) * 3, # Boundary points at x = 1.
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
            # PDE loss: N[u] = f => ∂²u/∂t² + ⍺ ∂²u/∂x² + β u + γ u^k = f.
            # -----------------------------------------------------------------------------------------------
            if region == 1:
                grad_u = torch.autograd.grad(u, xt, grad_outputs = torch.ones_like(u), create_graph = True)[0]        # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂t. 
                u_x, u_t = grad_u[:,0], grad_u[:,1]                                                                   # ∂u/∂x and ∂u/∂t.
                u_xx = torch.autograd.grad(u_x, xt, grad_outputs = torch.ones_like(u_x), create_graph = True)[0][:,0] # ∂²u/∂x².
                u_tt = torch.autograd.grad(u_t, xt, grad_outputs = torch.ones_like(u_t), create_graph = True)[0][:,1] # ∂²u/∂t².
                f = -xt[0,0] * torch.cos(xt[0,1]) + xt[0,0]**2 * torch.cos(xt[0,1])**2                                # Source term: -x cos(t) + x² cos²(t).
                alpha, beta, gamma, k = -1, 0, 1, 2                                                                   # PDE parameters.
                
                loss_pde += (u_tt + alpha * u_xx + beta * u + gamma * u**k - f).pow(2).squeeze() # PDE residual loss.

            # -----------------------------------------------------------------------------------------------
            # Boundary condition loss: B[u] = g => u(-1,t) = -cos(t), u(1,t) = cos(t).
            # -----------------------------------------------------------------------------------------------
            elif region in [2,3]:
                g = torch.cos(xt[0,1]) if region == 3 else -torch.cos(xt[0,1]) # Boundary condition value.
                loss_bc += (u - g).pow(2).squeeze()                            # Compute the boundary loss.

            # -----------------------------------------------------------------------------------------------
            # Initial condition loss: u0(x) = x, ∂u/∂t(x,0) = 0.
            # -----------------------------------------------------------------------------------------------
            elif region == 4:
                u_t = torch.autograd.grad(u, xt, grad_outputs = torch.ones_like(u), create_graph = True)[0][:,1] # ∂u/∂t.
                loss_ic += (u - xt[0,0]).pow(2).squeeze() + u_t.pow(2).squeeze()                                 # Compute the initial condition loss.

        # Normalize each term.
        loss_pde /= N_pde
        loss_ic /= N_ic
        loss_bc /= (N_bc_l + N_bc_r)

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
        'dim1_max'      :  1.,
        'dim2_min'      :  0.,
        'dim2_max'      :  3.,
        # Collocation points.
        'interiorSize'  : 250,
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
        'inputSize'  : 2,      # Because we do not have parameters.
        'hidden_lys' : [75]*4, # Hidden layers of the MLP.
        'outputSize' : 1       # Output size.
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

    checkpoint_filename = "klein-gordon_MLP.pth"
    kg_pinn = KleinGordonPinn(
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
    #kg_pinn.train()

    # Load the complete model.
    kg_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)   # Print model information.

    # Plot the loss and the solution.
    plot_loss(
        model_instance = kg_pinn,
        filename       = "loss_plot.png"
    )

    # Plot the loss and the solution.
    kg_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = kg_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.png",
        time_dependent = True,
        adjust_zlim    = True
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance = kg_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.png",
        time_dependent = True
    )