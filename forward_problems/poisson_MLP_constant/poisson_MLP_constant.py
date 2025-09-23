"""
poisson_MLP_constant_source.py
------------------------------
Physics-Informed Neural Network (PINN) for the two-dimensional Poisson equation.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the Poisson equation on the unit square domain $[0,1]\times[0,2]$ 
using a Physics-Informed Neural Network (PINN) with homogeneous Dirichlet 
boundary conditions.

Governing PDE:
$$
    \Delta u(x,y) = 4, \quad (x,y)\in(0,1)\times(0,2).
$$

Boundary conditions:
$$
    u(0,y) = y^2,
    u(1,y) = 1+y^2,
    u(x,0) = x^2,
    u(x,2) = x^2 + 4.
$$

Analytical solution:
$$
    u(x,y) = x^2 + y^2.
$$

Implementation
--------------
- Class `PoissonCSPinn` inheriting from `PinnCore`.
- Physics-informed loss:
  - PDE residual $L_\mathrm{pde}$ from the Poisson operator (computed via 
    automatic differentiation).
  - Boundary residual $L_\mathrm{bc}$ enforcing Dirichlet constraints.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Loss history plots.
  - Contour and surface plots of the PINN prediction.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python poisson_MLP_constant_source.py

Example (inside the script):
>>> poissonCS_pinn = PoissonCSPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=150,
...     patience=10,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="poissonCS_MLP.pth"
... )
>>> poissonCS_pinn.train()

To load and visualize:
>>> poissonCS_pinn.load_model(load_best=True)
>>> plot_solution_square(poissonCS_pinn, domain_kwargs, "solution.png")
>>> plot_comparison_contour_square(poissonCS_pinn, domain_kwargs, "comparison.pdf")

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the unit square.
- No initial condition is required since the problem is elliptic and fully 
  determined by the PDE and boundary conditions.
"""
# Necessary libraries.
import os                                          # File paths.
import sys                                         # System functions.
import random                                      # Random numbers.
from typing import Callable                        # Type hints.
import numpy as np                                 # Arrays and math.
import torch                                       # Tensors and autograd.
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
from pinn_core import PinnCore       # Base class for PINNs.
from plotting import (               # Plotting functions.
    plot_loss, 
    plot_solution_square, 
    plot_comparison_contour_square
)
from utils import get_model_info     # Model info utility.

class PoissonCSPinn(PinnCore):
    def __init__(self, **params: dict):
        """
        Initializes the PoissonCSPinn instance using the configuration dictionary
        passed to the base class.
        
        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnCore class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(PoissonCSPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x,y) = x^2 + y^2$
        evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point
            ($x, y$) in the domain.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at
            each input point.
        """
        return X[:, 0]**2 + X[:, 1]**2
    
    def loss_PINN(
            self, net: Callable, X: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE
        residual loss and the boundary condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution
                $\boldsymbol{\hat{u}}_{w}(x, y)$.
        X : torch.Tensor
            Tensor of input points, where the first N_pde entries correspond to
            interior domain points and the rest to boundary points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current
            batch.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components 'loss_pde' and 'loss_bc'.

        Notes
        -----
        In this example, initial and additional conditions are not used, so the
        loss is computed only from the PDE residual and boundary conditions.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0  # λ_pde.
        lb_bc = 1.0  # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc_xmin = self.domain_kwargs["dim1_minSize"]
        N_bc_xmax = self.domain_kwargs["dim1_maxSize"]
        N_bc_ymin = self.domain_kwargs["dim2_minSize"]
        N_bc_ymax = self.domain_kwargs["dim2_maxSize"]

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde), 
            torch.ones(N_bc_xmin) * 2,
            torch.ones(N_bc_xmax) * 3,
            torch.ones(N_bc_ymin) * 4,
            torch.ones(N_bc_ymax) * 5
        ))

        # Loss components.
        loss_pde = torch.tensor(0.0, device=X.device)
        loss_bc = torch.tensor(0.0, device=X.device)

        for i in range(len(X)):
            xy = X[i, :].unsqueeze(0).requires_grad_(True)  # Input point (x, y).
            region = int(indicators[i].item())  # Region indicator.
            u = net(xy)  # Output of the net for the current input point.

            # ----------------------------------------------------------------------
            # PDE loss: N[u] = f => 𝚫u = 4.
            # ----------------------------------------------------------------------
            if region == 1:

                # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂y.
                grad_u = torch.autograd.grad(u, xy, create_graph=True)[0]
                u_x, u_y = grad_u[:, 0], grad_u[:, 1]
                
                # ∂²u/∂x².
                u_xx = torch.autograd.grad(u_x, xy, create_graph=True)[0][:, 0]

                # ∂²u/∂y².
                u_yy = torch.autograd.grad(u_y, xy, create_graph=True)[0][:, 1]
                
                # PDE residual loss.
                loss_pde += (u_xx + u_yy - 4).pow(2).squeeze()

            # ----------------------------------------------------------------------
            # Boundary loss: B[u] = g.
            # ----------------------------------------------------------------------
            elif region == 2: # u(0,y) = y^2.
                loss_bc += (u - (xy[0, 1]**2)).pow(2).squeeze()

            elif region == 3: # u(1,y) = 1 + y^2.
                loss_bc += (u - (1 + xy[0, 1]**2)).pow(2).squeeze()

            elif region == 4: # u(x,0) = x^2.
                loss_bc += (u - (xy[0, 0]**2)).pow(2).squeeze()

            elif region == 5: # u(x,2) = x^2 + 4.
                loss_bc += (u - (xy[0, 0]**2 + 4)).pow(2).squeeze()

        # Normalize each term.
        loss_pde /= N_pde
        loss_bc /= (N_bc_xmin + N_bc_xmax + N_bc_ymin + N_bc_ymax)

        # --------------------------------------------------------------------------
        # PINN loss: λ_pde * L_pde + λ_bc * L_bc.
        # --------------------------------------------------------------------------
        loss_PINN = lb_pde * loss_pde + lb_bc * loss_bc

        return loss_PINN, {
            "loss_pde": loss_pde,
            "loss_bc": loss_bc
            }

# ==================================================================================
# Main function.
# ==================================================================================
if __name__ == "__main__":

    from architectures import MLP               # Import the MLP architecture.
    from sampling import sample_square_uniform  # Uniform sampling in a square domain.

    # ------------------------------------------------------------------------------
    # Domain and model parameters.
    # ------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'dim1_min': 0.,
        'dim1_max': 1.,
        'dim2_min': 0.,
        'dim2_max': 2.,
        # Collocation points.
        'interiorSize': 200,
        'dim1_minSize': 250,
        'dim1_maxSize': 250,
        'dim2_minSize': 250,
        'dim2_maxSize': 250,
        'valSize': 300,
        # Parameters for the PINN.
        'fixed_params': None,
        'param_domains': None,
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 2,  # Because we do not have parameters.
        'hidden_lys': [100, 50, 50], 
        'outputSize': 1,            
        'activation': 'tanh',         
        'dropout': 0.0,            
        'normalization': True,  # Whether to apply layer normalization.
    }
    
    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,  # Learning rate.
        'max_iter': 100,
        'tolerance_grad': 1e-09,  # Tolerance for the gradient.
        'tolerance_change': 1e-09,  # Tolerance for the change in the loss.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search function.
    }

    checkpoint_filename = 'poissonCS_MLP.pth'
    poissonCS_pinn = PoissonCSPinn(
        model_class=MLP,  # Model class for the PINN.
        model_kwargs=model_kwargs,                
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=500,
        patience=50,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Filename for the checkpoints.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # Train the model.
    #poissonCS_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print information.
    poissonCS_pinn.load_model(load_best=False) 
    get_model_info(checkpoint_filename)
    
    # Plot the loss and the solution.
    plot_loss(
        model_instance=poissonCS_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    poissonCS_pinn.load_model(load_best=True)  # Load the best model.
    plot_solution_square(
        model_instance=poissonCS_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot.png"
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance=poissonCS_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot.png",
    )