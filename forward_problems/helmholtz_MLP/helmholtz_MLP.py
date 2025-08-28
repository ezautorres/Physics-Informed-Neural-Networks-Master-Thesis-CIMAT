"""
helmholtz_nonhomogeneous_MLP.py
-------------------------------
Physics-Informed Neural Network (PINN) for the 2D Nonhomogeneous Helmholtz
Equation.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves a nonhomogeneous Helmholtz equation on the unit square domain 
$[0,1]\times[0,1]$ using a Physics-Informed Neural Network (PINN) with
homogeneous Dirichlet boundary conditions. The equation depends on a wave
number parameter $k$.

Governing PDE:
$$
    -\Delta u(x,y;k) - k^{2}u(x,y;k) = f(x,y;k), 
    \quad (x,y)\in(0,1)\times(0,1),
$$
with source term
$$
    f(x,y;k) = k^{2}\sin(kx)\sin(ky).
$$

Boundary conditions:
$$
    u(x,0;k) = u(x,1;k) = u(0,y;k) = u(1,y;k) = 0.
$$

Analytical solution:
$$
    u(x,y;k) = \sin(kx)\sin(ky), \quad k \in \pi\mathbb{Z}.
$$

Implementation
--------------
- Class `HelmholtzNonhomogeneousPinn` inheriting from `PinnBase`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE and boundary residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the Helmholtz operator 
    (via automatic differentiation).
  - Boundary residual $L_\mathrm{bc}$ enforcing homogeneous Dirichlet conditions.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,y)$ domain.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python helmholtz_nonhomogeneous_MLP.py

Example instantiation:
>>> helmholtz_pinn = HelmholtzNonhomogeneousPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=120,
...     patience=15,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="helmholtz_nonhomogeneous_MLP.pth"
... )
>>> helmholtz_pinn.train()

To load and visualize:
>>> helmholtz_pinn.load_model(load_best=True)
>>> plot_solution_square(
...     helmholtz_pinn, domain_kwargs, "solution.png", parameters=[k]
... )
>>> plot_comparison_contour_square(
...     helmholtz_pinn, domain_kwargs, "comparison.png", parameters=[k]
... )

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the unit square.
- Parameter $k$ must be a multiple of $\pi$ to match the analytical solution.
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
from pinn_base import PinnBase       # Base class for PINNs.
from plotting import (               # Plotting functions.
    plot_loss, 
    plot_solution_square, 
    plot_comparison_contour_square
)
from utils import get_model_info     # Model info utility.

class HelmholtzNonhomogeneousPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the HelmholtzNonhomogeneousPinn instance using the
        configuration dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(HelmholtzNonhomogeneousPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x, y; k) = \sin(kx) * \sin(ky)$
        evaluated at input points X, $k$ must be a multiple of $\pi$ (default
        $k = 3 * \pi$).

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point in
            the domain and the third column contains the wave number $k$.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        return torch.sin(X[:, 2] * X[:, 0]) * torch.sin(X[:, 2] * X[:, 1])

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
                $\boldsymbol{\hat{u}}_{w}(x, y; k)$.
        X : torch.Tensor
            Tensor of input points, where the first N_pde entries correspond
            to interior domain points and the rest to boundary points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current
            batch.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components 'loss_pde' and
            'loss_bc'.

        Notes
        -----
        In this example, initial and additional conditions are not used, so the loss
        is computed only from the PDE residual and boundary conditions.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0  # λ_pde.
        lb_bc = 1.0  # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc = len(X) - N_pde

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde),  # Interior points [0,1] x [0,1].
            torch.ones(N_bc) * 2  # Boundary points (x,y)ϵ ∂Ω.
        )).to(X.device)

        # Loss components.
        loss_pde = torch.tensor(0.0, device=X.device)
        loss_bc = torch.tensor(0.0, device=X.device)

        # Wave number.
        k = X[0, 2]

        for i in range(len(X)):
            x = X[i, :].unsqueeze(0).requires_grad_(True)  # Input point (x, y).
            region = int(indicators[i].item())  # Region indicator.
            u = net(x)  # Output of the network.

            # ----------------------------------------------------------------------
            # PDE loss: N[u] = f => -𝚫u - ku = k²sin(kx)sin(ky).
            # ----------------------------------------------------------------------
            if region == 1:

                # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂t. 
                grad_u = torch.autograd.grad(
                    u, x, grad_outputs=torch.ones_like(u), create_graph=True
                )[0]
                u_x, u_y  = grad_u[:, 0], grad_u[:, 1]

                # ∂²u/∂x².
                u_xx = torch.autograd.grad(
                    u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
                )[0][:, 0]

                # ∂²u/∂y².
                u_yy = torch.autograd.grad(
                    u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True
                )[0][:, 1]

                # Source term: k² sin(kx) sin(ky).
                f = k**2 * torch.sin(k * x[0, 0]) * torch.sin(k * x[0, 1])

                # PDE residual loss.
                loss_pde += (-u_xx - u_yy - k**2 * u - f).pow(2).squeeze() 

            # ----------------------------------------------------------------------
            # Boundary loss: B[u] = g => u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0.
            # ----------------------------------------------------------------------
            elif region == 2:
                loss_bc += u.pow(2).squeeze()

        # Normalize each term.
        loss_pde /= N_pde
        loss_bc /= N_bc

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

    from architectures import MLP              # Import the MLP architecture.
    from sampling import sample_square_uniform # Uniform sampling in a square domain.

    k = 3 * torch.pi # Wave number.

    # ------------------------------------------------------------------------------
    # Domain and model parameters.
    # ------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'dim1_min': 0.,
        'dim1_max': 1.,
        'dim2_min': 0.,
        'dim2_max': 1.,
        # Collocation points.
        'interiorSize': 1200,
        'dim1_minSize': 400,
        'dim1_maxSize': 400,
        'dim2_minSize': 400,
        'dim2_maxSize': 400,
        'valSize': 500,
        # Parameters for the PINN.
        'fixed_params': [k],  # Wave number.
        'param_domains': None,
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 3,  # 2 spatial dimensions (x, y) and k.
        'hidden_lys': [100, 120, 75, 50],
        'outputSize': 1,            
        'activation': 'tanh',
        'dropout': 0.0,
        'normalization': True,
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,  # Learning rate.
        'max_iter': 50,
        'tolerance_grad': 1e-09,  # Tolerance for the gradient.
        'tolerance_change': 1e-09,  # Tolerance for the change in the loss.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search function.
    }

    checkpoint_filename = 'helmholtz_nonhomogeneous_MLP.pth'
    helmholtz_pinn = HelmholtzNonhomogeneousPinn(
        model_class=MLP,  # Model class for the PINN.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=120,
        patience=15,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Checkpoint filename.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # helmholtz_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print information.
    helmholtz_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)       

    # Plot the loss and the solution.
    plot_loss(
        model_instance=helmholtz_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    helmholtz_pinn.load_model(load_best=True)  # Load the best model.
    plot_solution_square(
        model_instance=helmholtz_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot.png",
        parameters=[k]
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance=helmholtz_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot.png",
        parameters=[k],
        adjust_scale=True
    )