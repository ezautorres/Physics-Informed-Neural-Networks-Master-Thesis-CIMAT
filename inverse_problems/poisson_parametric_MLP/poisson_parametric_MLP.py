"""
poisson_parametric_MLP.py
-------------------------------------
Physics-Informed Neural Network (PINN) for the 2D Poisson Equation with
Parametric Coefficients.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the 2D Poisson equation on the spatial domain $[0,1]\times[0,1]$ using
a Physics-Informed Neural Network (PINN) with Dirichlet boundary conditions.
The source term $\alpha$ and the boundary slope / harmonic ramp coefficient
$\beta$ are treated as parameters.

Governing PDE:
$$
    \Delta u(x,y) = -\alpha \sin(\pi x) \sin(\pi y),
    \quad (x,y) \in(0,1)\times(0,1).
$$

Boundary conditions:
$$
    u(x,1) = \beta x, u(1,y) = \beta y, u(x,0) = u(0,y) = 0, \quad t \in[0,T].
$$

Analytical solution:
$$
    u(x,y) = \frac{\alpha}{2\pi^2} \sin(\pi x) \sin(\pi y) + \beta x y.
$$

Implementation
--------------
- Class `PoissonParametricPinn` inheriting from `PinnCore`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE, boundary, and initial condition residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the Poisson operator 
    (via automatic differentiation).
  - Boundary residual $L_\mathrm{bc}$.
- Parameters $\alpha$, and $\beta$ are included as network inputs,
  enabling parametric dependence.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,y)$ domain for different $(\alpha, \beta)$ values.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python poisson_parametric_MLP.py

Example instantiation:
>>> poisson_parametric_pinn = PoissonParametricPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=2000,
...     patience=200,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="poisson_parametric_MLP.pth"
... )
>>> poisson_parametric_pinn.train()

To load and visualize:
>>> poisson_parametric_pinn.load_model(load_best=True)
>>> plot_solution_square(
...     poisson_parametric_pinn,
...     domain_kwargs,
...     "solution.png",
...     time_dependent=True,
...     parameters=[-20.0, 3.0]
... )
>>> plot_comparison_contour_square(
...     poisson_parametric_pinn,
...     domain_kwargs,
...     "comparison.png",
...     time_dependent=True,
...     parameters=[-20.0, 3.0]
... )

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the rectangular domain $[0,1]\times[0,1]$.
- Parametric dependence allows evaluation at unseen $(\alpha,\beta)$ values.
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

class PoissonParametricPinn(PinnCore):
    def __init__(self, **params):
        """
        Initializes the PoissonParametricPinn instance using the configuration
        dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnCore class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(PoissonParametricPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x, y)$ evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point
            $(x, y)$ in the domain and the source term $\alpha$ and boundary slope
            coefficient $\beta$.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        x = X[:, 0]
        y = X[:, 1]
        alpha = X[:, 2]
        beta = X[:, 3]
    
        return (
            alpha / (2 * torch.pi**2)
            * torch.sin(torch.pi * x)
            * torch.sin(torch.pi * y)
            + beta * x * y
        )
    
    def loss_PINN2(
            self, net: Callable, X: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE
        residual loss and the boundary condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution
                $\boldsymbol{\hat{u}}_{w}(x, y; \alpha, \beta)$.
        X : torch.Tensor
            Tensor of input points, where the first N_pde entries correspond
            to interior domain points and the rest to initial and boundary
            points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current
            batch.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components 'loss_pde' and 'loss_bc'.

        Notes
        -----
        In this example, additional conditions are not used, so the loss is computed
        from the PDE residual and the boundary condition.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0  # λ_pde.
        lb_bc = 1.5  # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc_xmin = self.domain_kwargs["dim1_minSize"]
        N_bc_xmax = self.domain_kwargs["dim1_maxSize"]
        N_bc_ymin = self.domain_kwargs["dim2_minSize"]
        N_bc_ymax = self.domain_kwargs["dim2_maxSize"]

        # Split the input tensor X into the different regions.
        X_pde, X_bc_xmin, X_bc_xmax, X_bc_ymin, X_bc_ymax = torch.split(
            X, [N_pde, N_bc_xmin, N_bc_xmax, N_bc_ymin, N_bc_ymax], dim=0
        )

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => Δu = -α sin(πx) sin(πy).
        # --------------------------------------------------------------------------
        # Model output for the PDE points.
        u_pde = net(X_pde)

        # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂y.
        grad = torch.autograd.grad(
            u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True
        )[0]
        u_x, u_y = grad[:, 0], grad[:, 1]

        # ∂²u/∂x².
        u_xx = torch.autograd.grad(
            u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0]

        # ∂²u/∂y².
        u_yy = torch.autograd.grad(
            u_y, X_pde, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1]

        # Extract parameters from input.
        alpha_pde = X_pde[:, 2]

        # Source term.
        f = (
            - alpha_pde
            * torch.sin(torch.pi * X_pde[:, 0])
            * torch.sin(torch.pi * X_pde[:, 1])
        )

        # PDE residual loss.
        loss_pde = torch.mean((u_xx + u_yy - f)**2)

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g.
        # --------------------------------------------------------------------------
        # u(0, y) = 0.
        loss_bc_xmin = torch.mean(net(X_bc_xmin)**2)

        # u(1, y) = β * y.
        beta_bc_xmax = X_bc_xmax[:, 3]
        loss_bc_xmax = torch.mean(
            (net(X_bc_xmax) - beta_bc_xmax * X_bc_xmax[:, 1])**2
        )

        # u(x, 0) = 0.
        loss_bc_ymin = torch.mean(net(X_bc_ymin)**2)

        # u(x, 1) = β * x.
        beta_bc_ymax = X_bc_ymax[:, 3]
        loss_bc_ymax = torch.mean(
            (net(X_bc_ymax) - beta_bc_ymax * X_bc_ymax[:, 0])**2
        )  

        # Total boundary loss.
        loss_bc = loss_bc_xmin + loss_bc_xmax + loss_bc_ymin + loss_bc_ymax

        # --------------------------------------------------------------------------
        # PINN loss: λ_pde * L_pde + λ_bc * L_bc.
        # --------------------------------------------------------------------------
        loss_PINN = lb_pde * loss_pde + lb_bc * loss_bc

        return loss_PINN, {
            "loss_pde": loss_pde,
            "loss_bc": loss_bc
        }
    
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
                $\boldsymbol{\hat{u}}_{w}(x, y; \alpha, \beta)$.
        X : torch.Tensor
            Tensor of input points, where the first N_pde entries correspond
            to interior domain points and the rest to initial and boundary
            points.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current
            batch.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components 'loss_pde', 'loss_bc'.

        Notes
        -----
        In this example, additional conditions are not used, so the loss is computed
        from the PDE residual and the boundary condition.
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

        # Split the input tensor X into the different regions.
        X_pde, X_bc_xmin, X_bc_xmax, X_bc_ymin, X_bc_ymax = torch.split(
            X, [N_pde, N_bc_xmin, N_bc_xmax, N_bc_ymin, N_bc_ymax], dim=0
        )

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => -Δu = -α sin(πx) sin(πy).
        # --------------------------------------------------------------------------
        # Model output for the PDE points.
        u_pde = net(X_pde)

        # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂y.
        grad = torch.autograd.grad(
            u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True
        )[0]
        u_x, u_y = grad[:, 0], grad[:, 1]

        # ∂²u/∂x².
        u_xx = torch.autograd.grad(
            u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0]

        # ∂²u/∂y².
        u_yy = torch.autograd.grad(
            u_y, X_pde, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1]

        # Extract parameters from input.
        alpha_pde = X_pde[:, 2]

        # Source term.
        f = (
            - alpha_pde
            * torch.sin(torch.pi * X_pde[:, 0])
            * torch.sin(torch.pi * X_pde[:, 1])
        )

        # PDE residual loss.
        loss_pde = torch.mean((u_xx + u_yy - f)**2)

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g.
        # --------------------------------------------------------------------------
        # u(0, y) = 0.
        loss_bc_xmin = torch.mean(net(X_bc_xmin)**2)

        # u(1, y) = β * y.
        beta_bc_xmax = X_bc_xmax[:, 3]
        loss_bc_xmax = torch.mean(
            (net(X_bc_xmax) - beta_bc_xmax * X_bc_xmax[:, 1])**2
        )

        # u(x, 0) = 0.
        loss_bc_ymin = torch.mean(net(X_bc_ymin)**2)

        # u(x, 1) = β * x.
        beta_bc_ymax = X_bc_ymax[:, 3]
        loss_bc_ymax = torch.mean(
            (net(X_bc_ymax) - beta_bc_ymax * X_bc_ymax[:, 0])**2
        )  

        # Total boundary loss.
        loss_bc = loss_bc_xmin + loss_bc_xmax + loss_bc_ymin + loss_bc_ymax

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
        'dim2_max': 1.,
        # Collocation points.
        'interiorSize': 10000,
        'dim1_minSize': 4000,
        'dim1_maxSize': 4000,
        'dim2_minSize': 4000,
        'dim2_maxSize': 4000,
        'valSize': 1500,
        # Parameters for the PINN.
        'fixed_params': None,
        'param_domains': [(-25., 25.), (-5., 5.)],  # α, β domains.
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 4,
        'hidden_lys': [100, 70, 10],
        'outputSize': 1,
        'activation': 'tanh',
        'dropout': 0.0,
        'normalization': True,  # Whether to apply layer normalization.
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,  # Learning rate.
        'max_iter': 100,
        'tolerance_grad': 1e-09,  # Tolerance for gradient norm.
        'tolerance_change': 1e-09,  # Tolerance for parameter change.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search strategy.
    }

    checkpoint_filename = "poisson_parametric_MLP.pth"
    poisson_parametric_pinn = PoissonParametricPinn(
        model_class=MLP,  # Model class.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=3000,
        patience=300,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Checkpoint filename.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    poisson_parametric_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    poisson_parametric_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)

    # Plot the loss and the solution.
    plot_loss(
        model_instance=poisson_parametric_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    poisson_parametric_pinn.load_model(load_best=True)

    # ------------------------------------------------------------------------------
    # Graphs for ⍺ = 12.5 and β = 2.2.
    # ------------------------------------------------------------------------------
    alpha_test1 = 12.5
    beta_test1 = 2.2
    plot_solution_square(
        model_instance=poisson_parametric_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test1.png",
        time_dependent=True,
        parameters=[alpha_test1, beta_test1]
    )

    plot_comparison_contour_square(
        model_instance=poisson_parametric_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test1.png",
        time_dependent=True,
        parameters=[alpha_test1, beta_test1]
    )

    # ------------------------------------------------------------------------------
    # Graphs for ⍺ = -14.3 and β = 1.5.
    # ------------------------------------------------------------------------------
    alpha_test2 = -14.3
    beta_test2 = 1.5
    plot_solution_square(
        model_instance=poisson_parametric_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test2.png",
        time_dependent=True,
        parameters=[alpha_test2, beta_test2]
    )

    plot_comparison_contour_square(
        model_instance=poisson_parametric_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test2.png",
        time_dependent=True,
        parameters=[alpha_test2, beta_test2]
    )