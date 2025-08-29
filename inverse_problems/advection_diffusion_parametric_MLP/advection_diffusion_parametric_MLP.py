"""
advection_diffusion_parametric_MLP.py
-------------------------------------
Physics-Informed Neural Network (PINN) for the 1D Advection-Diffusion Equation
with Parametric Coefficients.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the 1D advection-diffusion equation on the space-time domain 
$[0,L]\times[0,T]$ using a Physics-Informed Neural Network (PINN) with Dirichlet
boundary conditions and an initial condition. The advection coefficient $\beta$,
thermal diffusivity $\alpha$, and mode number $n$ are treated as parameters.

Governing PDE:
$$
    u_{t}(x,t) - \alpha u_{xx}(x,t) + \beta u_{x}(x,t) = 0, 
    \quad (x,t)\in(0,L)\times(0,T).
$$

Boundary conditions:
$$
    u(0,t) = u(L,t) = 0, \quad t\in[0,T].
$$

Initial condition:
$$
    u(x,0) = \sin\left(frac{n \pi x}{L}\right) 
             \exp\left(frac{\beta x}{2\alpha}\right),  \quad x\in[0,L].
$$

Analytical solution:
$$
    u(x,t) = \sin\left(frac{n \pi x}{L}\right)
             \exp\left(-t\Big(\tfrac{\alpha n^{2}\pi^{2}}{L^{2}}
             + \tfrac{\beta^{2}}{4\alpha}\Big)\right)
             \exp\left(\tfrac{\beta x}{2\alpha}\right).
$$

Implementation
--------------
- Class `AdvectionDiffusionPinn` inheriting from `PinnCore`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE, boundary, and initial condition residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the advection-diffusion operator 
    (via automatic differentiation).
  - Initial condition residual $L_\mathrm{ic}$ enforcing $u(x,0)$.
  - Boundary residual $L_\mathrm{bc}$ enforcing $u(0,t)=u(L,t)=0$.
- Parameters $n$, $\alpha$, and $\beta$ are included as network inputs, 
  enabling parametric dependence.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,t)$ domain for different $(\alpha,\beta)$ values.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python advection_diffusion_parametric_MLP.py

Example instantiation:
>>> advection_diffusion_pinn = AdvectionDiffusionPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=2000,
...     patience=200,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="advection_diffusion_parametric_MLP.pth"
... )
>>> advection_diffusion_pinn.train()

To load and visualize:
>>> advection_diffusion_pinn.load_model(load_best=True)
>>> plot_solution_square(
...     advection_diffusion_pinn,
...     domain_kwargs,
...     "solution.png",
...     time_dependent=True,
...     parameters=[n, 0.06, 0.0]
... )
>>> plot_comparison_contour_square(
...     advection_diffusion_pinn,
...     domain_kwargs,
...     "comparison.png",
...     time_dependent=True,
...     parameters=[n, 0.06, 0.0]
... )

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the rectangular domain $[0,L]\times[0,T]$.
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

class AdvectionDiffusionPinn(PinnCore):
    def __init__(self, **params):
        """
        Initializes the AdvectionDiffusionPinn instance using the configuration
        dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnCore class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(AdvectionDiffusionPinn, self).__init__(**params)
        self.L = self.domain_kwargs["dim1_max"]  # Length of the domain in x.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x, t)$ evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 5), where each row corresponds to a 2D point
            $(x, t)$ in the domain and the frequency $n$, thermal diffusivity
            $\alpha$, and advection coefficient $\beta$.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        x = X[:, 0]
        t = X[:, 1]
        n = X[:, 2]
        alpha = X[:, 3]
        beta = X[:, 4]
    
        return (
            torch.exp(
                - t * (
                    (alpha * n**2 * torch.pi**2)/(self.L**2) + (beta**2)/(4*alpha)
                )
            )
            * torch.exp((beta * x)/(2 * alpha))
            * torch.sin((n * torch.pi * x) / self.L)
        )
    
    def loss_PINN(
            self, net: Callable, X: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE
        residual loss, the initial condition loss, and the boundary condition
        loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution
                $\boldsymbol{\hat{u}}_{w}(x, t; n, \alpha)$.
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
            Dictionary containing individual loss components 'loss_pde', 'loss_bc'
            and 'loss_ic'.

        Notes
        -----
        In this example, additional conditions are not used, so the loss is computed
        from the PDE residual, initial and boundary conditions.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0  # λ_pde.
        lb_ic = 1.0  # λ_ic.
        lb_bc = 1.0  # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc = (
            self.domain_kwargs["dim1_minSize"] + self.domain_kwargs["dim1_maxSize"]
            )
        N_ic = self.domain_kwargs["dim2_minSize"]

        # Split the input tensor X into the different regions.
        X_pde = X[0:N_pde]  # PDE collocation.
        X_bc = X[N_pde:N_pde+N_bc]  # Boundary.
        X_ic  = X[-N_ic:]  # Initial condition.

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => u_t - α u_xx + β u_x = 0.
        # --------------------------------------------------------------------------
        # Model output for the PDE points.
        u_pde = net(X_pde)

        # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂t.
        grad = torch.autograd.grad(
            u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True
        )[0]
        u_x, u_t = grad[:, 0], grad[:, 1]

        # ∂²u/∂x².
        u_xx = torch.autograd.grad(
            u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0]

        # Extract parameters from input.
        alpha_pde = X_pde[:, 3]
        beta_pde = X_pde[:, 4]

        # PDE residual loss.
        loss_pde = torch.mean((u_t - alpha_pde * u_xx + beta_pde * u_x)**2)

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => u(0,t) = u(L,t) = 0.
        # --------------------------------------------------------------------------
        loss_bc = torch.mean(net(X_bc)**2)

        # --------------------------------------------------------------------------
        # Initial condition loss: u0(x) = sin(n * π * x / L) * exp((β * x)/(2 * α)).
        # --------------------------------------------------------------------------
        # Model output for the initial condition points.
        u_ic = net(X_ic)

        # Extract parameters from input.
        x_ic = X_ic[:, 0]
        n_ic = X_ic[:, 2]
        alpha_ic = X_ic[:, 3]
        beta_ic = X_ic[:, 4]
        u0 = (
            torch.exp((beta_ic * x_ic) / (2 * alpha_ic))
            * torch.sin((n_ic * torch.pi * x_ic) / self.L)
        )

        # Boundary loss.
        loss_ic = torch.mean((u_ic.squeeze() - u0)**2)

        # --------------------------------------------------------------------------
        # PINN loss: λ_pde * L_pde + λ_ic * L_ic + λ_bc * L_bc.
        # --------------------------------------------------------------------------
        loss_PINN = lb_pde * loss_pde + lb_ic * loss_ic + lb_bc * loss_bc

        return loss_PINN, {
            "loss_pde": loss_pde,
            "loss_ic": loss_ic,
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
    L, T, n = 1, 1, 2  # Domain parameters and n.

    domain_kwargs = {
        # Domain parameters.
        'dim1_min': 0.,
        'dim1_max': L,  # x in [0, 1]
        'dim2_min': 0.,
        'dim2_max': T,  # t in [0, 1]
        # Collocation points.
        'interiorSize': 10000,
        'dim1_minSize': 4000,
        'dim1_maxSize': 4000,
        'dim2_minSize': 4000,
        'dim2_maxSize': 0,
        'valSize': 1500,
        # Parameters for the PINN.
        'fixed_params': [n],  # Fixed parameters (n)
        'param_domains': [(0.02, 0.12), (-0.15, 0.1)],  # α, β domains.
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 5,
        'hidden_lys': [100,50],
        'outputSize': 1,
        'activation': 'tanh',
        'dropout': 0.0,
        'normalization': True,  # Whether to apply layer normalization.
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,  # Learning rate.
        'max_iter': 90,
        'tolerance_grad': 1e-09,  # Tolerance for gradient norm.
        'tolerance_change': 1e-09,  # Tolerance for parameter change.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search strategy.
    }

    checkpoint_filename = "advection_diffusion_parametric_MLP.pth"
    advection_diffusion_pinn = AdvectionDiffusionPinn(
        model_class=MLP,  # Model class.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=2000,
        patience=200,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Checkpoint filename.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # advection_diffusion_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    advection_diffusion_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)

    # Plot the loss and the solution.
    plot_loss(
        model_instance=advection_diffusion_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    advection_diffusion_pinn.load_model(load_best=True)

    # ------------------------------------------------------------------------------
    # Graphs for ⍺ = 0.06 and β = 0.
    # ------------------------------------------------------------------------------
    alpha_test1 = 0.06
    beta_test1 = 0
    plot_solution_square(
        model_instance=advection_diffusion_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test1.png",
        time_dependent=True,
        parameters=[n, alpha_test1, beta_test1]
    )

    plot_comparison_contour_square(
        model_instance=advection_diffusion_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test1.png",
        time_dependent=True,
        parameters=[n, alpha_test1, beta_test1]
    )

    # ------------------------------------------------------------------------------
    # Graphs for ⍺ = 0.021 and β = -0.1.
    # ------------------------------------------------------------------------------
    alpha_test2 = 0.021
    beta_test2 = -0.1
    plot_solution_square(
        model_instance=advection_diffusion_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test2.png",
        time_dependent=True,
        parameters=[n, alpha_test2, beta_test2]
    )

    plot_comparison_contour_square(
        model_instance=advection_diffusion_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test2.png",
        time_dependent=True,
        parameters=[n, alpha_test2, beta_test2]
    )