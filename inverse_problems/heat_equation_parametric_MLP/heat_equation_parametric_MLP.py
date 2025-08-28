"""
heat_equation_parametric_MLP.py
----------------------
Physics-Informed Neural Network (PINN) for the 1D Heat Equation with Parametric
Diffusivity.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the 1D heat equation on the space-time domain $[0,L]\times[0,T]$ using
a Physics-Informed Neural Network (PINN) with Dirichlet boundary conditions
and an initial condition. The thermal diffusivity $\alpha$ and mode number $n$
are treated as parameters, $n$ is fixed and $\alpha$ is sampled uniformly.

Governing PDE:
$$
    u_{t}(x,t) - \alpha u_{xx}(x,t) = 0, \quad (x,t)\in(0,L)\times(0,T).
$$

Boundary conditions:
$$
    u(0,t) = u(L,t) = 0, \quad t\in[0,T].
$$

Initial condition:
$$
    u(x,0) = \sin\left(\frac{n \pi x}{L}\right), \quad x\in[0,L].
$$

Analytical solution:
$$
    u(x,t) = \sin\left(\frac{n \pi x}{L}\right)
             \exp\left(-\frac{\alpha n^{2}\pi^{2}}{L^{2}}t\right).
$$

Implementation
--------------
- Class `HeatEquationPinn` inheriting from `PinnBase`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE, boundary, and initial condition residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the heat operator 
    (via automatic differentiation).
  - Initial condition residual $L_\mathrm{ic}$ enforcing $u(x,0)$.
  - Boundary residual $L_\mathrm{bc}$ enforcing $u(0,t)=u(L,t)=0$.
- Parameters $n$ and $\alpha$ are included as network inputs, enabling 
  parametric dependence.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,t)$ domain for different $\alpha$ values.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python heat_parametric_MLP.py

Example instantiation:
>>> heat_pinn = HeatEquationPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=1500,
...     patience=150,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="heat_parametric_MLP.pth"
... )
>>> heat_pinn.train()

To load and visualize:
>>> heat_pinn.load_model(load_best=True)
>>> plot_solution_square(
...    heat_pinn,
...    domain_kwargs,
...    "solution_test.png",
...    time_dependent=True,
...    parameters=[n, 0.05]
... )
>>> plot_comparison_contour_square(
...    heat_pinn,
...    domain_kwargs,
...    "comparison_test.png",
...    time_dependent=True,
...    parameters=[n, 0.05]
... )

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the rectangular domain $[0,L]\times[0,T]$.
- Parametric dependence allows evaluation at unseen values of diffusivity $\alpha$.
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
class HeatEquationPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the HeatEquationPinn instance using the configuration
        dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(HeatEquationPinn, self).__init__(**params)
        self.L = self.domain_kwargs["dim1_max"]  # Length of the domain in x.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution
            $u(x, t) = \sin(n \pi x / L) e^{-\alpha n^2 \pi^2 t / L^2}$
        evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point
            $(x, t)$ in the domain and the frequency $n$ and thermal diffusivity
            $\alpha$.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        # Extract variables from input tensor
        x = X[:, 0]
        t = X[:, 1]
        n = X[:, 2]
        alpha = X[:, 3]

        return (
            torch.exp(-(n**2 * torch.pi**2 * alpha * t) / (self.L**2))
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
        X_pde = X[0:N_pde]  # PDE collocation points.
        X_bc = X[N_pde:N_pde+N_bc]  # Boundary points.
        X_ic  = X[-N_ic:]  # Initial condition points.

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => u_t - α u_xx = 0.
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

        # Extract α from input.
        alpha = X_pde[:, 3]

        # PDE residual loss.
        loss_pde = torch.mean((u_t - alpha * u_xx)**2)

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => u(0,t) = u(L,t) = 0.
        # --------------------------------------------------------------------------
        loss_bc = torch.mean(net(X_bc)**2)

        # --------------------------------------------------------------------------
        # Initial condition loss: u0(x) = sin(n * π * x / L)
        # --------------------------------------------------------------------------
        # Model output for the initial condition points.
        u_ic = net(X_ic)
        u0 = torch.sin((X_ic[:, 2] * torch.pi * X_ic[:, 0]) / self.L)

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

    L, T, n = 2, 2, 5  # Domain parameters and the n-th sine mode.

    # ------------------------------------------------------------------------------
    # Domain and model parameters.
    # ------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'dim1_min': 0.,
        'dim1_max': L,  # x in [0, L].
        'dim2_min': 0.,
        'dim2_max': T,  # t in [0, T].
        # Collocation points.
        'interiorSize': 15000,
        'dim1_minSize': 2000,
        'dim1_maxSize': 2000,
        'dim2_minSize': 2000,
        'dim2_maxSize': 0,
        'valSize': 1800,
        # Parameters for the PINN.
        'fixed_params': [n],  # n-th sine mode.
        'param_domains': [(0, 0.1)],  # α domain.
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 4,
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
        'tolerance_grad': 1e-09,  # Tolerance for gradient.
        'tolerance_change': 1e-09,  # Tolerance for change in the loss.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search strategy.
    }

    checkpoint_filename = "heat_parametric_MLP.pth"
    heat_pinn = HeatEquationPinn(
        model_class=MLP,  # Model class.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=1500,
        patience=150,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Filename for the checkpoints.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # heat_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    heat_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)

    # Plot the loss and the solution.
    plot_loss(
        model_instance=heat_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    heat_pinn.load_model(load_best=True)

    # ------------------------------------------------------------------------------
    # Graphs for ⍺ = 0.05.
    # ------------------------------------------------------------------------------
    alpha_test1 = 0.05
    plot_solution_square(
        model_instance=heat_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test1.png",
        time_dependent=True,
        parameters=[n, alpha_test1]
    )

    plot_comparison_contour_square(
        model_instance=heat_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test1.png",
        time_dependent=True,
        parameters=[n, alpha_test1]
    )

    # ------------------------------------------------------------------------------
    # Graph for ⍺ = 0.021.
    # ------------------------------------------------------------------------------
    alpha_test2 = 0.021
    plot_solution_square(
        model_instance=heat_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot_test2.png",
        time_dependent=True,
        parameters=[n, alpha_test2]
    )

    plot_comparison_contour_square(
        model_instance=heat_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot_test2.png",
        time_dependent=True,
        parameters=[n, alpha_test2]
    )