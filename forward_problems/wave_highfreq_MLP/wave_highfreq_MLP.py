"""
wave_highfreq_MLP.py
--------------------
Physics-Informed Neural Network (PINN) for the 1D Wave Equation with
High-Frequency Modes.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the 1D wave equation on the space-time domain 
$[0,1]\times[0,1]$ using a Physics-Informed Neural Network (PINN) with 
Dirichlet boundary conditions and initial conditions. The wave speed $c$ 
is treated as a fixed parameter.

Governing PDE:
$$
    u_{tt}(x,t) - c^{2}u_{xx}(x,t) = 0, \quad (x,t)\in(0,1)\times(0,1).
$$

Boundary conditions:
$$
    u(0,t) = u(1,t) = 0, \quad t\in[0,1].
$$

Initial conditions:
$$
    u(x,0) = \sin(\pi x) + \sin(2\pi x), \quad 
    \frac{\partial u}{\partial t}(x,0) = 0, \quad x\in[0,1].
$$

Analytical solution:
$$
    u(x,t) = \sin(\pi x)\cos(c\pi t) + \sin(2\pi x)\cos(2c\pi t).
$$

Implementation
--------------
- Class `WavePinn` inheriting from `PinnBase`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE, boundary, and initial condition residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the wave operator 
    (via automatic differentiation).
  - Initial condition residual $L_\mathrm{ic}$ enforcing $u(x,0)$ and $u_t(x,0)=0$.
  - Boundary residual $L_\mathrm{bc}$ enforcing $u(0,t)=u(1,t)=0$.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,t)$ domain.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python wave_highfreq_MLP.py

Example instantiation:
>>> wave_pinn = WavePinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=15000,
...     patience=1000,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="wave_highfreq_MLP.pth"
... )
>>> wave_pinn.train()

To load and visualize:
>>> wave_pinn.load_model(load_best=True)
>>> plot_solution_square(wave_pinn, domain_kwargs, "solution.png", time_dependent=True)
>>> plot_comparison_contour_square(wave_pinn, domain_kwargs, "comparison.png", time_dependent=True)

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the rectangular domain $[0,1]\times[0,1]$.
- Multiple frequency modes are captured through the choice of initial conditions.
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

class WavePinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the WavePinn instance using the configuration dictionary
        passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(WavePinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution
            $u(x, t) = \sin(\pi x) \cos(c \pi t) + \sin(2 \pi x) \cos(2 c \pi t)$
        evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 3), where each row corresponds to a 2D point
            $(x, t)$ in the domain and the frequency $c$.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        x = X[:,0]
        t = X[:,1]
        c = X[:,2]
        return (
            torch.sin(np.pi * x)
            * torch.cos(c * np.pi * t)
            + torch.sin(2 * np.pi * x)
            * torch.cos(2 * c * np.pi * t)
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
                $\boldsymbol{\hat{u}}_{w}(x, t)$.
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
        lb_pde = 2 # λ_pde.
        lb_ic = 1 # λ_ic.
        lb_bc = 1 # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc = (
            self.domain_kwargs["dim1_minSize"] + self.domain_kwargs["dim1_maxSize"]
        )
        N_ic = self.domain_kwargs["dim2_minSize"]

        # Split the input tensor X into the different regions.
        X_pde = X[:N_pde]  # PDE collocation points.
        X_bc = X[N_pde:N_pde+N_bc]  # Boundary points.
        X_ic = X[-N_ic:]  # Initial condition points.

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => u_tt - c² u_xx = 0.
        # --------------------------------------------------------------------------
        # Model output for the PDE points.
        u_pde = net(X_pde)

        # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂t.
        grad_u = torch.autograd.grad(
            u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True
        )[0]
        u_x, u_t = grad_u[:,0], grad_u[:,1]       
        
        # ∂²u/∂x².
        u_xx = torch.autograd.grad(
            u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0]

        # ∂²u/∂t².
        u_tt = torch.autograd.grad(
            u_t, X_pde, grad_outputs=torch.ones_like(u_t), create_graph=True
        )[0][:, 1]

        # Wave speed (constant).
        c = X[0, 2]

        # PDE residual loss.
        loss_pde = torch.mean((u_tt - c**2 * u_xx)**2)

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => u(0,t) = u(1,t) = 0.
        # --------------------------------------------------------------------------
        loss_bc = torch.mean(net(X_bc)**2)

        # --------------------------------------------------------------------------
        # Initial condition loss: u0(x) = sin(πx) + sin(2πx), ∂u/∂t(x,0) = 0.
        # --------------------------------------------------------------------------
        # Model output for the initial condition points.
        u_ic = net(X_ic)

        # ∇u, grad_ic[:, 0] = ∂u/∂x, grad_ic[:, 1] = ∂u/∂t.
        grad_ic = torch.autograd.grad(
            u_ic, X_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True
        )[0]
        u_t_ic = grad_ic[:, 1]
        u0 = torch.sin(np.pi * X_ic[:, 0]) + torch.sin(2 * np.pi * X_ic[:, 0])

        # Boundary loss.
        loss_ic = torch.mean((u_ic.squeeze() - u0)**2) + torch.mean(u_t_ic**2)

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

    # Wave speed.
    c = 10
    
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
        'interiorSize': 1500,
        'dim1_minSize': 200,
        'dim1_maxSize': 200,
        'dim2_minSize': 500,
        'dim2_maxSize': 0,
        'valSize': 500,
        # Parameters for the PINN.
        'fixed_params': [c],  # Wave speed.
        'param_domains': None,
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 3,       
        'hidden_lys': [100]*6, 
        'outputSize': 1,
        'activation': 'tanh',
        'dropout': 0.0,
        'normalization': True,  # Whether to apply layer normalization.
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,  # Learning rate.
        'max_iter': 32,
        'tolerance_grad': 1e-09,  # Tolerance for the gradient.
        'tolerance_change': 1e-09,  # Tolerance for the change in the loss.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search function.
    }

    checkpoint_filename = "wave_highfreq_MLP.pth"
    wave_pinn = WavePinn(
        model_class=MLP,  # Model class for the PINN.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=15000,
        patience=1000,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Filename for the checkpoints.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # wave_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    wave_pinn.load_model(load_best = False)
    get_model_info(checkpoint_filename)    

    # Plot the loss and the solution.
    plot_loss(
        model_instance=wave_pinn, filename="loss_plot.png"
    )

    # Plot the solution with the best model.
    wave_pinn.load_model(load_best = True)  # Load the best model.
    plot_solution_square(
        model_instance=wave_pinn,
        domain_kwargs=domain_kwargs,
        parameters=[c],
        filename="solution_plot.png",
        time_dependent=True,
        adjust_zlim=True
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance=wave_pinn,
        domain_kwargs=domain_kwargs,
        parameters=[c],
        filename="comparison_plot.png",
        time_dependent=True,
    )