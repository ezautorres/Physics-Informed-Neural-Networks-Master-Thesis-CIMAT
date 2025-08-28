"""
klein_gordon_MLP.py
-------------------
Physics-Informed Neural Network (PINN) for the 1D Nonlinear Klein-Gordon
Equation.

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Solves the nonlinear Klein-Gordon equation on the space-time domain 
$[-1,1]\times[0,3]$ using a Physics-Informed Neural Network (PINN) with
Dirichlet boundary conditions and initial conditions.

Governing PDE:
$$
    u_{tt}(x,t) + \alpha u_{xx}(x,t) + \beta u(x,t) + \gamma u(x,t)^{k} = f(x,t),
    \quad (x,t)\in(-1,1)\times(0,3),
$$
with parameters $\alpha=-1$, $\beta=0$, $\gamma=1$, $k=2$, and source term
$$
    f(x,t) = -x\cos(t) + x^{2}\cos^{2}(t).
$$

Boundary conditions:
$$
    u(-1,t) = -\cos(t), \quad u(1,t) = \cos(t), \quad t\in[0,3].
$$

Initial conditions:
$$
    u(x,0) = x, \quad \frac{\partial u}{\partial t}(x,0) = 0, \quad x\in[-1,1].
$$

Analytical solution:
$$
    u(x,t) = x\cos(t).
$$

Implementation
--------------
- Class `KleinGordonPinn` inheriting from `PinnBase`.
- Overrides:
  - `analytical_solution` returning the closed-form solution.
  - `loss_PINN` computing PDE, boundary, and initial condition residuals.
- Loss terms:
  - PDE residual $L_\mathrm{pde}$ from the Klein-Gordon operator 
    (via automatic differentiation).
  - Initial condition residual $L_\mathrm{ic}$ enforcing $u(x,0)=x$ and $u_t(x,0)=0$.
  - Boundary residual $L_\mathrm{bc}$ enforcing $u(-1,t)=-\cos(t)$ and $u(1,t)=\cos(t)$.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training loss curves.
  - Predicted solution plots in $(x,t)$ domain.
  - Comparison against the analytical solution.

Usage
-----
To train the model:
    $ python klein_gordon_MLP.py

Example instantiation:
>>> kg_pinn = KleinGordonPinn(
...     model_class=MLP,
...     model_kwargs=model_kwargs,
...     domain_kwargs=domain_kwargs,
...     optimizer_class=torch.optim.LBFGS,
...     optimizer_kwargs=optimizer_kwargs,
...     epochs=500,
...     patience=50,
...     sampling_fn=sample_square_uniform,
...     checkpoint_filename="klein-gordon_MLP.pth"
... )
>>> kg_pinn.train()

To load and visualize:
>>> kg_pinn.load_model(load_best=True)
>>> plot_solution_square(kg_pinn, domain_kwargs, "solution.png", time_dependent=True)
>>> plot_comparison_contour_square(kg_pinn, domain_kwargs, "comparison.png", time_dependent=True)

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation points sampled uniformly in the rectangular domain $[-1,1]\times[0,3]$.
- Both boundary and initial conditions are enforced explicitly in the loss.
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

class KleinGordonPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the KleinGordonPinn instance using the configuration
        dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(KleinGordonPinn, self).__init__(**params)

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x, t) = x \cos(t)$ evaluated at
        input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2), where each row corresponds to a 2D point
            $(x, t)$ in the domain.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        return X[:, 0] * torch.cos(X[:, 1])

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
        lb_pde = 1.0  # lb_pde.
        lb_ic = 1.5  # λ_ic.
        lb_bc = 1.5  # λ_bc.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc_l = self.domain_kwargs["dim1_minSize"]
        N_bc_r = self.domain_kwargs["dim1_maxSize"]
        N_ic = self.domain_kwargs["dim2_minSize"]
        N_total = N_pde + N_bc_l + N_bc_r + N_ic

        # Create indicators for each region.
        indicators = torch.cat((
            torch.ones(N_pde),  # Interior points [-1,1]x[0,3].
            torch.ones(N_bc_l) * 2,  # Boundary points at x = -1.
            torch.ones(N_bc_r) * 3,  # Boundary points at x = 1.
            torch.ones(N_ic)   * 4   # Initial condition points at t = 0.
        )).to(X.device)

        # Loss components.
        loss_pde = torch.tensor(0.0, device=X.device) 
        loss_ic = torch.tensor(0.0, device=X.device) 
        loss_bc = torch.tensor(0.0, device=X.device) 

        for i in range(N_total):
            xt = X[i, :].unsqueeze(0).requires_grad_(True)  # Input point (x, t).
            region = int(indicators[i].item())  # Region indicator.
            u = net(xt)  # Output of the network.

            # ----------------------------------------------------------------------
            # PDE loss: N[u] = f => ∂²u/∂t² + ⍺ ∂²u/∂x² + β u + γ u^k = f.
            # ----------------------------------------------------------------------
            if region == 1:

                # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂t. 
                grad_u = torch.autograd.grad(
                    u, xt, grad_outputs=torch.ones_like(u), create_graph=True
                )[0]    
                u_x, u_t = grad_u[:, 0], grad_u[:, 1]

                # ∂²u/∂x².
                u_xx = torch.autograd.grad(
                    u_x, xt, grad_outputs=torch.ones_like(u_x),create_graph=True
                )[0][:, 0]

                # ∂²u/∂t².
                u_tt = torch.autograd.grad(
                    u_t, xt, grad_outputs=torch.ones_like(u_t), create_graph=True
                )[0][:, 1]

                # Source term: -x cos(t) + x² cos²(t).
                f = (
                    - xt[0, 0] * torch.cos(xt[0, 1])
                    + xt[0, 0]**2 * torch.cos(xt[0, 1])**2
                )

                # PDE parameters.
                alpha, beta, gamma, k = -1, 0, 1, 2
                
                # PDE residual loss.
                loss_pde += (
                    u_tt + alpha * u_xx + beta * u + gamma * u**k - f
                ).pow(2).squeeze() 

            # ----------------------------------------------------------------------
            # Boundary loss: B[u] = g => u(-1,t) = -cos(t), u(1,t) = cos(t).
            # ----------------------------------------------------------------------
            elif region in [2, 3]:
                g = torch.cos(xt[0, 1]) if region == 3 else -torch.cos(xt[0, 1])

                # Boundary loss.
                loss_bc += (u - g).pow(2).squeeze()

            # ----------------------------------------------------------------------
            # Initial loss: u0(x) = x, ∂u/∂t(x,0) = 0.
            # ----------------------------------------------------------------------
            elif region == 4:
                
                # ∂u/∂t.
                u_t = torch.autograd.grad(
                    u, xt, grad_outputs=torch.ones_like(u), create_graph=True
                )[0][:, 1]

                # Initial condition loss.
                loss_ic += (u - xt[0, 0]).pow(2).squeeze() + u_t.pow(2).squeeze()

        # Normalize each term.
        loss_pde /= N_pde
        loss_ic /= N_ic
        loss_bc /= (N_bc_l + N_bc_r)

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
    domain_kwargs = {
        # Domain parameters.
        'dim1_min': -1.,
        'dim1_max': 1.,
        'dim2_min': 0.,
        'dim2_max': 3.,
        # Collocation points.
        'interiorSize': 230,
        'dim1_minSize': 100,
        'dim1_maxSize': 100,
        'dim2_minSize': 100,
        'dim2_maxSize': 0,
        'valSize': 180,
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
        'inputSize': 2,      
        'hidden_lys': [75]*4,
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

    checkpoint_filename = "klein-gordon_MLP.pth"
    kg_pinn = KleinGordonPinn(
        model_class=MLP,  # Model class for the PINN.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=75,
        patience=10,
        sampling_fn=sample_square_uniform,  # Sampling function.
        checkpoint_filename=checkpoint_filename,   # Filename for the checkpoints.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # Train the model.
    # kg_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    kg_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)

    # Plot the loss and the solution.
    plot_loss(
        model_instance=kg_pinn, filename="loss_plot.png"
    )

    # Plot the loss and the solution.
    kg_pinn.load_model(load_best=True)  # Load the best model.
    plot_solution_square(
        model_instance=kg_pinn,
        domain_kwargs=domain_kwargs,
        filename="solution_plot.png",
        time_dependent=True,
        adjust_zlim=True
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance=kg_pinn,
        domain_kwargs=domain_kwargs,
        filename="comparison_plot.png",
        time_dependent=True
    )