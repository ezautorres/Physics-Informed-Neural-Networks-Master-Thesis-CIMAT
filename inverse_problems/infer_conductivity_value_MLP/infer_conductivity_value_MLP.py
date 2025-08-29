"""
infer_conductivity_value_MLP.py
-------------------------------
Parametric Physics-Informed Neural Network (PINN) for the Conductivity
Equation in the Unit Disk (parameter \rho).

Author: Ezau Faridh Torres Torres.
Date: 25 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
Trains a Physics-Informed Neural Network (PINN) to approximate the solution of a
conductivity problem in the unit disk, where the conductivity depends on two
parameters: fixed $R$ (interface radius) and a variable $\rho$ (contrast). The
conductivity is piecewise defined as:

$$
\lambda(x) =
\begin{cases}
    1 + \rho, & |x| < R, \\\\
    1, & R < |x| < 1.
\end{cases}
$$

Governing PDE:
$$
    \nabla \cdot (\lambda \nabla u) = 0, \quad (x,y) \in B,
$$
where $B$ is the unit disk.

Boundary condition:
$$
    \lambda \frac{\partial u}{\partial n} = \cos(4\theta),
    \quad (x,y)\in\partial B.
$$

Additional condition (gauge):
$$
    \int_{\partial B} u \, ds = 0,
$$
to ensure uniqueness of the Neumann problem.

Analytical solution:
$$
    u(r,\theta) =
    \begin{cases}
        2(b+c)\,(r/R)^4 \cos(4\theta), & r < R, \\\\
        2\big(b\,(r/R)^4 + c\,(r/R)^{-4}\big)\cos(4\theta), & r \ge R,
    \end{cases}
$$
with coefficients $b$ and $c$ depending on $R$ and $\rho$.

Implementation
--------------
- Class `InferringConductivityValue` inheriting from `PinnCore`.
- Custom conductivity function `lambda_fn` defined in terms of $R$ and $\rho$.
- Physics-informed loss:
  - PDE residual $L_\mathrm{pde}$ enforcing $\nabla \cdot (\lambda \nabla u) = 0$.
  - Boundary residual $L_\mathrm{bc}$ enforcing Neumann boundary condition.
  - Gauge residual $L_\mathrm{add}$ enforcing uniqueness.
- Training with L-BFGS optimizer and strong Wolfe line search.
- Visualization utilities:
  - Training/validation loss curves.
  - Solution and contour plots on the circular domain.
  - Comparison with the analytical solution.

Usage
-----
To train the model:
    $ python infer_conductivity_value_MLP.py

Example:
>>> rho = 3.2
>>> plot_solution_circle(
...     model_instance=infer_rho_pinn,
...     domain_kwargs=domain_kwargs,
...     parameters=[R, rho],
...     filename="solution_plot.png"
... )

>>> plot_comparison_contour_circle(
...     model_instance=infer_rho_pinn,
...     domain_kwargs=domain_kwargs,
...     parameters=[R, rho],
...     filename="comparison_plot.png"
... )

Notes
-----
- Reproducibility ensured via fixed seeds (NumPy, Python, PyTorch).
- Collocation and auxiliary boundary points sampled uniformly on the disk.
- This script trains a **parametric PINN** where $(R, \rho)$ are treated as
  input parameters, enabling evaluation for different conductivity settings.
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
    plot_solution_circle, 
    plot_comparison_contour_circle
)
from utils import get_model_info     # Model info utility.

class InferringConductivityValue(PinnCore):
    def __init__(self, **params):
        """
        Initializes the InferringConductivityValue instance using the
        configuration dictionary passed to the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnCore class, including
            model configuration, optimizer settings, and domain sampling
            specifications.
        """
        # Initialize the PINN with parameters from the base class.
        super(InferringConductivityValue, self).__init__(**params)

    def lambda_fn(self, X: torch.Tensor) -> torch.Tensor:
        """
        Lambda function for the Unit Disk problem. This function modifies the
        lambda value depending on the radius of the point. If the radius is
        less than R, then lambda = 1 + rho, otherwise lambda = 1.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point
            ($x, y$) in the domain. The last two columns correspond to the
            parameters ($R$, $\rho$).

        Returns
        -------
        torch.Tensor
            Lambda values.
        """
        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)  # Radius.
        R = X[:, 2]
        rho = X[:, 3]
        lambda_vals = torch.ones_like(r)
        mask = r < R  # Mask where r < R.

        # Apply lambda modification only where the condition is met.
        lambda_vals[mask] += rho[mask]  

        return lambda_vals
    
    def calculate_coefs(self, rho: torch.Tensor, R: torch.Tensor) -> tuple:
        """
        Calculate the coefficients b and c for the analytical solution of the
        Unit Disk problem.

        Parameters
        ----------
        rho : torch.Tensor
            Parameter rho.
        R : torch.Tensor
            Parameter R.

        Returns
        -------
        tuple
            Coefficients b and c.
        """
        denom = 8 * (rho * R**8 + rho + 2)
        bn = ((rho + 2) * R**4) / denom  # Coefficient b.
        cn = - (rho * R**4) / denom  # Coefficient c.

        return bn, cn

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution $u(x, t)$ evaluated at input points X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point
            ($x, y$) in the domain. The last two columns correspond to the
            parameters ($R$, $\rho$).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated
            at each input point.
        """
        R = X[:, 2]
        rho = X[:, 3]
        b, c = self.calculate_coefs(rho, R)  # Coefficients b and c.
        r = torch.linalg.norm(X[:, 0:2], dim=1)  # Radius.
        theta = torch.atan2(X[:, 1], X[:, 0])  # Angle. 
    
        return torch.where(                         
            r < R,
            2 * (b + c) * (r / R)**4 * torch.cos(4 * theta),
            2 * (b * (r / R)**4 + c * (r / R)** -4) * torch.cos(4 * theta)
        )

    def loss_PINN(
            self, net: Callable, X: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE
        residual loss, the boundary condition loss and an additional condition
        loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution
                $\boldsymbol{\hat{u}}_{w}(x,y; \theta)$.
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point
            ($x, y$) in the domain. The last two columns correspond to the
            parameters ($R$, $\rho$).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current
            batch.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components 'loss_pde', 'loss_bc'
            and 'loss_add'.

        Notes
        -----
        In this example, initial conditions are not used, so the loss is
        computed only from the PDE residual, boundary conditions and an
        additional condition.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0  # λ_pde.
        lb_bc = 1.2  # λ_bc.
        lb_add = 1.5  # λ_add.

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]
        N_bc  = self.domain_kwargs["boundarySize"]
        N_add = self.domain_kwargs["auxiliarySize"]

        # Split the input tensor X into the different regions.
        X_pde = X[0:N_pde]  # PDE collocation.
        X_bc  = X[N_pde:N_pde + N_bc] # Boundary.
        X_add1 = X[N_pde + N_bc:N_pde + N_bc + N_add] # First additional constraint.
        X_add2 = X[-N_add:]  # Second additional constraint.

        # Lambda values for the different regions.
        lambda_vals_pde = self.lambda_fn(X_pde)  # Lambda values for the PDE.
        lambda_vals_bc = self.lambda_fn(X_bc)  # Lambda values for the boundary.

        # --------------------------------------------------------------------------
        # PDE loss: N[u] = f => ∇·(λ∇u) = 0. 
        # --------------------------------------------------------------------------
        # Model output for the PDE points.
        u_pde = net(X_pde)                

        # ∇u, grad_u[:, 0] = ∂u/∂x, grad_u[:, 1] = ∂u/∂y.
        grad_u = torch.autograd.grad(
            u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True
        )[0]

        lambda_grad_x = lambda_vals_pde * grad_u[:, 0]  # λ * ∂u/∂x.
        lambda_grad_y = lambda_vals_pde * grad_u[:, 1]  # λ * ∂u/∂y.

        # ∂(λ*∂u/∂x)/∂x.
        div_x = torch.autograd.grad(
            lambda_grad_x,
            X_pde,
            grad_outputs=torch.ones_like(lambda_grad_x),
            create_graph=True
        )[0][:, 0]

        # ∂(λ*∂u/∂y)/∂y.
        div_y = torch.autograd.grad(
            lambda_grad_y,
            X_pde,
            grad_outputs=torch.ones_like(lambda_grad_y),
            create_graph=True
        )[0][:, 1]

        # PDE residual loss.
        loss_pde = torch.mean((div_x + div_y) ** 2)  # ∇·(λ∇u).

        # --------------------------------------------------------------------------
        # Boundary condition loss: B[u] = g => λ * ∂u/∂n = f = cos(4θ).
        # --------------------------------------------------------------------------
        # Model output for the boundary points.
        u_bc = net(X_bc)  
        theta = torch.atan2(X_bc[:, 1], X_bc[:, 0])  # Angle theta.
        
        # Normal vector components.
        normal_x = torch.cos(theta)
        normal_y = torch.sin(theta)

        # ∇u
        grad_u_bc = torch.autograd.grad(
            u_bc, X_bc, grad_outputs=torch.ones_like(u_bc), create_graph=True
        )[0]

        # Normal derivative ∂u/∂n = ∇u·n.
        normal_derivative = grad_u_bc[:, 0] * normal_x + grad_u_bc[:, 1] * normal_y

        # Boundary loss λ * ∂u/∂n - f = 0.
        loss_bc = torch.mean(
            (lambda_vals_bc * normal_derivative - torch.cos(4 * theta)) ** 2
        )

        # --------------------------------------------------------------------------
        # Additional condition loss: Zero-mean gauge condition on the boundary.
        # --------------------------------------------------------------------------
        # Monte Carlo estimate of boundary average.
        mean_u = torch.mean(net(X_add1))

        # Enforce ∫∂B u ds = 0 and u = 0 on the boundary.
        loss_add = mean_u**2 + torch.mean((net(X_add2)) ** 2)

        # --------------------------------------------------------------------------
        # PINN loss: λ_pde * L_pde + λ_bc * L_bc + λ_add * L_add.
        # --------------------------------------------------------------------------
        loss_PINN = lb_pde * loss_pde + lb_bc * loss_bc + lb_add * loss_add

        return loss_PINN, {
            "loss_pde": loss_pde,
            "loss_bc": loss_bc,
            "loss_add": loss_add
        }

# ==================================================================================
# Main function.
# ==================================================================================
if __name__ == "__main__":

    from architectures import MLP                                 # Import the MLP architecture.
    from sampling import sample_circle_uniform_gauge_restriction  # Sampling function.

    # ------------------------------------------------------------------------------
    # Domain and model parameters.
    # ------------------------------------------------------------------------------
    R = 0.85
    param_supp = [(0., 10.)]
    domain_kwargs = {
        # Domain parameters.
        'center': [0, 0],
        'radius': 1,
        # Collocation points.
        'interiorSize': 2700,
        'boundarySize': 3000,
        'auxiliarySize': 3000,
        'valSize': 2500,
        # Parameters for the PINN.
        'fixed_params': [R],  # Fixed R.
        'param_domains': param_supp,  # Domain for ⍴.
        # Observed data.
        'data_x': None,
        'data_u': None,
    }

    # ------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize': 4,
        'hidden_lys': [50]*10,
        'outputSize': 1,
        'activation': 'tanh',
        'dropout': 0.0,
        'normalization': True,  # Whether to apply layer normalization.
    }
    
    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 0.8,  # Learning rate.
        'max_iter': 50,
        'tolerance_grad': 1e-09,  # Tolerance for gradient norm.
        'tolerance_change': 1e-09,  # Tolerance for the parameter change.
        'history_size': 100,
        'line_search_fn': "strong_wolfe"  # Line search strategy.
    }

    checkpoint_filename = 'infer_conductivity_value_MLP.pth'
    infer_rho_pinn = InferringConductivityValue(
        model_class=MLP,  # Model class for the PINN.
        model_kwargs=model_kwargs,
        domain_kwargs=domain_kwargs,  # Domain parameters.
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        epochs=3000,
        patience=300,
        sampling_fn=sample_circle_uniform_gauge_restriction,  # Sampling function.
        checkpoint_filename=checkpoint_filename,  # Filename for the checkpoints.
    )

    # ------------------------------------------------------------------------------
    # Train and plot.
    # ------------------------------------------------------------------------------
    # infer_rho_pinn.train()  # Uncomment to train the model.

    # Load the complete model and print model information.
    infer_rho_pinn.load_model(load_best=False)
    get_model_info(checkpoint_filename)

    # Plot the loss and the solution.
    plot_loss(
        model_instance=infer_rho_pinn, filename="loss_plot.png", ic=False
    )

    # Plot the solution with the best model.
    infer_rho_pinn.load_model(load_best=True)  # Load the best model.
    rho = 3.2
    plot_solution_circle(
        model_instance=infer_rho_pinn,
        domain_kwargs=domain_kwargs,
        parameters=[R, rho],
        filename="solution_plot.png"
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_circle(
        model_instance=infer_rho_pinn,
        domain_kwargs=domain_kwargs,
        parameters=[R, rho],
        filename="comparison_plot.png",
        adjust_scale=True
    )

    # Review boundary conditions
    with torch.no_grad():
        theta = torch.linspace(0, 2*np.pi, 2000, device=device)
        X_boundary = torch.stack(
            [torch.cos(theta), torch.sin(theta),
             torch.full_like(theta, R),
             torch.full_like(theta, rho)], dim=1
        )
        u_boundary = infer_rho_pinn.pinn(X_boundary)
        print("Mean on boundary:", u_boundary.mean().item())