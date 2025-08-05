"""
infer_conductivity_value_MLP.py
-------------------------------
Instance of a Physics-Informed Neural Network (PINN) for inferring a spatial parameter in the steady-state
diffusion equation on the unit disk.

Author      : Ezau Faridh Torres Torres  
Date        : 3 August 2025  
Institution : Centro de Investigación en Matemáticas (CIMAT)

Description
-----------
This script solves an inverse problem using a Physics-Informed Neural Network (PINN) defined over a
circular domain. The goal is to infer the unknown parameter $\rho$ in the piecewise-constant diffusion
coefficient:
\[
\lambda(r) = \begin{cases}
1 + \rho, & \text{if } r < R, \\
1, & \text{otherwise}
\end{cases}
\]
where \( r = \sqrt{x^2 + y^2} \), and \( R \in (0,1) \) is a known interface radius.

The governing PDE is:
\[
\nabla \cdot (\lambda(r) \nabla u) = 0 \quad \text{in } \Omega = \{(x,y) \in \mathbb{R}^2 : \| (x,y) \| < 1 \}
\]
with mixed boundary and auxiliary conditions derived from an analytical solution.

This implementation includes:
    - A custom subclass `InferringConductivityValue` derived from a general PINN base class.
    - The definition of a discontinuous diffusion coefficient $\lambda(r)$.
    - An analytical solution for benchmarking inverse results.
    - Custom loss functions that enforce PDE, boundary, and auxiliary constraints.
    - Training, checkpointing, and visualization routines.

Usage
-----
Run the script to:
    - Instantiate and train a PINN model for the inverse problem.
    - Infer the unknown parameter $\rho$ using synthetic observations.
    - Visualize the training loss and compare the predicted solution with the analytical one.
    - Save model checkpoints and diagnostic plots.
"""
import numpy as np                                                                   # Numpy library.
import torch                                                                         # Import PyTorch
from typing import Callable                                                          # Type hinting.
import sys, os                                                                       # Import sys and os modules.
import random                                                                        # Random library.
np.set_printoptions(precision = 17, suppress = False)                                # Set print options for numpy.
np.random.seed(0)                                                                    # Seed for reproducibility.
random.seed(0)                                                                       # Seed for reproducibility.
torch.manual_seed(0)                                                                 # Seed for reproducibility.
torch.backends.cudnn.benchmark = False                                               # Reproducibility for CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                # Set device to GPU if available, else CPU.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # Add the parent directory to the path.
from pinn_base import PinnBase                                                       # Import the base class for PINNs.
from plotting import plot_loss, plot_solution_circle, plot_comparison_contour_circle # Plotting functions.
from utils import get_model_info                                                     # Utility function to get model information.

class InferringConductivityValue(PinnBase):
    def __init__(self, **params):
        """
        Initializes the InferringConductivityValue instance using the configuration dictionary passed to
        the base class.

        Parameters
        ----------
        **params : dict
            Dictionary of arguments required by the PinnBase class, including model configuration,
            optimizer settings, and domain sampling specifications.
        """
        super(InferringConductivityValue, self).__init__(**params) # Initialize the PINN with parameters from the base class.

    def lambda_fn(self, X: torch.Tensor) -> torch.Tensor:
        """
        Lambda function for the Unit Disk problem. This function modifies the lambda value depending on
        the radius of the point. If the radius is less than R, then lambda = 1 + rho, otherwise lambda = 1.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point (x,y) in the domain. The
            last two columns correspond to the parameters (rho, R).

        Returns
        -------
        torch.Tensor
            Lambda values.
        """
        r    = torch.sqrt(X[:,0]**2 + X[:,1]**2) # Radius.
        R    = X[:,2]                            # Parameter rho.
        rho  = X[:,3]                            # Parameter R.
        lambda_vals = torch.ones_like(r)         # Default lambda = 1.
        mask = r < R                             # Mask where r < R.

        # Apply lambda modification only where the condition is met.
        lambda_vals[mask] += rho[mask]  

        return lambda_vals
    
    def calculate_coefs(self, rho: torch.Tensor, R: torch.Tensor) -> tuple:
        """
        Calculate the coefficients b and c for the analytical solution of the Unit Disk problem.

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
        denom = 8 * (rho * R**8 + rho + 2) # Denominator for the coefficients.
        bn = ((rho + 2) * R**4) / denom    # Coefficient b.
        cn = - (rho * R**4) / denom        # Coefficient c.
        return bn, cn

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution for the Unit Disk problem: u(x,y): R^2 -> R, u(x,y): (x,y) -> u(x,y).

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point (x,y) in the domain. The
            last two columns correspond to the parameters (rho, R).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the analytical solution evaluated at each input point.
        """
        R   = X[:,2]                             # Parameter rho.
        rho = X[:,3]                             # Parameter R.
        b, c = self.calculate_coefs(rho, R)      # Coefficients b and c.
        r = torch.linalg.norm(X[:,0:2], dim = 1) # Calculate the radius.
        theta = torch.atan2(X[:,1], X[:,0])      # Calculate the angle. 
    
        u = torch.where(                         # Analytical solution for the Unit Disk.
            r < R,
            2 * (b + c) * (r / R)**4 * torch.cos(4 * theta),
            2 * (b * (r / R)**4 + c * (r / R)** -4) * torch.cos(4 * theta)
        )
        return u

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss, the boundary
        condition loss and an additional condition loss.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(x,y; \theta)$.
        X : torch.Tensor
            Tensor of shape (N, 4), where each row corresponds to a 2D point (x,y) in the domain. The
            last two columns correspond to the parameters (rho, R).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total training loss for the current batch.

        Notes
        -----
        In this example, initial and additional conditions are not used, so the loss is computed only
        from the PDE residual, boundary conditions and an additional condition.
        """
        # Define the weights for the different loss components.
        lb_pde = 1.0 # λ_pde
        lb_bc  = 1.0 # λ_bc
        lb_add = 1.0 # λ_add

        # Extract the number of points for each region from the domain_kwargs.
        N_pde = self.domain_kwargs["interiorSize"]  # Number of interior points for the PDE.
        N_bc  = self.domain_kwargs["boundarySize"]  # Number of boundary points.
        N_add = self.domain_kwargs["auxiliarySize"] # Number of additional constraint points.

        # Split the input tensor X into the different regions.
        X_pde = X[0:N_pde]            # Points in the interior for the PDE.
        X_bc  = X[N_pde:N_pde + N_bc] # Points on the boundary.
        X_add = X[-(N_add):]          # Points for the additional constraint.

        # Lambda values for the different regions.
        lambda_vals_pde = self.lambda_fn(X_pde) # Lambda values for the PDE points.
        lambda_vals_bc  = self.lambda_fn(X_bc)  # Lambda values for the boundary points.
        lambda_vals_add = self.lambda_fn(X_add) # Lambda values for the additional constraint

        # -----------------------------------------------------------------------------------------------
        # PDE loss: ∇·(λ∇u) = 0. 
        # -----------------------------------------------------------------------------------------------
        u_pde = net(X_pde)                                                                                        # Compute the model output for the PDE points.
        grad_u = torch.autograd.grad(u_pde, X_pde, grad_outputs = torch.ones_like(u_pde), create_graph = True)[0] # ∇u, grad_u[:,0] = ∂u/∂x, grad_u[:,1] = ∂u/∂y.
        lambda_grad_x = lambda_vals_pde * grad_u[:,0] # λ * ∂u/∂x.
        lambda_grad_y = lambda_vals_pde * grad_u[:,1] # λ * ∂u/∂y.

        div_x = torch.autograd.grad(lambda_grad_x, X_pde, grad_outputs = torch.ones_like(lambda_grad_x), create_graph = True)[0][:,0] # ∂(λ*∂u/∂x)/∂x.
        div_y = torch.autograd.grad(lambda_grad_y, X_pde, grad_outputs = torch.ones_like(lambda_grad_y), create_graph = True)[0][:,1] # ∂(λ*∂u/∂y)/∂y.
        divergence = div_x + div_y                                                                                                    # ∇·(λ∇u).

        loss_pde = torch.mean(divergence ** 2)

        # -----------------------------------------------------------------------------------------------
        # Boundary condition loss: λ * ∂u/∂n - f = 0.
        # -----------------------------------------------------------------------------------------------
        theta = torch.atan2(X_bc[:,1], X_bc[:,0]) # Angle theta.
        normal_x = torch.cos(theta)               # Normal vector x-component.
        normal_y = torch.sin(theta)               # Normal vector y-component.

        u_bc = net(X_bc)                                                                                          # Neural network output in the boundary region.
        grad_u_bc = torch.autograd.grad(u_bc, X_bc, grad_outputs = torch.ones_like(u_bc), create_graph = True)[0] # Gradient of the NN ∇u.
        normal_derivative = grad_u_bc[:,0] * normal_x + grad_u_bc[:,1] * normal_y                                 # Normal derivative ∂u/∂n = ∇u·n.

        loss_bc = torch.mean((lambda_vals_bc * normal_derivative - torch.cos(4 * theta)) ** 2) # λ * ∂u/∂n - f = 0.

        # -----------------------------------------------------------------------------------------------
        # Additional condition loss: u(x,y) = 0.
        # -----------------------------------------------------------------------------------------------
        loss_add = torch.mean((net(X_add)) ** 2) # Additional condition loss.

        return lb_pde * loss_pde + lb_bc * loss_bc + lb_add * loss_add

# =======================================================================================================
# Main function.
# =======================================================================================================
if __name__ == "__main__":

    from architectures import MLP                                 # Import the MLP architecture for the PINN.
    from sampling import sample_circle_uniform_center_restriction # Sampling function for uniform sampling in a circular domain.

    # ---------------------------------------------------------------------------------------------------
    # Domain and model parameters.
    # ---------------------------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'center'        : [0,0],
        'radius'        : 1,
        # Collocation points.
        'interiorSize'  : 1700,
        'boundarySize'  : 2000,
        'auxiliarySize' : 1200,
        'valSize'       : 2100,
        # Parameters for the PINN.
        'fixed_params'  : [0.85],     # Fixed parameter R.
        'param_domains' : [(0.,10.)], # Parameter domains for rho.
        # Observed data.
        'data_x'        : None,
        'data_u'        : None, 
    }

    # ---------------------------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ---------------------------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize'  : 4,                     # Input size of the MLP (2D coordinates + 2 parameters).
        'hidden_lys' : [100, 1000, 100, 100], # Hidden layers of the MLP.
        'outputSize' : 1                      # Output size of the MLP.
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

    checkpoint_filename = 'infer_conductivity_value_MLP.pth'
    infer_rho_pinn = InferringConductivityValue(
        model_class         = MLP,                                      # Model class for the PINN.
        model_kwargs        = model_kwargs,                             # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,                            # Domain parameters.
        optimizer_class     = optimizer_class,                          # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,                         # Optimizer parameters.
        epochs              = 250,                                      # Number of epochs for training.
        patience            = 50,                                       # Patience for early stopping.
        sampling_fn         = sample_circle_uniform_center_restriction, # Sampling function.
        checkpoint_filename = checkpoint_filename,                      # Filename for the checkpoints.
    )

    # Train the model.
    #infer_rho_pinn.train()

    # Load the complete model.
    infer_rho_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)          # Print model information.

    # Plot the loss and the solution.
    plot_loss(
        model_instance = infer_rho_pinn,
        filename       = "loss_plot.png"
    )

    # Plot the solution with the best model.
    infer_rho_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_circle(
        model_instance = infer_rho_pinn,
        domain_kwargs  = domain_kwargs, 
        parameters     = [0.85, 3.2],
        filename       = "solution_plot.png"
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_circle(
        model_instance = infer_rho_pinn,
        domain_kwargs  = domain_kwargs,
        parameters     = [0.85, 3.2],
        filename       = "comparison_plot.png"
    )