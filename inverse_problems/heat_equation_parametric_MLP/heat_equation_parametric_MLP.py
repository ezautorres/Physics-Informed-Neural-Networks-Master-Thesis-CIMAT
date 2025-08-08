"""
heat_equation_parametric_MLP.py
-------------------------------
Instance of a Physics-Informed Neural Network (PINN) applied to a 1D Heat Equation with parametric
diffusivity.

Author: Ezau Faridh Torres Torres.
Date: 7 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This script solves the 1D Heat Equation using a Physics-Informed Neural Network (PINN), where the thermal
diffusivity \alpha is treated as an input parameter. The domain is [x,t] \in [0,L] \times [0,T] with
Dirichlet boundary conditions and sinusoidal initial condition.

The PDE to be solved is:
    \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}.

Subject to:
    - Initial condition: u(x,0) = sin(n \pi x)
    - Boundary conditions: u(0,t) = u(1,t) = 0.

The analytical solution is:
    u(x,t) = \exp(-n^2 \pi^2 \alpha t) \sin(n \pi x)

This implementation includes:
    - A custom class 'HeatEquationPinn' inheriting from a general PINN base class.
    - Parametric training with \alpha as a third input.
    - PDE residual, initial and boundary losses.
    - Visualization utilities.

Note
-----
In this implementation L = T = 3 and n = 5 and \alpha is treated as a parameter to be inferred.

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

class HeatEquationPinn(PinnBase):
    def __init__(self, **params):
        super(HeatEquationPinn, self).__init__(**params)
        self.L = self.domain_kwargs["dim1_max"]  # Length of the domain in x.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        x, t, n, alpha = X[:,0], X[:,1], X[:,2], X[:,3]
        return torch.exp(-(n**2 * torch.pi**2 * alpha * t) / (self.L**2)) * torch.sin((n * torch.pi * x) / self.L)

    def loss_PINN2(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        lb_pde = 1.0
        lb_ic  = 1.0
        lb_bc  = 1.0

        N_pde   = self.domain_kwargs["interiorSize"]
        N_bc_l  = self.domain_kwargs["dim1_minSize"]
        N_bc_r  = self.domain_kwargs["dim1_maxSize"]
        N_ic    = self.domain_kwargs["dim2_minSize"]
        N_total = N_pde + N_bc_l + N_bc_r + N_ic

        indicators = torch.cat((
            torch.ones(N_pde),
            torch.ones(N_bc_l) * 2,
            torch.ones(N_bc_r) * 3,
            torch.ones(N_ic)   * 4
        )).to(X.device)

        loss_pde = torch.tensor(0.0, device = X.device)
        loss_ic  = torch.tensor(0.0, device = X.device)
        loss_bc  = torch.tensor(0.0, device = X.device)

        for i in range(N_total):
            xt = X[i,:].unsqueeze(0).requires_grad_(True)
            region = int(indicators[i].item())
            u = net(xt) 
            x, t, n, alpha = xt[0,0], xt[0,1], xt[0,2], xt[0,3]

            if region == 1:
                grad_u = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                u_x = grad_u[:,0]
                u_t = grad_u[:,1]
                u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0]
                loss_pde += (u_t - alpha * u_xx).pow(2).squeeze()

            elif region in [2,3]:
                loss_bc += u.pow(2).squeeze()

            elif region == 4:
                loss_ic += (u - torch.sin((n * np.pi * x) / self.L)).pow(2).squeeze()

        loss_pde /= N_pde
        loss_ic  /= N_ic
        loss_bc  /= (N_bc_l + N_bc_r)

        return lb_pde * loss_pde + lb_ic * loss_ic + lb_bc * loss_bc
    
    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        lb_pde = 1.0
        lb_ic  = 1.0
        lb_bc  = 1.0

        N_pde   = self.domain_kwargs["interiorSize"]
        N_bc_l  = self.domain_kwargs["dim1_minSize"]
        N_bc_r  = self.domain_kwargs["dim1_maxSize"]
        N_ic    = self.domain_kwargs["dim2_minSize"]

        # Separar los subdominios
        X_pde = X[0:N_pde]                          # PDE collocation points.
        X_bc_l = X[N_pde:N_pde+N_bc_l]               # Left boundary points.
        X_bc_r = X[N_pde+N_bc_l:N_pde+N_bc_l+N_bc_r] # Right boundary points.
        X_ic  = X[-N_ic:]                           # Initial condition points.

        # PDE loss
        u_pde = net(X_pde)
        grads = torch.autograd.grad(u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
        u_x = grads[:, 0]
        u_t = grads[:, 1]
        u_xx = torch.autograd.grad(u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        alpha_pde = X_pde[:, 3]
        residual = u_t - alpha_pde * u_xx
        
        loss_pde = torch.mean(residual**2)

        # Boundary loss (u = 0 en ambas fronteras)
        u_bc_l = net(X_bc_l)
        u_bc_r = net(X_bc_r)
        loss_bc = torch.mean(u_bc_l**2) + torch.mean(u_bc_r**2)

        # Initial condition loss
        x_ic = X_ic[:, 0]
        n_ic = X_ic[:, 2]
        u_ic = net(X_ic)
        u_ic_true = torch.sin(n_ic * np.pi * x_ic / self.L)
        loss_ic = torch.mean((u_ic.squeeze() - u_ic_true)**2)

        # Total loss
        return lb_pde * loss_pde + lb_ic * loss_ic + lb_bc * loss_bc

if __name__ == "__main__":
    from architectures import MLP
    from sampling import sample_square_uniform

    L, T, n = 2, 2, 5  # Domain parameters.

    domain_kwargs = {
        'dim1_min': 0., 'dim1_max': L,     # x in [0, 5]
        'dim2_min': 0., 'dim2_max': T,     # t in [0, 5]
        'interiorSize': 15000,
        'dim1_minSize': 2000,
        'dim1_maxSize': 2000,
        'dim2_minSize': 2000,
        'dim2_maxSize': 0,
        'valSize'     : 1800,
        'fixed_params': [n],
        'param_domains': [(0, 0.1)], 
        'data_x': None,
        'data_u': None,
    }

    model_kwargs = {
        'inputSize': 4,
        'hidden_lys': [100,50],
        'outputSize': 1
    }

    optimizer_class = torch.optim.LBFGS
    optimizer_kwargs = {
        'lr': 1,
        'max_iter': 100,
        'tolerance_grad': 1e-09,
        'tolerance_change': 1e-09,
        'history_size': 100,
        'line_search_fn': "strong_wolfe"
    }

    checkpoint_filename = "heat_parametric_MLP.pth"
    heat_pinn = HeatEquationPinn(
        model_class         = MLP,
        model_kwargs        = model_kwargs,
        domain_kwargs       = domain_kwargs,
        optimizer_class     = optimizer_class,
        optimizer_kwargs    = optimizer_kwargs,
        epochs              = 400,
        patience            = 100,
        sampling_fn         = sample_square_uniform,
        checkpoint_filename = checkpoint_filename,
    )

    #heat_pinn.train()
    heat_pinn.load_model(load_best = False)
    get_model_info(checkpoint_filename)

    #plot_loss(heat_pinn, "loss_plot.png")
    heat_pinn.load_model(load_best = True)

    plot_solution_square(
        model_instance = heat_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.png",
        time_dependent = True,
        parameters = [n, 0.05]
    )

    plot_comparison_contour_square(
        model_instance = heat_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.png",
        time_dependent = True,
        parameters = [n, 0.05]
    )
