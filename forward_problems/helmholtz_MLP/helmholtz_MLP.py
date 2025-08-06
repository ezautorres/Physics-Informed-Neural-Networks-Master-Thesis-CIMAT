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
from plotting import plot_loss, plot_solution_square, plot_comparison_contour_square # Plotting functions.
from utils import get_model_info                                                     # Utility function to get model information.

class HelmholtzPinn(PinnBase):
    def __init__(self, **params):
        """
        Initializes the HelmholtzPinn instance using the configuration dictionary passed to the base class.
        """
        super(HelmholtzPinn, self).__init__(**params)
        self.k = 1 * np.pi  # Wave number for the Helmholtz equation.

    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the analytical solution u(x,y) = sin(kx) sin(ky) evaluated at input points X.
        """
        return torch.sin(self.k * X[:,0]) * torch.sin(self.k * X[:,1])

    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the total PINN loss as a weighted sum of the interior PDE residual loss and 
        the boundary condition loss. The PDE is re-scaled to improve training stability:
        """
        lb_pde = 1.0
        lb_bc  = 1.0

        N_pde = self.domain_kwargs["interiorSize"]
        X_pde = X[:N_pde]
        X_bc  = X[N_pde:]

        # Forward pass
        u_pde = net(X_pde)

        # First derivatives
        grad_u = torch.autograd.grad(u_pde, X_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
        u_x, u_y = grad_u[:, 0], grad_u[:, 1]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, X_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, X_pde, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]

        # Define wave number and source
        source = self.k**2 * torch.sin(self.k * X_pde[:,0]) * torch.sin(self.k * X_pde[:,1])

        # Normalized residual: 
        residual = -u_xx - u_yy - self.k**2 * u_pde - source
        loss_pde = torch.mean(residual**2)

        # Boundary loss
        loss_bc = torch.mean(net(X_bc)**2)

        return lb_pde * loss_pde + lb_bc * loss_bc

# =======================================================================================================
# Main function.
# =======================================================================================================
if __name__ == "__main__":

    from architectures import MLP            # Import the SIREN architecture for the PINN.
    from sampling import sample_square_uniform # Sampling function for uniform sampling in a square domain.
    
    # ---------------------------------------------------------------------------------------------------
    # Domain and model parameters.
    # ---------------------------------------------------------------------------------------------------
    domain_kwargs = {
        # Domain parameters.
        'dim1_min'      : 0.,
        'dim1_max'      : 1.,
        'dim2_min'      : 0.,
        'dim2_max'      : 1.,
        # Collocation points.
        'interiorSize'  : 500,
        'dim1_minSize'  : 2000,
        'dim1_maxSize'  : 2000,
        'dim2_minSize'  : 2000,
        'dim2_maxSize'  : 2000,
        'valSize'       : 2000,
        # Parameters for the PINN.
        'fixed_params'  : None,
        'param_domains' : None,
        # Observed data.
        'data_x'        : None,
        'data_u'        : None, 
    }

    # ---------------------------------------------------------------------------------------------------
    # Architecture and optimizer parameters.
    # ---------------------------------------------------------------------------------------------------
    model_kwargs = {
        'inputSize'  : 2,               # Because we do not have parameters.
        'hidden_lys' : [128, 128, 128], # Hidden layers of the MLP.
        'outputSize' : 1                # Output size of the MLP.
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

    checkpoint_filename = 'helmholtz_MLP.pth'
    helmholtz_pinn = HelmholtzPinn(
        model_class         = MLP,                 # Model class for the PINN.
        model_kwargs        = model_kwargs,          # Model parameters for the PINN.
        domain_kwargs       = domain_kwargs,         # Domain parameters.
        optimizer_class     = optimizer_class,       # Optimizer class (default is LBFGS).
        optimizer_kwargs    = optimizer_kwargs,      # Optimizer parameters.
        epochs              = 150,                  # Number of epochs for training.
        patience            = 500,                   # Patience for early stopping.
        sampling_fn         = sample_square_uniform, # Sampling function.
        checkpoint_filename = checkpoint_filename,   # Filename for the checkpoints.
    )

    # Train the model.
    #helmholtz_pinn.train()

    # Load the complete model.
    helmholtz_pinn.load_model(load_best = False) # Load the complete model.
    get_model_info(checkpoint_filename)        # Print model information.
    
    # Plot the loss and the solution.
    plot_loss(
        model_instance = helmholtz_pinn,
        filename       = "loss_plot.pdf"
    )

    # Plot the solution with the best model.
    helmholtz_pinn.load_model(load_best = True) # Load the best model.
    plot_solution_square(
        model_instance = helmholtz_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "solution_plot.pdf"
    )

    # Plot the comparison of the PINN solution with the analytical solution.
    plot_comparison_contour_square(
        model_instance = helmholtz_pinn,
        domain_kwargs  = domain_kwargs,
        filename       = "comparison_plot.pdf"
    )