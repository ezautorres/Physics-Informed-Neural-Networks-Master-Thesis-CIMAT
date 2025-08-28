"""
pinn_base.py
------------
Abstract base class for Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 20 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module defines the abstract base class `PinnBase`, which provides the
core infrastructure for training Physics-Informed Neural Networks (PINNs).
It encapsulates:
    - Model instantiation and optimizer configuration.
    - Domain sampling strategy integration.
    - Training loop delegation with checkpointing and logging.
    - Abstract methods for analytical solution and PDE loss, to be implemented
      by problem-specific subclasses.

The class ensures reproducibility by setting random seeds across NumPy,
PyTorch, and Python's `random` module, and it manages device configuration
(CPU/GPU).

Classes
-------
PinnBase(ABC)
    Abstract base class for PINNs. Provides training, checkpointing, and
    model-loading utilities. Subclasses must implement:
        - `.analytical_solution(X)`
        - `.loss_PINN(model, X)`

Usage
-----
Example: defining a custom PINN subclass
>>> import torch
>>> from pinn_base import PinnBase

>>> class HeatEquationPINN(PinnBase):
...     def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
...         # Example analytical solution
...         x, t = X[:, 0], X[:, 1]
...         return torch.exp(-t) * torch.sin(torch.pi * x)

...     def loss_PINN(
...         self, net: Callable, X: torch.Tensor
...     ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
...         # Example PDE residual loss
...         X.requires_grad_(True)
...         u_pred = net(X)
...         u_t = torch.autograd.grad(u_pred, X, torch.ones_like(u_pred), create_graph=True)[0][:, 1]
...         u_xx = torch.autograd.grad(
...             torch.autograd.grad(u_pred, X, torch.ones_like(u_pred), create_graph=True)[0][:, 0],
...             X,
...             torch.ones_like(u_pred),
...             create_graph=True
...         )[0][:, 0]
...         residual = u_t - 0.01 * u_xx
...         return torch.mean(residual**2), {"loss_pde": torch.mean(residual**2)}

>>> # Instantiate and train
>>> pinn = HeatEquationPINN(
...     epochs=500,
...     patience=20,
...     model_class=torch.nn.Sequential,
...     model_kwargs={"layers": [2, 50, 50, 1]},
...     sampling_fn=my_sampling_function,
...     domain_kwargs={"dim1_min": 0, "dim1_max": 1, "dim2_min": 0, "dim2_max": 1},
...     optimizer_class=torch.optim.Adam,
...     optimizer_kwargs={"lr": 1e-3},
... )
>>> pinn.train()

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.
  Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/
- NumPy documentation: https://numpy.org/doc/
"""
# Necessary libraries.
import os                                          # File paths.
import sys                                         # System functions.
import random                                      # Random numbers.
from abc import ABC, abstractmethod                # Abstract base classes.
from typing import Callable                        # Type hints.
import numpy as np                                 # Arrays and math.
import torch                                       # Tensors and autograd.
np.set_printoptions(precision=17, suppress=False)  # Set NumPy printing precision.
np.random.seed(0)                                  # NumPy random seed.
random.seed(0)                                     # Python random seed.
torch.manual_seed(0)                               # PyTorch random seed.
torch.backends.cudnn.benchmark = False             # Disable for reproducibility.

# Select GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from trainer import train_pinn                     # Training loop.
from utils import load_model, save_checkpoint      # Model I/O.

class PinnBase(ABC):
    def __init__(
        self,
        epochs: int,
        patience: int,
        model_class: type,
        model_kwargs: dict,
        sampling_fn: Callable,
        domain_kwargs: dict,
        optimizer_class: type,
        optimizer_kwargs: dict,
        checkpoint_filename: str = "checkpoint.pth"
        ):
        """
        Initializes the base class for Physics-Informed Neural Networks (PINNs)
        in 2D or 1D+time domains. This class defines the core infrastructure for
        training PINNs, including:
            - Model and optimizer instantiation.
            - Sampling strategy and domain specification.
            - Checkpointing setup and logging configuration.
        Subclasses must implement problem-specific components such as the PDE
        residual and analytical solution.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        patience : int
            Number of epochs without improvement before triggering early stopping.
        model_class : type
            Class of the neural network model to instantiate (e.g., MLP).
        model_kwargs : dict
            Keyword arguments for the neural network constructor.
        sampling_fn : Callable
            Function to sample training and validation points from the domain.
        domain_kwargs : dict
            Dictionary of parameters to pass to the sampling function.
        optimizer_class : type
            Optimizer class from `torch.optim` (e.g., LBFGS or Adam).
        optimizer_kwargs : dict
            Dictionary of keyword arguments for the optimizer.
        checkpoint_path : str, optional
            Directory where checkpoints will be saved. Default is "trained_model".
        checkpoint_filename : str, optional
            Base filename for saving model checkpoints. Default is "checkpoint.pth".
        """
        # Initialize the PinnBase class.
        self.epochs = epochs
        self.patience = patience
        self.pinn = model_class(**model_kwargs)
        self.model_kwargs = model_kwargs
        self.model_class = model_class
        self.sampling_fn = sampling_fn
        self.domain_kwargs = domain_kwargs

        # Checkpoint config.
        self.checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), "trained_models"
        )
        self.checkpoint_filename = checkpoint_filename
        self.best_model_filename = checkpoint_filename.replace(
            '.pth', '_best.pth'
        )

        # Optimizer config.
        self.optimizer_class = (
            optimizer_class if optimizer_class is not None else torch.optim.LBFGS
        )
        self.optimizer_kwargs = (
            optimizer_kwargs
            if optimizer_kwargs is not None
            else dict(
                lr=0.5,
                max_iter=40,
                tolerance_grad=1e-9,
                tolerance_change=1e-9,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
        )

        # Initialize training history.
        self.loss_history = []
        self.loss_components_history = []
        self.val_loss_history = []
        self.best_train_loss = float('inf')

    @abstractmethod
    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact analytical solution of the PDE at the given input
        points $\boldsymbol{u}_{w}(\mathbf{x}, t; \theta)$. This method must
        be implemented in subclasses for problems with known solutions, to
        enable validation and error computation during training.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2 + n_params), where:
                - X[:, 0] = dim1 (e.g., x).
                - X[:, 1] = dim2 (e.g., y or t).
                - X[:, 2:] = optional parameters for parametric problems.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the exact solution values at each point.
        """
        pass

    @abstractmethod
    def loss_PINN(
            self, net: Callable, X: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the full PINN loss $\mathcal{L}_{\text{PINN}}(w)$, which
        combines physics-based residuals and data-driven supervision. The total
        loss function is defined as: $\mathcal{L}_{\text{PINN}}(w) = 
            \lambda_{\text{physics}} \, \mathcal{L}_{\text{physics}}(w)
                + \lambda_{\text{data}} \, \mathcal{L}_{\text{data}}(w)$,
        where:
            - $\mathcal{L}_{\text{physics}}(w)$ enforces consistency with the
            governing physical laws (e.g., PDE residuals, boundary and initial
            conditions),
            - $\mathcal{L}_{\text{data}}(w)$ corresponds to standard supervised
            loss with respect to observed or synthetic data,
            - and $\lambda_{\text{physics}}$, $\lambda_{\text{data}}$ are
            user-defined weights balancing both components.

        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution
                $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$.
        X : torch.Tensor                                     
            Input collocation points of shape $(N, 2 + n_{\text{params}})$ where:
                - X[:, 0] = first coordinate (e.g., $x$).
                - X[:, 1] = second coordinate (e.g., $y$ or $t$).
                - X[:, 2:] = optional physical or geometric parameters $\theta$.
            This tensor must have `requires_grad = True` for automatic diff.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total loss $\mathcal{L}_{\text{PINN}}(w)$
            used to train the model.
        dict[str, torch.Tensor]
            Dictionary containing individual loss components (e.g., 'loss_pde', ...).
        """
        pass

    def train(self):
        """
        Starts the training loop for the PINN using the configured components.
        This function delegates the training logic to `train_pinn`, passing the
        model, sampling function, optimizer, and domain-specific arguments
        stored in the instance.
        """
        train_pinn(
            pinn_instance=self,
            model=self.pinn,
            sampling_fn=self.sampling_fn,
            domain_kwargs=self.domain_kwargs,
            epochs=self.epochs,
            patience=self.patience,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs
        )

    def save_checkpoint(self, state: dict, is_best: bool) -> None:
        """
        Saves the current training state, model weights, optimizer state, and
        training metadata. Delegates the logic to the global `save_checkpoint`
        function using the current instance context.

        Parameters
        ----------
        state : dict
            Dictionary containing model weights, optimizer state, training
            statistics, etc.
        is_best : bool
            If True, a separate copy of the best-performing model is saved.
        """
        save_checkpoint(self, state, is_best)

    def load_model(self, load_best: bool = True):
        """
        Loads a saved model checkpoint and restores weights and training metadata.

        Parameters
        ----------
        load_best : bool, optional
            If True, loads the best model version if available. Default is True.
        """
        load_model(
            self,
            filename=os.path.join(
                self.checkpoint_path,
                self.best_model_filename if load_best else self.checkpoint_filename,
            ),
            load_best=load_best,
        )