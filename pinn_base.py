"""
base_pinn.py
------------
Abstract base class for defining and training Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 25 June 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides a general abstract base class for PINNs, called `PinnBase`, which handles:
    - Model initialization and training configuration.
    - Sampling setup and optimizer selection.
    - Logging of loss histories and checkpoint management.

It requires the user to implement:
    - The analytical solution (for validation).
    - The PDE residual function (for physical loss).

Functions
---------
analytical_solution :
    Abstract method to define the analytical solution of the PDE.
loss_PINN :
    Abstract method to define the physical loss.
train :
    Launches the PINN training loop using an external trainer.
save_checkpoint :
    Saves the current model state.
load_model :
    Loads a previously saved model checkpoint.

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
  Journal of Computational Physics, 378, 686-707.
"""
# Necessary libraries.
import torch                                          # PyTorch library for tensor operations.
import numpy as np                                    # NumPy for numerical operations.
import random                                         # Random module for reproducibility.         
from abc import ABC, abstractmethod                   # Abstract Base Class for defining abstract methods.
from typing import Callable                           # Type hinting for callable functions.
import os, sys                                        # OS module for file path operations.
from trainer import train_pinn                        # Import the training function for PINNs.
from utils import load_model, save_checkpoint         # Utility functions for loading and saving model checkpoints.
np.set_printoptions(precision = 17, suppress = False) # Set numpy print options for better precision and suppress scientific notation.
np.random.seed(0)                                     # Set random seed for reproducibility.
random.seed(0)                                        # Set random seed for reproducibility.
torch.manual_seed(0)                                  # Set PyTorch random seed for reproducibility.
torch.backends.cudnn.benchmark = False                # Disable cuDNN benchmark for reproducibility.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, otherwise CPU.

class PinnBase(ABC):
    def __init__(
            self, epochs: int, patience: int, model_class: type, model_kwargs: dict,
            sampling_fn: Callable, domain_kwargs: dict, optimizer_class: type, optimizer_kwargs: dict,
            checkpoint_filename: str = "checkpoint.pth"
        ):
        """
        Initializes the base class for Physics-Informed Neural Networks (PINNs) in 2D or 1D+time domains.
        This class defines the core infrastructure for training PINNs, including:
            - Model and optimizer instantiation.
            - Sampling strategy and domain specification.
            - Checkpointing setup and logging configuration.
        Subclasses must implement problem-specific components such as the PDE residual and analytical
        solution.

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
        self.epochs        = epochs                      # Number of epochs for training.              
        self.patience      = patience                    # Number of epochs without improvement before early stopping.
        self.pinn          = model_class(**model_kwargs) # PINN: Neural network model.
        self.model_kwargs  = model_kwargs                # Model parameters.
        self.model_class   = model_class                 # Class of the neural network model.
        self.sampling_fn   = sampling_fn                 # Function to sample points for training and validation.
        self.domain_kwargs = domain_kwargs               # Domain parameters for the sampling function.

        # Checkpoint config.
        self.checkpoint_path     = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "trained_models") # Path to save the checkpoints.
        self.checkpoint_filename = checkpoint_filename                                                           # Filename for the checkpoints. 
        self.best_model_filename = checkpoint_filename.replace('.pth', '_best.pth')                              # Filename for the best model.

        # Optimizer config.
        self.optimizer_class = optimizer_class if optimizer_class is not None else torch.optim.LBFGS
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else dict(
            lr               = 0.5,
            max_iter         = 40,
            tolerance_grad   = 1e-09,
            tolerance_change = 1e-09,
            history_size     = 100,
            line_search_fn   = "strong_wolfe"
        )

        # Logging.
        self.loss_history     = []           # Training loss history.
        self.val_loss_history = []           # Validation loss history.
        self.best_train_loss  = float('inf') # Best validation loss.

    @abstractmethod
    def analytical_solution(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact analytical solution of the PDE at the given input points
            $\boldsymbol{u}_{w}(\mathbf{x}, t; \theta)$.
        This method must be implemented in subclasses for problems with known solutions, to enable
        validation and error computation during training.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (N, 2 + n_params), where:
                - X[:,0] = dim1 (e.g., x).
                - X[:,1] = dim2 (e.g., y or t).
                - X[:,2:] = optional parameters for parametric problems.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N,) containing the exact solution values at each point.
        """
        pass

    @abstractmethod
    def loss_PINN(self, net: Callable, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the full PINN loss $\mathcal{L}_{\text{PINN}}(w)$, which combines physics-based residuals
        and data-driven supervision. The total loss function is defined as:
            $\mathcal{L}_{\text{PINN}}(w) = \lambda_{\text{physics}} \, \mathcal{L}_{\text{physics}}(w)
                + \lambda_{\text{data}} \, \mathcal{L}_{\text{data}}(w)$,
        where:
            - $\mathcal{L}_{\text{physics}}(w)$ enforces consistency with the governing physical laws
            (e.g., PDE residuals, boundary and initial conditions),
            - $\mathcal{L}_{\text{data}}(w)$ corresponds to standard supervised loss with respect to
            observed or synthetic data,
            - and $\lambda_{\text{physics}}$, $\lambda_{\text{data}}$ are user-defined weights balancing
            both components.
      
        Parameters
        ----------
        net : Callable
            Neural network model approximating the solution $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$.
        X : torch.Tensor                                     
            Input collocation points of shape $(N, 2 + n_{\text{params}})$ where:
                - X[:,0] = first coordinate (e.g., $x$).
                - X[:,1] = second coordinate (e.g., $y$ or $t$).
                - X[:,2:] = optional physical or geometric parameters $\theta$.
            This tensor must have `requires_grad = True` for automatic differentiation.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the total loss $\mathcal{L}_{\text{PINN}}(w)$ used to train the model.
        """
        pass

    def train(self):
        """
        Starts the training loop for the PINN using the configured components. This function delegates
        the training logic to `train_pinn`, passing the model, sampling function, optimizer, and
        domain-specific arguments stored in the instance.
        """
        train_pinn(
            pinn_instance    = self,
            model            = self.pinn,
            sampling_fn      = self.sampling_fn,
            domain_kwargs    = self.domain_kwargs,
            epochs           = self.epochs,
            patience         = self.patience,
            optimizer_class  = self.optimizer_class,
            optimizer_kwargs = self.optimizer_kwargs
            )

    def save_checkpoint(self, state: dict, is_best: bool) -> None:
        """
        Saves the current training state, model weights, optimizer state, and training metadata. Delegates
        the logic to the global `save_checkpoint` function using the current instance context.

        Parameters
        ----------
        state : dict
            Dictionary containing model weights, optimizer state, training statistics, etc.
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
            self, filename = os.path.join(self.checkpoint_path, self.best_model_filename if load_best else self.checkpoint_filename),
                   load_best = load_best
        )