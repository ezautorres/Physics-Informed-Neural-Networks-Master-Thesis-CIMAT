"""
trainer.py
----------
Training loop and optimizer closure for Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 25 June 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module defines the training loop used for training PINNs on forward or inverse problems. Includes:
    - Epoch-based training with custom closure function for L-BFGS or other optimizers.
    - Dynamic resampling of training and validation points at each epoch.
    - Early stopping and checkpoint saving.
    - Logs training loss, validation loss (against analytical solution), and best epoch tracking.

Functions
---------
train_pinn :
    Main training loop for a given PINN model and problem instance.
closure_fn :
    Closure used by optimizers to compute loss and perform backpropagation.

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
  Journal of Computational Physics, 378, 686-707.
"""
# Necessary libraries.
import torch                      # PyTorch library for tensor operations.
import time                       # Time module.
from tqdm import tqdm             # Progress bar for training epochs.
from colorama import Fore, Style  # For colored terminal output.
from typing import Callable       # Type hinting for callable functions.
from utils import save_checkpoint # Function to save the model checkpoint.

def train_pinn(
    pinn_instance: object, model: torch.nn.Module, sampling_fn: Callable, domain_kwargs: dict,
    epochs: int, patience: int, optimizer_class: type, optimizer_kwargs: dict, device: str = 'cpu'
    ) -> None:
    """
    Trains a Physics-Informed Neural Network (PINN) model using dynamic sampling over a specified domain.
    This function performs the full training loop for a PINN model using the provided sampling function, 
    optimizer, and training hyperparameters. It records training and validation losses, tracks the best 
    epoch, implements early stopping, and saves checkpoints at each iteration. Validation is performed by 
    comparing the model output with an analytical solution provided in the `pinn_instance`.

    Parameters
    ----------
    pinn_instance : object
        Instance of the PINN class. Must contain:
            - `.loss_PINN(model, X)` to compute the physics loss.
            - `.analytical_solution(X)` to compute the reference validation output.
            - Attributes like `.loss_history`, `.val_loss_history`, `.best_epoch`, etc., will be updated.
    model : torch.nn.Module
        Neural network model representing the approximation $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$.
    sampling_fn : Callable
        Sampling function that generates training and validation points. Must accept `train = True/False`
        and `**domain_kwargs`.
    domain_kwargs : dict
        Dictionary of domain-specific parameters to pass to the sampling function.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Number of consecutive epochs without improvement allowed before triggering early stopping.
    optimizer_class : type
        Optimizer class from `torch.optim`, e.g., `torch.optim.LBFGS` or `torch.optim.Adam`.
    optimizer_kwargs : dict
        Keyword arguments to initialize the optimizer, such as learning rate, momentum, etc.
    device : str, optional
        Device on which training is performed ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    None
        The function modifies the `pinn_instance` in-place. It does not return any value. Training metrics
        are stored and the best model state is saved via `save_checkpoint`.
    """
    start_time = time.time() # Start time for training.
    n_no_improvement = 0     # Number of epochs without improvement.

    for epoch in tqdm(range(1, epochs + 1), desc = "Training..."): # Loop over the epochs.
        epoch_time = time.time()                                   # Start time for the epoch.
        
        # Dentro de train_pinn(...)
        sampling_args = {k: v for k, v in domain_kwargs.items() if k not in ['data_x', 'data_u']} # Prepare sampling arguments excluding data_x and data_u.
        X = sampling_fn(train = True, **sampling_args).to(device)                                 # Sample training points and move to the specified device.
        pinn_instance.X = X                                                                       # Store the training points in the PINN instance.

        # Optimization.
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs) # Initialize the optimizer with model parameters and kwargs.
        pinn_instance.optimizer = optimizer                                 # Store the optimizer in the PINN instance.
        optimizer.step(lambda: closure_fn(pinn_instance, model, X))         # Perform the optimization step using the closure function.

        # Training loss.
        loss_train = pinn_instance.loss_PINN(model, X).item() # Compute the training loss.
        pinn_instance.loss_history.append(loss_train)        # Append the training loss to the history.

        # Validation loss.
        with torch.no_grad():
            X_val = sampling_fn(train = False, **sampling_args).to(next(model.parameters()).device) # Sample validation points.
            u_val_true = pinn_instance.analytical_solution(X_val).squeeze()                         # Compute the analytical solution.
            u_val_pred = model(X_val).squeeze()                                                     # Compute the predicted solution.
            loss_val = torch.norm(u_val_pred - u_val_true) / torch.norm(u_val_true)                 # Compute the validation loss.
            pinn_instance.val_loss_history.append(loss_val)                                         # Append the validation loss to the history.

        # Check for improvement and update best model state.
        is_improvement = loss_train < pinn_instance.best_train_loss
        if is_improvement:
            pinn_instance.best_train_loss = loss_train
            pinn_instance.best_time = time.time() - start_time
            pinn_instance.best_epoch = epoch
            pinn_instance.best_val_loss = loss_val
            n_no_improvement = 0
        else:
            n_no_improvement += 1

        # Save checkpoint.
        elapsed_time = time.time() - start_time # Total elapsed time.
        save_checkpoint(
            pinn_instance,
            state =
            {
                # This epoch information.
                'epoch'            : epoch,
                'state_dict'       : model.state_dict(),
                'loss_train'       : loss_train,
                'loss_val'         : loss_val,
                'elapsed_time'     : elapsed_time,
                # Best information.
                'best_train_loss'  : pinn_instance.best_train_loss,
                'best_time'        : pinn_instance.best_time,
                'best_epoch'       : pinn_instance.best_epoch,
                'best_val_loss'    : pinn_instance.best_val_loss,
                # History information.
                'loss_history'     : pinn_instance.loss_history,
                'val_loss_history' : pinn_instance.val_loss_history,
                'optimizer'        : optimizer.state_dict(),
            },
            is_best = is_improvement
        )

        # Early stopping.
        if n_no_improvement > patience:
            tqdm.write(f"{Fore.RED}Early stopping triggered at epoch {epoch}!{Style.RESET_ALL}")
            break
        elif loss_train < 1e-6:
            tqdm.write(f"{Fore.RED}Loss is too low, stopping training!{Style.RESET_ALL}")
            break

        # Print epoch information.
        tqdm.write(
            f"{Style.BRIGHT}{Fore.CYAN} Epoch{epoch:>{4 if epochs >= 100 else 3}}/{epochs}{Style.RESET_ALL}"
            f"  ➤ Training loss: {Fore.GREEN}{loss_train:.4f}{Style.RESET_ALL} | "
            f"Validation loss: {Fore.YELLOW}{loss_val:.4f}{Style.RESET_ALL} | "
            f"Epoch Time: {time.time() - epoch_time:.2f} s | "
            f"Total Time: {elapsed_time:.2f} s")

    print(f"\n{Style.BRIGHT}--- Total Training Time: {elapsed_time:.2f} seconds ---{Style.RESET_ALL}")

def closure_fn(pinn_instance: object, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Closure function for computing the loss and performing backpropagation in PINN optimization. This
    function is designed to be passed to optimizers like `torch.optim.LBFGS`, which require a callable
    that returns the loss and performs `backward()` internally. It computes the physics-informed loss
    from the `pinn_instance` and triggers gradient computation.

    Parameters
    ----------
    pinn_instance : object
        Instance of the PINN class. Must implement the method:
            - `.loss_PINN(model, X)` to compute the physics loss.
            - Must also have the attribute `.optimizer` with a valid PyTorch optimizer.
    model : torch.nn.Module
        The neural network model representing the approximation $\boldsymbol{\hat{u}}_{w}(\mathbf{x})$.
    X : torch.Tensor
        Input tensor containing collocation points, possibly with parameters. Shape: $(N, d)$ where
        $d = 2 + n_{\text{params}}$.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the total physics-informed loss for the current batch.
    """
    pinn_instance.optimizer.zero_grad()     # Reset the gradients of the optimizer.
    loss = pinn_instance.loss_PINN(model, X) # Compute the total cost.
    loss.backward()                         # Compute the gradients.

    return loss