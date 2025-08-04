"""
utils.py
--------
Checkpointing and utility functions for training and restoring Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 25 June 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides core utility functions for saving, loading, and inspecting model checkpoints during
PINN training. It supports:
    - Saving full training state including parameters, seeds, and model representation.
    - Loading trained models for evaluation or continued training.
    - Displaying metadata from saved checkpoints.
These utilities facilitate model reproducibility and debugging, especially in forward/inverse PDE problems.

Usage
-----
>>> from utils import save_checkpoint, load_model, get_model_info
>>> save_checkpoint(pinn_instance, state, is_best = True)
>>> load_model(pinn_instance, filename = "checkpoint.pth", device = "cuda")
>>> get_model_info("checkpoint.pth", device = "cpu")

Functions
---------
save_checkpoint :
    Saves the model, optimizer, training metadata, and configuration to disk.
load_model :
    Loads model weights and training state from a checkpoint file.
get_model_info :
    Prints a summary of model configuration, architecture, and training metadata.

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
    Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/
"""
# Necessary libraries.
import numpy as np            # NumPy for numerical operations.
import torch                  # PyTorch library for tensor operations.
import os, sys                # OS module for file operations.
import shutil                 # Shutil module for file operations.
import datetime               # Datetime module for timestamping.
import random                 # Random module for reproducibility.
from architectures import MLP # Import the MLP architecture.
import torch.optim            # PyTorch optimizers.
from sampling import sample_circle_uniform_center_restriction

# Map string names to actual classes
MODEL_REGISTRY = {
    "MLP" : MLP,
}
OPTIMIZER_REGISTRY = {
    "LBFGS" : torch.optim.LBFGS,
    "Adam"  : torch.optim.Adam,
}
SAMPLING_REGISTRY = {
    "sample_square_uniform"                    : sample_circle_uniform_center_restriction,
    "sample_circle_uniform_center_restriction" : sample_circle_uniform_center_restriction,
}

def get_model_info(filename: str, device: str = 'cpu') -> None:
    """
    Displays metadata, architecture, and training statistics from a saved PINN checkpoint. This utility
    prints model configuration, architecture summary, and final training results from a checkpoint file.
    It is useful for inspecting saved models without loading them into memory.

    Parameters
    ----------
    filename : str
        Path to the checkpoint file (e.g., 'checkpoint.pth').
    device : str, optional
        Device to map tensors to when loading ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    None
        Outputs formatted information to the console. Returns the raw checkpoint if parameters are missing.
    """
    # Load the checkpoint file.
    filename = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "trained_models", filename)
    checkpoint = torch.load(filename, map_location = torch.device(device), weights_only = False)
    params = checkpoint.get('params', None)

    # Check if the checkpoint contains parameters.
    if params is None:
        print(f"\nNo parameters found in checkpoint {filename}")
        return checkpoint

    # Print the model information.
    print("\n" + "─"*60)
    print(f"Model Info:")
    print("─" * 60)
    
    all_keys = []
    for k, v in params.items():
        if isinstance(v, dict):
            all_keys.extend([str(inner_k) for inner_k in v])
        elif k != "model_repr":
            all_keys.append(k)
    max_key_len = max(len(k) for k in all_keys)
    
    # Function to print dictionary blocks with aligned keys.
    def print_block_dict(d: dict):
        for k, v in d.items():
            print(f"  • {str(k):<{max_key_len}} : {v}")
    
    # Print each parameter with aligned keys.
    for k, v in params.items():
        if isinstance(v, dict):
            print(f"{k}:")
            print_block_dict(v)
        elif k == "model_repr":
            continue
        else:
            print(f"{k:<{max_key_len+4}} : {v}")
    
    # Print the model architecture if available.
    if "model_repr" in params:
        print("\n" + "─"*60)
        print("Model Architecture:")
        print("─" * 60)
        for line in params["model_repr"].splitlines():
            print(f"  {line}")
    
    # Print the final information from the checkpoint.
    def _format_val(val, fmt = ".8f", int_fmt = "d"):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, float) and val.is_integer():
            return f"{int(val):{int_fmt}}"
        elif isinstance(val, int):
            return f"{val:{int_fmt}}"
        elif isinstance(val, float):
            return f"{val:{fmt}}"
        else:
            return str(val)
    
    # Print final information from the checkpoint.
    print("\n" + "─"*60)
    print("Model Final Information:")
    print("─" * 60)
    info_lines = [
        ("Finished at Epoch"    , checkpoint.get('epoch', 'N/A')),
        ("Final Training Loss"  , checkpoint.get('loss_train', 'N/A')),
        ("Final Validation Loss", checkpoint.get('loss_val', 'N/A')),
        ("Total Training Time"  , f"{_format_val(checkpoint.get('elapsed_time', 'N/A'), '.2f')} s"),
        ("Best Epoch"           , checkpoint.get('best_epoch', 'N/A')),
        ("Best Training Loss"   , checkpoint.get('best_train_loss', 'N/A')),
        ("Best Validation Loss" , checkpoint.get('best_val_loss', 'N/A')),
        ("Best Training Time"   , f"{_format_val(checkpoint.get('best_time', 'N/A'), '.2f')} s"),
    ]

    max_info_key_len = max(len(k) for k, _ in info_lines)
    for key, val in info_lines:
        print(f"{key:<{max_info_key_len}} : {_format_val(val)}")

def save_checkpoint(pinn_instance: object, state: dict, is_best: bool) -> None:
    """
    Saves the current state of a PINN model, optimizer, and training metadata to disk. This function
    writes a complete checkpoint including model weights, optimizer state, training history,
    hyperparameters, and reproducibility metadata. If the current model achieved the best performance so
    far, it also saves a separate best model copy.

    Parameters
    ----------
    pinn_instance : object
        The main PINN wrapper. Must include attributes like:
            - `pinn` (torch.nn.Module)
            - `checkpoint_path`, `checkpoint_filename`, `best_model_filename`
            - `model_kwargs`, `domain_kwargs`, `optimizer_class`, `optimizer_kwargs`
            - `epochs`, `patience`
    state : dict
        Dictionary with current training state (epoch, losses, model state, optimizer state, etc.).
    is_best : bool
        Whether the current model is the best so far. If True, a separate best model is saved.

    Returns
    -------
    None
        Writes checkpoint files to disk.
    """
    # Create the checkpoint path if it does not exist.
    if not os.path.exists(pinn_instance.checkpoint_path): 
        os.makedirs(pinn_instance.checkpoint_path, exist_ok = True)

    # Update the best model state dictionary.
    if is_best:
        state['best_model_state_dict'] = pinn_instance.pinn.state_dict()

    # Store initialization parameters.
    state['params'] = {
        "model_kwargs"        : pinn_instance.model_kwargs,
        "domain_kwargs"       : pinn_instance.domain_kwargs,
        "model_class"         : pinn_instance.model_class.__name__,
        "optimizer_class"     : pinn_instance.optimizer_class.__name__,
        "optimizer_kwargs"    : pinn_instance.optimizer_kwargs,
        "epochs"              : pinn_instance.epochs,
        "patience"            : pinn_instance.patience,
        "device"              : str(next(pinn_instance.pinn.parameters()).device),
        "sampling_fn"         : pinn_instance.sampling_fn.__name__,
        "torch_version"       : torch.__version__,
        "random_seeds"        : {
            "torch"           : torch.initial_seed(),
            "numpy"           : np.random.get_state()[1][0],
            "python"          : random.randint(0, 100000),
        },
        "datetime"            : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_filename" : pinn_instance.checkpoint_filename,
        "model_repr"          : str(pinn_instance.pinn)
    }

    # Save the checkpoint (includes all necessary details).
    filepath = os.path.join(pinn_instance.checkpoint_path, pinn_instance.checkpoint_filename)
    torch.save(state, filepath)

    # Save the best model as a separate file.
    if is_best:
        best_path = os.path.join(pinn_instance.checkpoint_path, pinn_instance.best_model_filename)
        shutil.copyfile(filepath, best_path)


def load_model(
        pinn_instance: object, filename: str = "checkpoint.pth", load_best: bool = True, device: str = 'cpu'
        ) -> None:
    """
    Restores a trained PINN model and its metadata from a saved checkpoint. This function loads either
    the best model or the last training state from a checkpoint, restores training history, and updates
    the internal state of the provided PINN instance.

    Parameters
    ----------
    pinn_instance : object
        The PINN wrapper instance. Its model and training metadata will be updated in-place.
    filename : str, optional
        Path to the checkpoint file. Default is 'checkpoint.pth'.
    load_best : bool, optional
        If True and available, loads the best model weights. Otherwise loads the latest state. Default is
        True.
    device : str, optional
        Device to load the model onto ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    None
        Updates `pinn_instance` in-place with model weights and metadata.
    """
    checkpoint = torch.load(filename, map_location = torch.device(device), weights_only = False)

    # Load model weights.
    if load_best and 'best_model_state_dict' in checkpoint:
        pinn_instance.pinn.load_state_dict(checkpoint['best_model_state_dict'])
    elif 'state_dict' in checkpoint:
        pinn_instance.pinn.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"No valid model state found in {filename}")

    # Restore training info.
    pinn_instance.epoch            = checkpoint.get('epoch', -1)
    pinn_instance.loss_train       = checkpoint.get('loss_train', 'Unknown')
    pinn_instance.loss_val         = checkpoint.get('loss_val', 'Unknown')
    pinn_instance.elapsed_time     = checkpoint.get('elapsed_time', 'Unknown')
    pinn_instance.best_train_loss  = checkpoint.get('best_train_loss', pinn_instance.loss_train)
    pinn_instance.best_time        = checkpoint.get('best_time', pinn_instance.elapsed_time)
    pinn_instance.best_epoch       = checkpoint.get('best_epoch', pinn_instance.epoch)
    pinn_instance.best_val_loss    = checkpoint.get('best_val_loss', pinn_instance.loss_val)
    pinn_instance.loss_history     = checkpoint.get('loss_history', [])
    pinn_instance.val_loss_history = checkpoint.get('val_loss_history', [])

    # Restore initialization parameters (if they were saved).
    if 'params' in checkpoint:
        for key, value in checkpoint['params'].items():
            setattr(pinn_instance, key, value)

    pinn_instance.pinn.eval()


def load_full_model(checkpoint_path: str, model_class: type) -> object:
    """
    Reconstructs and loads a trained PINN model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the `.pth` checkpoint file.
    model_class : type
        Class of the PINN (e.g., UnitDiskGivenR).
    sampling_fn : Callable or None
        Sampling function. If None, the function will use the one saved in the checkpoint (if available).

    Returns
    -------
    pinn : model_class instance
        Reconstructed and loaded PINN instance.
    """
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
    params = checkpoint['params']

    model_cls     = MODEL_REGISTRY[params['model_class']]
    optimizer_cls = OPTIMIZER_REGISTRY[params['optimizer_class']]

    pinn = model_class(
        model_class         = model_cls,
        model_kwargs        = params['model_kwargs'],
        domain_kwargs       = params['domain_kwargs'],
        optimizer_class     = optimizer_cls,
        optimizer_kwargs    = params['optimizer_kwargs'],
        epochs              = params['epochs'],
        patience            = params['patience'],
        sampling_fn         = SAMPLING_REGISTRY[params['sampling_fn']],
        checkpoint_filename = os.path.basename(checkpoint_path),
)
    
    pinn.load_model(load_best = True)

    return pinn