"""
utils.py
--------
Utility functions and registries for managing PINN training, checkpoints, and
MCMC post-processing.

Author: Ezau Faridh Torres Torres.
Date: 20 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides utility functions for Physics-Informed Neural Networks
(PINNs). It includes:
    - Registries for models, optimizers, and sampling functions.
    - Checkpoint saving, loading, and inspection utilities.
    - Support for restoring full models from disk.
    - Loading and summarizing MCMC samples from CSV files.
    - Statistical reporting of inference results.

Functions
---------
get_model_info :
    Display metadata, architecture, and statistics from a saved checkpoint.
save_checkpoint :
    Save model, optimizer, and metadata to disk (and best model separately).
load_model :
    Restore a trained model and its state from checkpoint files.
load_full_model :
    Reconstruct a full PINN instance from a saved checkpoint.
load_samples_from_csv :
    Load MCMC samples and execution time from a CSV file.
summarize_results :
    Compute and print summary statistics (mean, std, quantiles) of samples.

Usage
-----
Example: saving and reloading a trained PINN model
>>> from utils import save_checkpoint, load_model, get_model_info
>>> # Save checkpoint after one epoch
>>> save_checkpoint(
...     pinn_instance=my_pinn,
...     state={"epoch": 1, "loss_train": 0.01, "loss_val": 0.02},
...     is_best=True
... )
>>>
>>> # Inspect model metadata
>>> get_model_info("my_checkpoint.pth", device="cpu")
>>>
>>> # Reload best model into PINN instance
>>> load_model(my_pinn, filename="trained_models/my_checkpoint.pth")

Example: summarizing MCMC inference results
>>> from utils import load_samples_from_csv, summarize_results
>>> samples = load_samples_from_csv("mcmc_results.csv")
>>> stats = summarize_results(samples, par_true=[1.0, 0.5], par_names=["kappa", "lambda"])
>>> print(stats["kappa"]["mean"])

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.
  Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/
- NumPy documentation: https://numpy.org/doc/
- SciPy documentation: https://docs.scipy.org/doc/scipy/
- Pandas documentation: https://pandas.pydata.org/docs/
"""
# Necessary libraries.
import numpy as np                        # Arrays and math.
import pandas as pd                       # Data handling.
import torch                              # Tensors and autograd.
import scipy.stats as stats               # Probability and stats.
import os                                 # File paths.
import sys                                # System functions.
import shutil                             # File operations.
import datetime                           # Date and time.
import random                             # Random numbers.
from typing import Callable, Sequence     # Type hints.
import torch.optim                        # Optimizers.

from architectures import MLP, ConvNet2D  # Neural network models.
from sampling import (                    # Sampling utilities.
    sample_circle_uniform_center_restriction, 
    sample_square_uniform,
    generate_square_grid_points
)
# Map string names to actual classes.
MODEL_REGISTRY = {
    "MLP": MLP,
    "ConvNet2D": ConvNet2D,
}
OPTIMIZER_REGISTRY = {
    "LBFGS": torch.optim.LBFGS,
    "Adam": torch.optim.Adam,
}
SAMPLING_REGISTRY = {
    "sample_square_uniform": sample_square_uniform,
    "sample_circle_uniform_center_restriction": sample_circle_uniform_center_restriction,
    "generate_square_grid_points": generate_square_grid_points,
}

def get_model_info(filename: str, device: str = 'cpu') -> None:
    """
    Displays metadata, architecture, and training statistics from a saved PINN
    checkpoint. This utility prints model configuration, architecture summary,
    and final training results from a checkpoint file. It is useful for inspecting
    saved models without loading them into memory.

    Parameters
    ----------
    filename : str
        Path to the checkpoint file (e.g., 'checkpoint.pth').
    device : str, optional
        Device to map tensors to when loading ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    None
        Outputs formatted information to the console. Returns the raw checkpoint
        if parameters are missing.
    """
    # Load the checkpoint file.
    filename = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])), "trained_models", filename
    )
    checkpoint = torch.load(
        filename, map_location=torch.device(device), weights_only=False
    )
    params = checkpoint.get('params', None)

    # Check if the checkpoint contains parameters.
    if params is None:
        print(f"\nNo parameters found in checkpoint {filename}")
        return checkpoint

    # Print the model information.
    print("\n" + "─"*60 + f"\nModel Info:\n" + "─"*60)
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
        print("\n" + "─"*60 + "\nModel Architecture:\n" + "─" * 60)
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
    print("\n" + "─"*60 + "\nModel Final Information:\n" + "─" * 60)
    info_lines = [
        ("Finished at Epoch", checkpoint.get('epoch', 'N/A')),
        ("Final Training Loss", checkpoint.get('loss_train', 'N/A')),
        ("Final Validation Loss", checkpoint.get('loss_val', 'N/A')),
        ("Total Training Time",
         f"{_format_val(checkpoint.get('elapsed_time', 'N/A'), '.2f')} s"),
        ("Best Epoch", checkpoint.get('best_epoch', 'N/A')),
        ("Best Training Loss", checkpoint.get('best_train_loss', 'N/A')),
        ("Best Validation Loss", checkpoint.get('best_val_loss', 'N/A')),
        ("Best Training Time",
         f"{_format_val(checkpoint.get('best_time', 'N/A'), '.2f')} s"),
    ]

    max_info_key_len = max(len(k) for k, _ in info_lines)
    for key, val in info_lines:
        print(f"{key:<{max_info_key_len}} : {_format_val(val)}")

def save_checkpoint(pinn_instance: Callable, state: dict, is_best: bool) -> None:
    """
    Saves the current state of a PINN model, optimizer, and training metadata
    to disk. This function writes a complete checkpoint including model weights,
    optimizer state, training history, hyperparameters, and reproducibility
    metadata. If the current model achieved the best performance so far, it also
    saves a separate best model copy.

    Parameters
    ----------
    pinn_instance : Callable
        The main PINN wrapper. Must include attributes like:
            - `pinn` (torch.nn.Module)
            - `checkpoint_path`, `checkpoint_filename`, `best_model_filename`
            - `model_kwargs`, `domain_kwargs`, `optimizer_class`, `optimizer_kwargs`
            - `epochs`, `patience`
    state : dict
        Dictionary with current training state (epoch, losses, model state,
        optimizer state, etc.).
    is_best : bool
        Whether the current model is the best so far. If True, a separate best
        model is saved.

    Returns
    -------
    None
        Writes checkpoint files to disk.
    """
    # Create the checkpoint path if it does not exist.
    if not os.path.exists(pinn_instance.checkpoint_path):
        os.makedirs(pinn_instance.checkpoint_path, exist_ok=True)

    # Update the best model state dictionary.
    if is_best:
        state['best_model_state_dict'] = pinn_instance.pinn.state_dict()

    # Store initialization parameters.
    state["params"] = {
        "model_kwargs": pinn_instance.model_kwargs,
        "domain_kwargs": pinn_instance.domain_kwargs,
        "model_class": pinn_instance.model_class.__name__,
        "optimizer_class": pinn_instance.optimizer_class.__name__,
        "optimizer_kwargs": pinn_instance.optimizer_kwargs,
        "epochs": pinn_instance.epochs,
        "patience": pinn_instance.patience,
        "device": str(next(pinn_instance.pinn.parameters()).device),
        "sampling_fn": pinn_instance.sampling_fn.__name__,
        "torch_version": torch.__version__,
        "random_seeds": {
            "torch": torch.initial_seed(),
            "numpy": np.random.get_state()[1][0],
            "python": random.randint(0, 100000),
        },
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_filename": pinn_instance.checkpoint_filename,
        "model_repr": str(pinn_instance.pinn),
    }

    # Save the checkpoint (includes all necessary details).
    filepath = os.path.join(
        pinn_instance.checkpoint_path, pinn_instance.checkpoint_filename
    )
    torch.save(state, filepath)

    # Save the best model as a separate file.
    if is_best:
        best_path = os.path.join(
            pinn_instance.checkpoint_path, pinn_instance.best_model_filename
        )
        shutil.copyfile(filepath, best_path)

def load_model(
    pinn_instance: Callable,
    filename: str = None,
    load_best: bool = True,
    device: str = 'cpu'
) -> None:
    """
    Restores a trained PINN model and its metadata from a saved checkpoint.
    This function loads either the best model or the last training state from
    a checkpoint, restores training history, and updates the internal state of
    the provided PINN instance.

    Parameters
    ----------
    pinn_instance : Callable
        The PINN wrapper instance. Its model and training metadata will be
        updated in-place.
    filename : str, optional
        Path to the checkpoint file. If None, it uses `trained_models/<checkpoint_filename>`
        relative to the script.
    load_best : bool, optional
        If True and available, loads the best model weights. Otherwise loads
        the latest state. Default is True.
    device : str, optional
        Device to load the model onto ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    None
        Updates `pinn_instance` in-place with model weights and metadata.
    """
    # Determine the checkpoint file to load.
    if filename is None:
        script_dir = os.path.dirname(
            os.path.abspath(sys.modules["__main__"].__file__)
        )
        filename = os.path.join(
            script_dir, "trained_models", pinn_instance.checkpoint_filename
        )
    checkpoint = torch.load(
        filename, map_location=torch.device(device), weights_only=False
    )

    # Load model weights.
    if load_best and 'best_model_state_dict' in checkpoint:
        pinn_instance.pinn.load_state_dict(checkpoint['best_model_state_dict'])
    elif 'state_dict' in checkpoint:
        pinn_instance.pinn.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"No valid model state found in {filename}")

    # Restore training info.
    pinn_instance.epoch = checkpoint.get('epoch', -1)
    pinn_instance.loss_train = checkpoint.get('loss_train', 'Unknown')
    pinn_instance.loss_val = checkpoint.get('loss_val', 'Unknown')
    pinn_instance.elapsed_time = checkpoint.get('elapsed_time', 'Unknown')
    pinn_instance.best_train_loss = checkpoint.get(
        'best_train_loss', pinn_instance.loss_train
    )
    pinn_instance.best_time = checkpoint.get(
        'best_time', pinn_instance.elapsed_time
    )
    pinn_instance.best_epoch = checkpoint.get(
        'best_epoch', pinn_instance.epoch
    )
    pinn_instance.best_val_loss = checkpoint.get(
        'best_val_loss', pinn_instance.loss_val
    )
    pinn_instance.loss_history = checkpoint.get('loss_history', [])
    pinn_instance.val_loss_history = checkpoint.get('val_loss_history', [])

    # Restore initialization parameters (optional).
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
        Path to the checkpoint file (e.g., 'trained_models/infer_conductivity_value_MLP.pth').
    model_class : type
        The class of the model to be loaded (e.g., InferringConductivityValue).

    Returns
    -------
    pinn : object
        An instance of the model class with loaded weights and training state.  
    """
    if not os.path.isabs(checkpoint_path):
        caller_path = os.path.abspath(sys.argv[0])
        script_dir = os.path.dirname(caller_path)
        checkpoint_path = os.path.join(script_dir, checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    params = checkpoint['params']

    model_cls = MODEL_REGISTRY[params['model_class']]
    optimizer_cls = OPTIMIZER_REGISTRY[params['optimizer_class']]

    pinn = model_class(
        model_class=model_cls,
        model_kwargs=params['model_kwargs'],
        domain_kwargs=params['domain_kwargs'],
        optimizer_class=optimizer_cls,
        optimizer_kwargs=params['optimizer_kwargs'],
        epochs=params['epochs'],
        patience=params['patience'],
        sampling_fn=SAMPLING_REGISTRY[params['sampling_fn']],
        checkpoint_filename=os.path.basename(checkpoint_path),
    )

    load_model(pinn, filename=checkpoint_path, load_best=True)

    return pinn

def load_samples_from_csv(filename: str) -> dict:
    """
    Load MCMC samples and execution time from a CSV file.

    Parameters
    ----------
    filename : str
        The name of the file to load the samples from.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'samples' : np.ndarray of samples (without execution time column).
        - 'execution_time' : float with execution time (from last column).
    """
    script_dir = os.path.dirname(
        os.path.abspath(sys.modules["__main__"].__file__)
    )
    path = os.path.join(script_dir, filename)
    df = pd.read_csv(path)

    samples = df.iloc[:, :-1].values
    execution_time = df.iloc[0, -1] 

    return {
        "samples": samples,
        "execution_time": float(execution_time),
    }

def summarize_results(
    samples: np.ndarray | dict,
    par_true: float | Sequence[float],
    par_names: Sequence[str] | None = None
) -> dict:
    """
    Compute and print summary statistics for MCMC samples. Supports single or
    multiple parameters.

    Parameters
    ----------
    samples : np.ndarray | dict
        The MCMC samples to analyze, or a dictionary with keys 'samples'
        and 'execution_time'.
    par_true : float | Sequence[float]
        The true value(s) of the parameter(s) for reference.
    par_names : Sequence[str] | None
        Names of the parameters to print.

    Returns
    -------
    dict
        Dictionary containing statistics for each parameter.
    """
    # Extract execution time.
    if isinstance(samples, dict):
        execution_time = samples.get("execution_time", None)
        samples = samples["samples"]
    else:
        execution_time = None

    samples = np.atleast_2d(samples)
    if samples.shape[0] < samples.shape[1]:
        samples = samples.T 

    n_params = samples.shape[1]

    # Normalize par_true to an array.
    if np.isscalar(par_true):
        par_true = [par_true] * n_params
    par_true = np.array(par_true, dtype=float)

    # Default names.
    if par_names is None:
        par_names = [f"param_{i+1}" for i in range(n_params)]

    stats_dict = {}

    for j in range(n_params):
        mean = np.mean(samples[:, j])
        median = np.median(samples[:, j])
        mode = stats.mode(samples[:, j], axis=None, keepdims=False)[0]
        std = np.std(samples[:, j])
        q16, q84 = np.percentile(samples[:, j], [16, 84])

        # Print results.
        print("\n" + 50 * "-" + f"\nResults for {par_names[j]}:\n" + 50 * "-")
        print(f"True value     : {par_true[j]:.6f}")
        print(f"Mean           : {mean:.6f}")
        print(f"Median         : {median:.6f}")
        print(f"Mode           : {mode:.6f}")
        print(f"Std            : {std:.6f}")
        print(f"Conf. interval : [{mean - std:.6f}, {mean + std:.6f}]")
        print(f"16th percent   : {q16:.6f}")
        print(f"84th percent   : {q84:.6f}")
        if execution_time is not None:
            print(f"Execution time : {execution_time:.2f} seconds")

        # Store statistics.
        stats_dict[par_names[j]] = {
            "mean": mean,
            "median": median,
            "mode": mode,
            "std": std,
            "conf_interval": (mean - std, mean + std),
            "q16": q16,
            "q84": q84,
            "execution_time": execution_time
        }

    return stats_dict