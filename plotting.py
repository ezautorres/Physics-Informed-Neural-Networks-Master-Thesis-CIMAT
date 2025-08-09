"""
plotting.py
-----------
Visualization utilities for PINN training and evaluation in 2D domains (square and circular).

Author: Ezau Faridh Torres Torres.
Date: 25 June 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides tools to visualize:
    - Training and validation loss over epochs.
    - Learned solutions in square and circular domains.
    - Comparisons with analytical solutions (including error visualization).
    - 3D surface plots and 2D contour plots for qualitative analysis.

Functions
---------
plot_loss :
    Plots training and validation loss curves.
plot_solution_square :
    Visualizes a 2D solution predicted by the model over a square domain.
plot_solution_circle :
    Visualizes a 2D solution predicted by the model over a circular domain.
plot_comparison_contour_square :
    Compares predicted and analytical solutions in a square domain.
plot_comparison_contour_circle :
    Compares predicted and analytical solutions in a circular domain.

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
    Journal of Computational Physics, 378, 686-707.
"""
# Necessary libraries.
import numpy as np                    # NumPy for numerical operations.
import os, sys                        # OS and sys modules for file path operations.
import torch                          # PyTorch library for tensor operations.
import matplotlib.pyplot as plt       # Matplotlib for plotting.
from typing import Callable, Optional, Sequence, Union # Type hinting for callable functions and optional parameters.

def plot_loss(
        model_instance: Callable, filename: str = None, ax: Optional[plt.Axes] = None,
        complete_training: bool = True) -> None:
    """
    Plots the training and validation loss history of a PINN model instance. This function generates a
    semilog plot of the loss values per epoch using the stored loss history within the model instance.
    If available, it also marks the best epoch found during training. It is useful for visually assessing
    convergence behavior during optimization.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model instance with attributes:
        - `loss_history` (list or array): Training loss values per epoch.
        - `val_loss_history` (list or array): Validation loss values per epoch.
        - `best_epoch` (int, optional): Epoch index with the best validation performance.
    filename : str, optional
        If provided, the plot will be saved to this path as a PDF.
    ax : matplotlib.axes.Axes, optional
        An existing matplotlib axis to draw the plot on. If not provided, a new figure will be created.
    complete_training : bool, optional
        If True, the best epoch will be highlighted on the plot (if available). Default is True.

    Returns
    -------
    None
        The function only produces a visual output. If `ax` is not provided, the figure is shown. If
        `filename` is provided, the plot is saved as a PDF.
    """
    # Extract loss history and best epoch from the model instance.
    loss_history = model_instance.loss_history
    val_loss_history = model_instance.val_loss_history
    best_epoch = model_instance.best_epoch if hasattr(model_instance, 'best_epoch') else None

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6))
        created_figure = True

    epochs = list(range(1, len(loss_history) + 1))
    ax.plot(epochs, loss_history, label = 'Training loss', color = '#00629B', linewidth = 3)
    if val_loss_history is not None:
        ax.plot(epochs, val_loss_history, label = 'Validation loss', color = '#E87722', linewidth = 3)

    if best_epoch and complete_training:
        ax.axvline(best_epoch, linestyle = "--", color = '#75787B', alpha = 0.7, label = f"Best Epoch: {best_epoch}", linewidth = 3)

    ax.set_xlabel('Epochs', fontsize = 22)
    ax.set_ylabel('Loss', fontsize = 22)
    ax.set_yscale('log')
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.legend(fontsize = 20)
    ax.grid(True)

    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    if created_figure:
        plt.show()

def plot_solution_square(
        model_instance: Callable, domain_kwargs: dict, parameters: Optional[list] = None,
        filename: Optional[str] = None, ax: Optional[plt.Axes] = None, time_dependent: bool = False,
        adjust_zlim: bool = False) -> None:
    """
    Plots the model prediction $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$ over a 2D square domain.
    This function evaluates the PINN model over a structured grid defined by `domain_kwargs` and produces
    a 3D surface plot of the predicted solution. It supports both stationary and time-dependent problems,
    and optionally appends fixed parameters for parametric PINNs.

    Parameters
    ----------
    model_instance : Callable
        
    domain_kwargs : dict
        Dictionary containing the limits of the square domain. Must include:
            dim1_min : float
                Lower bound of the first input dimension (e.g., x or t).
            dim1_max : float
                Upper bound of the first input dimension (e.g., x or t).
            dim2_min : float
                Lower bound of the second input dimension (e.g., y or x).
            dim2_max : float
                Upper bound of the second input dimension (e.g., y or x).
    parameters : list, optional
        List of fixed parameters to append to the input grid points, used for parametric PINNs. Each
        point in the domain is evaluated with the same parameters. 
    filename : str, optional
        If provided, the plot will be saved to the specified path as a PDF.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axis to draw the surface plot on. If not provided, a new figure will be created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are shown as spatial ($x,y$).
    adjust_zlim : bool, optional
        If True, adjusts the z-axis limits based on the valid data range.

    Returns
    -------
    None
        The function generates a 3D surface plot of the model prediction. If `filename` is given, it
        saves the figure. If `ax` is not provided, the plot is displayed interactively.
    """
    # Create a grid over the square domain.
    eje1 = torch.linspace(domain_kwargs['dim1_min'], domain_kwargs['dim1_max'], 400)
    eje2 = torch.linspace(domain_kwargs['dim2_min'], domain_kwargs['dim2_max'], 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing = 'ij')
    Z = torch.full(grid_1.shape, float('nan'))

    # Iterate over the grid and compute the model output for each point.
    for i in range(len(Z)):
        for j in range(len(Z)):
            input_tensor = [grid_1[i,j], grid_2[i,j]] + (parameters if parameters else [])
            Z[i,j] = model_instance.pinn(torch.tensor(input_tensor))

    created_figure = False
    if ax is None:
        fig = plt.figure(figsize = (10,6))
        ax = fig.add_subplot(111, projection = '3d')
        created_figure = True

    # Plot the surface.
    ax.plot_surface(grid_1.numpy(), grid_2.numpy(), Z.detach().numpy(), cmap = 'winter', edgecolor = 'none')
    ax.set_xlabel(r'$x$', fontsize = 15)
    ax.set_ylabel(r'$y$', fontsize = 15) if not time_dependent else ax.set_ylabel(r'$t$', fontsize = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.tick_params(axis = 'z', labelsize = 12)
    ax.set_xticks(np.linspace(eje1[0].item(), eje1[-1].item(), 5))
    ax.set_yticks(np.linspace(eje2[0].item(), eje2[-1].item(), 5))
    if adjust_zlim:
        Z_valid = Z[~torch.isnan(Z)]
        zmin, zmax = Z_valid.min().item(), Z_valid.max().item()
        ax.set_zlim(zmin, zmax)
        ax.set_zticks(torch.linspace(zmin, zmax, 5).tolist())  
    if time_dependent:
        if parameters:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,t;\theta)$'
        else:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,t)$'
    else:
        if parameters:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,y;\theta)$'
        else:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,y)$'
    ax.set_zlabel(zlabel, fontsize = 15)

    # If observed data is provided in domain_kwargs, plot it as scatter points.
    if domain_kwargs.get("data_x") is not None and domain_kwargs.get("data_u") is not None:
        X_data_np = domain_kwargs["data_x"]
        u_data_np = domain_kwargs["data_u"]
        ax.scatter(X_data_np[:,0], X_data_np[:,1], u_data_np + .1, color = 'red', label = 'Observed Data',
                   s = 50, zorder = 30, depthshade = False)
        ax.legend()

    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.45, dpi = 500)
    if created_figure:
        plt.show()

def plot_solution_circle(
        model_instance: Callable, domain_kwargs: dict, parameters: Optional[list] = None,
        filename: Optional[str] = None, ax: Optional[plt.Axes] = None, time_dependent: bool = False,
        adjust_zlim: bool = False) -> None:
    """
    Plots the model prediction $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$ over a circular domain.
    This function evaluates a PINN model over a structured grid defined by the bounding square of a circular
    domain and plots the prediction as a 3D surface. It supports both stationary and time-dependent
    problems, as well as parametric models.

    Parameters
    ----------
    model_instance : Callable
        
    domain_kwargs : dict
        Dictionary defining the circular domain. Must include:
            center : tuple of float
                Coordinates (x,y) or (x,t) of the circle's center.
            radius : float
                Radius of the circle.
    parameters : list, optional
        List of fixed parameters to append to each evaluation point, for parametric PINNs.
    filename : str, optional
        If provided, the plot will be saved to this path as a PDF.
    ax : matplotlib.axes.Axes, optional
        Existing 3D axis object to render the plot. If None, a new figure will be created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are shown as spatial ($x,y$).
    adjust_zlim : bool, optional
        If True, adjusts the z-axis limits based on the valid data range.

    Returns
    -------
    None
        The function produces a 3D surface plot of the model solution. If `filename` is provided, the
        plot is saved. If no axis is passed, the figure is displayed.
    """
    center = domain_kwargs["center"]
    radius = domain_kwargs["radius"]

    # Create a grid over the bounding square of the circle.
    eje1 = torch.linspace(center[0] - radius, center[0] + radius, 400)
    eje2 = torch.linspace(center[1] - radius, center[1] + radius, 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing = 'ij')
    Z = torch.full(grid_1.shape, float('nan'))

    # Iterate over the grid and compute the model output for each point.
    for i in range(grid_1.shape[0]):
        for j in range(grid_2.shape[1]):
            r = torch.sqrt((grid_1[i,j] - center[0])**2 + (grid_2[i,j] - center[1])**2)
            if r <= radius:
                z_input = torch.tensor([grid_1[i,j], grid_2[i,j]] + (parameters if parameters else []))
                with torch.no_grad():
                    Z[i,j] = model_instance.pinn(z_input).cpu()

    created_figure = False
    if ax is None:
        fig = plt.figure(figsize = (10,6))
        ax = fig.add_subplot(111, projection = '3d')
        created_figure = True

    ax.plot_surface(grid_1.numpy(), grid_2.numpy(), Z.numpy(), cmap = 'winter', edgecolor = 'none')
    ax.set_xlabel(r'$x$', fontsize = 15)
    ax.set_ylabel(r'$y$', fontsize = 15) if not time_dependent else ax.set_ylabel(r'$t$', fontsize = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.tick_params(axis = 'z', labelsize = 12)
    ax.set_xticks(np.linspace(eje1[0].item(), eje1[-1].item(), 5))
    ax.set_yticks(np.linspace(eje2[0].item(), eje2[-1].item(), 5))
    if adjust_zlim:
        Z_valid = Z[~torch.isnan(Z)]
        zmin, zmax = Z_valid.min().item(), Z_valid.max().item()
        ax.set_zlim(zmin, zmax)
        ax.set_zticks(torch.linspace(zmin, zmax, 5).tolist())
    if time_dependent:
        if parameters:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,t;\theta)$'
        else:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,t)$'
    else:
        if parameters:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,y;\theta)$'
        else:
            zlabel = r'$\boldsymbol{\hat{u}}_{w}(x,y)$'
    ax.set_zlabel(zlabel, fontsize = 15)

    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        fig.savefig(path, bbox_inches = 'tight', pad_inches = 0.45, dpi = 500)
    if created_figure:
        plt.show()

def plot_comparison_contour_square(
        model_instance: Callable, domain_kwargs: dict, parameters: list = None,
        filename: str = None, levels: int = 20, ax: Optional[plt.Axes] = None,
        time_dependent: bool = False, fix: bool = False) -> None:
    """
    Plots a contour comparison between the PINN prediction, the analytical solution, and their absolute
    error over a square domain. This function evaluates both the trained PINN model and the reference
    analytical solution over a structured 2D grid, then generates three contour plots: (i) predicted
    solution, (ii) true solution, and (iii) absolute error.

    Parameters
    ----------
    model_instance : Callable
        
    domain_kwargs : dict
        Dictionary specifying the square domain. Must include:
            dim1_min : float
                Lower bound of the first dimension (e.g., x or t).
            dim1_max : float
                Upper bound of the first dimension.
            dim2_min : float
                Lower bound of the second dimension (e.g., y or x).
            dim2_max : float
                Upper bound of the second dimension.
    parameters : list, optional
        Fixed parameters to append to each evaluation point. Used in parametric PINNs.
    filename : str, optional
        If provided, the resulting plot is saved to the given path in PDF format.
    levels : int, optional
        Number of contour levels for each subplot. Default is 20.
    ax : matplotlib.axes.Axes, optional
        Pre-existing array of axes for the three subplots. If None, new subplots are created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are shown as spatial ($x,y$).

    Returns
    -------
    None
        The function produces a composite figure with three contour plots. If filename is provided, the
        figure is saved. If ax is not provided, the plot is displayed interactively.
    """
    # Create meshgrid over the square domain.
    eje1 = torch.linspace(domain_kwargs["dim1_min"], domain_kwargs["dim1_max"], 400)
    eje2 = torch.linspace(domain_kwargs["dim2_min"], domain_kwargs["dim2_max"], 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing = 'ij')

    # Initialize tensors for storing values.
    Z_pinn  = torch.full_like(grid_1, float('nan'))
    Z_true  = torch.full_like(grid_1, float('nan'))
    Z_error = torch.full_like(grid_1, float('nan'))

    # Evaluate model and analytical solution.
    for i in range(grid_1.shape[0]):
        for j in range(grid_2.shape[1]):
            z_input = torch.tensor([grid_1[i,j], grid_2[i,j]] + (parameters if parameters is not None else []))
            with torch.no_grad():
                pred = model_instance.pinn(z_input).cpu()
                true = model_instance.analytical_solution(z_input.unsqueeze(0)).cpu()
                Z_pinn[i,j] = pred
                Z_true[i,j] = true
                Z_error[i,j] = torch.abs(pred - true)

    # Shared scale for solution plots.
    vmin = min(Z_pinn.min(), Z_true.min()).item()
    vmax = max(Z_pinn.max(), Z_true.max()).item()

    # Plot setup with three subplots and two colorbars.
    created_figure = False
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize = (18,6), constrained_layout = False)
        created_figure = True

    # PINN prediction.
    cs1 = axes[0].contourf(grid_1, grid_2, Z_pinn, levels = levels, vmin = vmin, vmax = vmax)
    axes[0].set_title('Neural Network Estimation', fontsize = 20, fontweight = 'bold')
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis = 'both', labelsize = 18)

    # Analytical solution.
    cs2 = axes[1].contourf(grid_1, grid_2, Z_true, levels = levels, vmin = vmin, vmax = vmax)
    axes[1].set_title('Analytical Solution', fontsize = 20, fontweight = 'bold')
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis = 'both', labelsize = 18)

    # Absolute error.
    cs3 = axes[2].contourf(grid_1, grid_2, Z_error, levels = levels)
    axes[2].set_title('Absolute Error', fontsize = 20, fontweight = 'bold')
    axes[2].set_aspect('equal')
    axes[2].tick_params(axis = 'both', labelsize = 18)

    # Set common labels for all subplots.
    fig.supxlabel(r'$x$', fontsize = 20, y = 0.03)
    fig.supylabel(r'$y$', fontsize = 20, x = 0.08) if not time_dependent else fig.supylabel(r'$t$', fontsize = 20, x = 0.06)

    # Colorbar for solution plots (left two).
    cbar_ax1 = fig.add_axes([0.92, 0.58, 0.015, 0.30])
    fig.colorbar(cs2, cax = cbar_ax1).set_label("Solution Scale", fontsize = 15)

    # Colorbar for error plot (right).
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.30])
    fig.colorbar(cs3, cax = cbar_ax2).set_label("Absolute Error", fontsize = 15)

    # Save or display.
    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    if created_figure:
        plt.show()

def plot_comparison_contour_circle(
        model_instance: Callable, domain_kwargs: dict, parameters: list = None,
        filename: str = None, levels: int = 20, ax: Optional[plt.Axes] = None,
        time_dependent: bool = False) -> None:
    """
    Plots a contour comparison between the PINN prediction, the analytical solution, and their absolute
    error over a circular domain. This function evaluates both the PINN model and a reference analytical
    solution over a structured grid that covers the bounding box of a circular domain. Three contour plots
    are generated: predicted solution, true solution, and absolute error.

    Parameters
    ----------
    model_instance : Callable

    domain_kwargs : dict
        Dictionary specifying the circular domain. Must include:
            center : tuple of float
                Coordinates (x,y) or (x,t) of the circle's center.
            radius : float
                Radius of the circular domain.
    parameters : list, optional
        Fixed parameters to append to each evaluation point, for parametric PINNs.
    filename : str, optional
        If provided, the resulting plot is saved to the given path in PDF format.
    levels : int, optional
        Number of contour levels for each subplot. Default is 20.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axis object. If None, new subplots are created. If a single axis is passed,
        the same axis is reused.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are shown as spatial ($x,y$).

    Returns
    -------
    None
        The function produces a figure with three contour plots. If `filename` is provided, the figure is
        saved. If no axis is passed, the plot is displayed interactively.
    """
    # Extract circle info
    center = domain_kwargs["center"]
    radius = domain_kwargs["radius"]

    # Meshgrid over bounding square
    eje1 = torch.linspace(center[0] - radius, center[0] + radius, 400)
    eje2 = torch.linspace(center[1] - radius, center[1] + radius, 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing = 'ij')

    # Initialize tensors for storing values.
    Z_pinn  = torch.full_like(grid_1, float('nan'))
    Z_true  = torch.full_like(grid_1, float('nan'))
    Z_error = torch.full_like(grid_1, float('nan'))

    for i in range(grid_1.shape[0]):
        for j in range(grid_2.shape[1]):
            x, y = grid_1[i,j], grid_2[i,j]
            if torch.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius:
                input_tensor = torch.tensor([x,y] + (parameters if parameters else []), dtype = torch.float32)
                with torch.no_grad():
                    pred = model_instance.pinn(input_tensor).cpu()
                    true = model_instance.analytical_solution(input_tensor.unsqueeze(0)).cpu()
                    Z_pinn[i,j] = pred
                    Z_true[i,j] = true
                    Z_error[i,j] = torch.abs(pred - true)

    # Shared scale for solutions.
    vmin = min(torch.min(Z_pinn[~Z_pinn.isnan()]), torch.min(Z_true[~Z_true.isnan()])).item()
    vmax = max(torch.max(Z_pinn[~Z_pinn.isnan()]), torch.max(Z_true[~Z_true.isnan()])).item()

    created_figure = False
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize = (18,6), constrained_layout = False)
        created_figure = True

    # PINN prediction.
    cs1 = axes[0].contourf(grid_1, grid_2, Z_pinn, levels = levels, vmin = vmin, vmax = vmax)
    axes[0].set_title('Neural Network Estimation', fontsize = 20, fontweight = 'bold')
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis = 'both', labelsize = 18)

    # Analytical solution.
    cs2 = axes[1].contourf(grid_1, grid_2, Z_true, levels = levels, vmin = vmin, vmax = vmax)
    axes[1].set_title('Analytical Solution', fontsize = 20, fontweight = 'bold')
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis = 'both', labelsize = 18)

    # Absolute error.
    cs3 = axes[2].contourf(grid_1, grid_2, Z_error, levels = levels)
    axes[2].set_title('Absolute Error', fontsize = 20, fontweight = 'bold')
    axes[2].set_aspect('equal')
    axes[2].tick_params(axis = 'both', labelsize = 18)

    # Set common labels for all subplots.
    fig.supxlabel(r'$x$', fontsize = 20, y = 0.03)
    fig.supylabel(r'$y$', fontsize = 20, x = 0.06) if not time_dependent else fig.supylabel(r'$t$', fontsize = 20, x = 0.06)

    # Colorbar for solution plots (left two).
    cbar_ax1 = fig.add_axes([0.92, 0.58, 0.015, 0.30])
    fig.colorbar(cs2, cax = cbar_ax1).set_label("Solution Scale", fontsize = 15)

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.30])
    fig.colorbar(cs3, cax = cbar_ax2).set_label("Absolute Error", fontsize = 15)

    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    if created_figure:
        plt.show()

def plot_joint_posteriors(
        samples1: np.ndarray,
        samples2: np.ndarray = None,
        par_true: Union[float, Sequence[float], None] = None,
        par_names: Union[str, Sequence[str], None] = None,
        bins: int = 30,
        ax: Optional[plt.Axes] = None,
        filename: str = None,
        param_idx: Optional[int] = None,
    ):
    """
    Plot posterior histogram(s) for one parameter from possibly multi-parameter samples.

    If `samples1`/`samples2` are 2D (shape (N, P)), you must specify `param_idx` to
    choose which parameter column to plot. If they are 1D, `param_idx` is ignored.

    Parameters
    ----------
    samples1 : np.ndarray
        MCMC samples from the first posterior. Shape (N,) or (N, P).
    samples2 : np.ndarray, optional
        MCMC samples from the second posterior. Shape (N,) or (N, P).
    par_true : float or Sequence[float], optional
        Ground-truth value(s). If a sequence is provided and `param_idx` is set, the
        `param_idx`-th value is used.
    par_names : str or Sequence[str], optional
        Parameter name(s). If a sequence is provided and `param_idx` is set, the
        `param_idx`-th name is used for labeling.
    bins : int
        Number of bins in the histogram.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axis to draw on. If None, a new figure will be created.
    filename : str, optional
        Path to save the figure (as .pdf/.png).
    param_idx : int, optional
        Column index of the parameter to plot when samples are 2D.
    """

    def _select_1d(a: np.ndarray, which: str) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            if param_idx is None:
                raise ValueError(f"{which} has shape {a.shape}; provide param_idx to choose which parameter to plot.")
            return a[:, param_idx]
        raise ValueError(f"{which} must be 1D or 2D; got shape {a.shape}.")

    # Extract 1D series to plot
    s1 = _select_1d(samples1, "samples1")
    s2 = _select_1d(samples2, "samples2") if samples2 is not None else None

    # Resolve true value for the selected parameter
    true_val = None
    if par_true is not None:
        if isinstance(par_true, (list, tuple, np.ndarray)):
            if param_idx is None and np.ndim(par_true) != 0 and (s1.ndim == 1):
                # If user passed a sequence but no param_idx for 1D data, pick first
                true_val = float(np.asarray(par_true).ravel()[0])
            else:
                par_true_arr = np.asarray(par_true).ravel()
                if param_idx is None:
                    # samples were 1D, try to take first
                    true_val = float(par_true_arr[0])
                else:
                    if param_idx >= par_true_arr.size:
                        raise ValueError(f"par_true has size {par_true_arr.size} but param_idx={param_idx} requested.")
                    true_val = float(par_true_arr[param_idx])
        else:
            true_val = float(par_true)

    # Resolve parameter name
    xlabel = None
    if par_names is not None:
        if isinstance(par_names, (list, tuple)):
            if param_idx is None:
                xlabel = str(par_names[0])
            else:
                if param_idx >= len(par_names):
                    raise ValueError(f"par_names has length {len(par_names)} but param_idx={param_idx} requested.")
                xlabel = str(par_names[param_idx])
        else:
            xlabel = str(par_names)

    # Create axis if needed
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_figure = True

    # Plot histograms
    ax.hist(s1, bins=bins, alpha=0.8, label="Analytical solution",
            color='#1f77b4', edgecolor='#1f77b4', density=True)
    if s2 is not None:
        ax.hist(s2, bins=bins, alpha=0.7, label="PINN solution",
                color='#ff7f0e', edgecolor='#ff7f0e', density=True)

    # True value line
    if true_val is not None:
        label_str = f"True {xlabel}" if xlabel else "True value"
        ax.axvline(x=true_val, color="red", linestyle="-", linewidth=3,
                   label=fr"{label_str} = {true_val:.2f}")

    # Labels & styling
    ax.legend(fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("density", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    plt.tight_layout()

    # Save/show
    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.4, dpi=500)
    if created_figure:
        plt.show()

def plot_trajectory_2d(
    samples: Union[np.ndarray, dict],
    par_names: Sequence[str] = (r"$\alpha$", r"$\beta$"),
    par_true: Optional[Sequence[float]] = None,
    color: str = "tab:blue",
    label: str = "Chain",
    arrows_every: int = 60,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    stride: int = 1,
    show_mean: bool = True,
    ax: Optional[plt.Axes] = None,
    filename: Optional[str] = None,
):
    """
    Traza la trayectoria MCMC en 2D (param 0 vs param 1) con flechas, inicio/fin, media y punto verdadero.

    Parameters
    ----------
    samples : (N,2) array o dict con key 'samples'
        Muestras MCMC. Si es (N,P), se toma P>=2 y se usan las dos primeras columnas.
    par_names : (2,) sequence of str
        Nombres a poner en ejes (por defecto alfa y beta).
    par_true : (2,) sequence of float, optional
        Punto verdadero para marcar con 'x'.
    color : str
        Color base de la trayectoria.
    label : str
        Etiqueta para leyenda.
    arrows_every : int
        Cada cuántos pasos poner una flecha (quiver) indicando dirección temporal.
    start_idx : int
        Índice de inicio (para saltar burn-in, por ejemplo).
    end_idx : int or None
        Índice final exclusivo. None => hasta el final.
    stride : int
        Submuestreo de la cadena (cada 'stride' puntos).
    show_mean : bool
        Si True, marca el promedio posterior con un punto grande.
    ax : matplotlib.axes.Axes, optional
        Ejes donde dibujar. Si None, crea una figura nueva.
    filename : str, optional
        Si se da, guarda la figura en esa ruta.
    """
    # Obtener array (N, P)
    if isinstance(samples, dict):
        samples = samples.get("samples", samples)
    S = np.asarray(samples)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.shape[0] < S.shape[1]:
        S = S.T

    if S.shape[1] < 2:
        raise ValueError("Se requieren al menos 2 parámetros para una trayectoria 2D.")

    # Recorte y stride
    N = S.shape[0]
    if end_idx is None:
        end_idx = N
    sl = slice(start_idx, end_idx, stride)
    C = S[sl, :2]  # solo 2 primeros parámetros
    x = C[:, 0]
    y = C[:, 1]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created = True

    # Línea principal
    ax.plot(x, y, lw=0.8, alpha=0.7, color=color, label=f"{label} (path)")

    # Flechitas cada 'arrows_every'
    if len(x) > 1 and arrows_every > 0:
        step = max(1, int(arrows_every))
        ax.quiver(
            x[:-1:step], y[:-1:step],
            np.diff(x)[::step], np.diff(y)[::step],
            angles="xy", scale_units="xy", scale=1,
            width=0.002, color=color, alpha=0.6,
        )

    # Inicio / Fin
    ax.scatter(x[-1], y[-1], marker='*', s=150, color=color, edgecolor='k', zorder=6, label=f"{label} end")

    # Media posterior
    if show_mean:
        m = C.mean(axis=0)
        ax.scatter(m[0], m[1], color=color, s=110, zorder=7, label=f"{label} mean")

    # Punto verdadero
    if par_true is not None:
        ax.plot(par_true[0], par_true[1], marker='x', ms=10, mew=2, color='red', label='True', zorder=10)

    ax.set_xlabel(par_names[0])
    ax.set_ylabel(par_names[1])
    ax.set_title("MCMC trajectory in parameter space")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=500)

    if created:
        plt.show()

def plot_corner_comparison(
    samples_analytical: Union[np.ndarray, dict],
    samples_pinn: Union[np.ndarray, dict, None] = None,
    par_names: Sequence[str] | None = None,
    par_true: Sequence[float] | None = None,
    bins: int = 40,
    filename: str | None = None,
    show_upper: bool = False,
    # styling knobs
    s: float = 6.0,
):
    """
    Corner-like comparison (P>=2). Diagonal: histograms overlaid. Off-diagonal: joint scatter.
    - If show_upper=False (default), the upper-right triangle is hidden.
    - Use scatter_alpha_* and hist_alpha_* to tune transparency.
    """
    def _to_2d(arr):
        if isinstance(arr, dict):
            arr = arr.get("samples", arr)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr

    A = _to_2d(samples_analytical)
    P = A.shape[1]
    if P < 2:
        raise ValueError("plot_corner_comparison requiere al menos 2 parámetros.")

    has_pinn = samples_pinn is not None
    B = _to_2d(samples_pinn) if has_pinn else None
    if has_pinn and B.shape[1] != P:
        raise ValueError(f"samples_pinn tiene {B.shape[1]} params; se esperaban {P}.")

    if par_names is None or len(par_names) != P:
        par_names = [f"par{i}" for i in range(P)]

    if par_true is None:
        par_true = [None] * P
    else:
        par_true = list(par_true)
        if len(par_true) != P:
            raise ValueError(f"par_true tiene longitud {len(par_true)}; se esperaba {P}.")

    fig, axes = plt.subplots(P, P, figsize=(3.2 * P, 2.6 * P))

    for i in range(P):
        for j in range(P):
            ax = axes[i, j]

            # Oculta el triángulo superior (redundante)
            if (not show_upper) and (i < j):
                ax.axis('off')
                continue

            if i == j:
                # Diagonal: histogramas marginales
                ax.hist(A[:, j], bins=bins, density=True, alpha=0.8,
                        color="#1f77b4", label="Analytical")
                if has_pinn:
                    ax.hist(B[:, j], bins=bins, density=True, alpha=0.7,
                            color="#ff7f0e", label="PINN")
                if par_true[j] is not None:
                    ax.axvline(par_true[j], color="red", lw=2, ls="--")
                ax.set_ylabel("density", fontsize=12)
                ax.set_xlabel(par_names[j], fontsize=12)
            else:
                # Off-diagonal: dispersión conjunta
                ax.scatter(A[:, j], A[:, i], s=s, alpha=0.01, color="#1f77b4")
                if has_pinn:
                    ax.scatter(B[:, j], B[:, i], s=s, alpha=0.01, color="#ff7f0e")
                if par_true[j] is not None and par_true[i] is not None:
                    ax.plot(par_true[j], par_true[i], marker="x", color="red", ms=8, mew=2)
                if i == P - 1:
                    ax.set_xlabel(par_names[j], fontsize=12)
                if j == 0:
                    ax.set_ylabel(par_names[i], fontsize=12)
            ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.8)]
    labels = ["Analytical"]
    if has_pinn:
        handles.append(plt.Line2D([0], [0], color="#ff7f0e", lw=6, alpha=0.7))
        labels.append("PINN")
    fig.legend(handles, labels, loc="upper right", fontsize = 12)

    fig.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()


    # ============================================================
# Scatter-style trace plots per-parameter (iteration vs value)
# ============================================================
def plot_trace_scatter(
    samples_analytical: Union[np.ndarray, dict],
    samples_pinn: Union[np.ndarray, dict, None] = None,
    par_names: Sequence[str] | None = None,
    par_true: Sequence[float] | None = None,
    burn_in: int = 0,
    max_points: int = 20000,
    s: float = 6.0,
    alpha: float = 0.25,
    add_running_mean: bool = True,
    running_mean_window: int = 200,
    filename: str | None = None,
):
    """
    Scatter-style trace plots: iteration index on x-axis, parameter value on y-axis.

    Parameters
    ----------
    samples_analytical : array-like or dict
        MCMC samples for the analytical forward model. Shape (N, P) or dict with key 'samples'.
    samples_pinn : array-like or dict or None
        Optional MCMC samples for the PINN model. Same shape rules as above.
    par_names : sequence of str, optional
        Names of the parameters (length P). If None, uses ['par0', ...].
    par_true : sequence of float, optional
        True values per-parameter to draw horizontal lines. If None, omitted.
    burn_in : int, default 0
        Burn-in iterations to shade on the plots.
    max_points : int, default 20000
        If a chain has more than this many iterations, downsample indices to this cap for speed/clarity.
    s : float, default 6.0
        Marker size for scatter points.
    alpha : float, default 0.25
        Alpha (transparency) for scatter points.
    add_running_mean : bool, default True
        If True, overlay a simple running mean to show drift/stationarity.
    running_mean_window : int, default 200
        Window size for running mean (moving average).
    filename : str, optional
        If provided, save the figure to this path.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    def _to_2d(arr):
        if isinstance(arr, dict):
            arr = arr.get("samples", arr)
        arr = _np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr

    def _downsample_indices(N: int, cap: int) -> _np.ndarray:
        if N <= cap:
            return _np.arange(N)
        # uniform stride to reach approximately 'cap' points
        step = max(1, int(_np.floor(N / cap)))
        idx = _np.arange(0, N, step)
        if idx[-1] != N - 1:
            idx = _np.append(idx, N - 1)
        return idx

    def _running_mean(x: _np.ndarray, w: int) -> _np.ndarray:
        if w <= 1:
            return x
        c = _np.cumsum(_np.insert(x, 0, 0.0))
        out = (c[w:] - c[:-w]) / float(w)
        # pad to length N by repeating edges
        pad_left = _np.full(w // 2, out[0])
        pad_right = _np.full(x.size - out.size - pad_left.size, out[-1])
        return _np.concatenate([pad_left, out, pad_right])

    A = _to_2d(samples_analytical)
    P = A.shape[1]

    has_pinn = samples_pinn is not None
    B = _to_2d(samples_pinn) if has_pinn else None
    if has_pinn and B.shape[1] != P:
        raise ValueError(f"samples_pinn has {B.shape[1]} parameters, expected {P}.")

    if par_names is None or len(par_names) != P:
        par_names = [f"par{i}" for i in range(P)]

    if par_true is None:
        par_true = [None] * P
    else:
        par_true = list(par_true)
        if len(par_true) != P:
            raise ValueError(f"par_true has length {len(par_true)}, expected {P}.")

    ncols = 2 if has_pinn else 1
    fig, axes = _plt.subplots(nrows=P, ncols=ncols, figsize=(6 * ncols, 2.8 * P), sharex=False)
    if P == 1:
        axes = _np.atleast_2d(axes)
    if ncols == 1:
        axes = _np.column_stack([axes])

    titles = ["Analytical", "PINN"] if has_pinn else ["Analytical"]

    for j in range(P):
        for k in range(ncols):
            arr = A if k == 0 else B
            ax = axes[j, k]
            N = arr.shape[0]
            idx = _downsample_indices(N, max_points)
            ax.scatter(idx, arr[idx, j], s=s, alpha=alpha, color=("#1f77b4" if k == 0 else "#ff7f0e"))
            if burn_in > 0:
                ax.axvspan(0, burn_in, color="0.9", alpha=0.6)
            if par_true[j] is not None:
                ax.axhline(par_true[j], color="red", lw=2, ls="--")
            if add_running_mean:
                rm = _running_mean(arr[:, j], running_mean_window)
                ax.plot(_np.arange(N), rm, lw=1.2, color=("#0d3b66" if k == 0 else "#b15928"), alpha=0.9)
            if j == 0:
                ax.set_title(titles[k])
            if k == 0:
                ax.set_ylabel(par_names[j])
            if j == P - 1:
                ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if filename:
        _plt.savefig(filename, bbox_inches="tight", dpi=300)
    _plt.show()