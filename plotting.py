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
from typing import Callable, Optional # Type hinting for callable functions and optional parameters.

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
        samples1: np.ndarray, samples2: np.ndarray = None, par_true: float = None, par_names: str = None,
        bins: int = 30, ax: Optional[plt.Axes] = None, filename: str = None
    ):
    """
    Plot two posterior histograms (e.g., from MCMC samples) on the same axes.

    Parameters
    ----------
    samples1 : np.ndarray
        MCMC samples from the first posterior.
    samples2 : np.ndarray
        MCMC samples from the second posterior.
    par_true : float, optional
        Ground truth value to mark with a vertical line.
    par_names : str, optional
        Name of the parameter being inferred, used for labeling.
    bins : int
        Number of bins in the histogram.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axis to draw the histograms on. If None, a new figure will be created.
    filename : str, optional
        Path to save the figure (as .pdf).
    """
    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6))
        created_figure = True
    
    ax.hist(samples1.flatten(), bins = bins, alpha = 0.8, label = "Analytical solution",
            color = '#1f77b4', edgecolor = '#1f77b4', density = True)
    if samples2 is not None:
        ax.hist(samples2.flatten(), bins = bins, alpha = 0.7, label = "PINN solution",
                color = '#ff7f0e', edgecolor = '#ff7f0e', density = True)

    if par_true is not None:
        ax.axvline(x = par_true, color = "red", linestyle = "-", linewidth = 3,
                   label = fr"True {par_names} = {par_true}")

    ax.legend(fontsize = 14)
    if par_names:
        ax.set_xlabel(par_names, fontsize = 14)
    ax.set_ylabel("density", fontsize = 14)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.grid(True)
    plt.tight_layout()

    if filename:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename)
        plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    if created_figure:
        plt.show()