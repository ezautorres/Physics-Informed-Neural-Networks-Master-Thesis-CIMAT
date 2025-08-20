"""
plotting.py
-----------
Visualization utilities for Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 20 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides a collection of plotting functions tailored for analyzing
and visualizing the performance of Physics-Informed Neural Networks (PINNs).
It includes tools for visualizing training convergence, solution fields in
square and circular domains, and Bayesian uncertainty quantification results.

Functions
---------
plot_loss :
    Plot the training and validation loss history across epochs.
plot_solution_square :
    Generate 3D surface plots of PINN predictions on a square domain.
plot_solution_circle :
    Generate 3D surface plots of PINN predictions on a circular domain.
plot_comparison_contour_square :
    Compare PINN predictions, analytical solutions, and absolute errors
    over a square domain using contour plots.
plot_comparison_contour_circle :
    Compare PINN predictions, analytical solutions, and absolute errors
    over a circular domain using contour plots.
plot_joint_posteriors :
    Plot posterior histograms for parameters from one or two posterior
    distributions.
plot_corner_comparison :
    Generate corner-plot style comparisons of analytical and PINN-based
    posterior samples, including marginal histograms and joint scatter plots.

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.
  Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/nn.html
- Matplotlib documentation: https://matplotlib.org/stable/gallery/index.html
"""
# Necessary libraries.
import numpy as np                     # Arrays and math.
import os                              # File paths.
import sys                             # System functions.
import torch                           # Tensors and autograd.
import matplotlib.pyplot as plt        # Plotting.
from typing import Callable, Sequence  # Type hints.

def plot_loss(
    model_instance: Callable,
    filename: str | None = None,
    ax: plt.Axes | None = None,
    complete_training: bool = True,
) -> None:
    """
    Plots the training and validation loss history of a PINN model instance.
    This function generates a semilog plot of the loss values per epoch using
    the stored loss history within the model instance. If available, it also
    marks the best epoch found during training. It is useful for visually
    assessing convergence behavior during optimization.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model instance with attributes:
            - `loss_history` (list or array): Training loss values per epoch.
            - `val_loss_history` (list or array): Validation loss values per
            epoch.
            - `best_epoch` (int, optional): Epoch index with the best validation
            performance.
    filename : str | None, optional
        If provided, the plot will be saved to this path as a PDF.
    ax : plt.Axes | None, optional
        An existing matplotlib axis to draw the plot on. If not provided, a new
        figure will be created.
    complete_training : bool, optional
        If True, the best epoch will be highlighted on the plot (if available).
        Default is True.

    Returns
    -------
    None
        The function only produces a visual output. If `ax` is not provided, the
        figure is shown. If `filename` is provided, the plot is saved as a PDF.
    """
    # Extract loss history and best epoch from the model instance.
    loss_history = model_instance.loss_history
    val_loss_history = model_instance.val_loss_history
    best_epoch = (
        model_instance.best_epoch
        if hasattr(model_instance, "best_epoch")
        else None
    )

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_figure = True

    # Plot training and validation loss.
    epochs = list(range(1, len(loss_history) + 1))
    ax.plot(
        epochs,
        loss_history,
        label='Training loss',
        color='#00629B',
        linewidth=3
    )
    if val_loss_history is not None:
        ax.plot(
            epochs,
            val_loss_history,
            label='Validation loss',
            color='#E87722',
            linewidth=3
        )

    # Best epoch line.
    if best_epoch and complete_training:
        ax.axvline(
            best_epoch,
            linestyle="--",
            color="#75787B",
            alpha=0.7,
            label=f"Best Epoch: {best_epoch}",
            linewidth=3
        )

    # Labels and styling.
    ax.set_xlabel('Epochs', fontsize=22)
    ax.set_ylabel('Loss', fontsize=22)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)

    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        plt.savefig(path, bbox_inches='tight', pad_inches=0.4, dpi=500)
    if created_figure:
        plt.show()

def plot_solution_square(
    model_instance: Callable,
    domain_kwargs: dict,
    parameters: list | None = None,
    filename: str | None = None,
    ax: plt.Axes | None = None,
    time_dependent: bool = False,
    adjust_zlim: bool = False,
) -> None:
    """
    Plots the model prediction $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$
    over a 2D square domain. This function evaluates the PINN model over a
    structured grid defined by `domain_kwargs` and produces a 3D surface plot
    of the predicted solution. It supports both stationary and time-dependent
    problems, and optionally appends fixed parameters for parametric PINNs.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model for predictions.
    domain_kwargs : dict
        Dictionary containing the limits of the square domain. Must include:
            dim1_min : float
                Lower bound of the first input dimension (e.g., $x$ or $t$).
            dim1_max : float
                Upper bound of the first input dimension (e.g., $x$ or $t$).
            dim2_min : float
                Lower bound of the second input dimension (e.g., $y$ or $x$).
            dim2_max : float
                Upper bound of the second input dimension (e.g., $y$ or $x$).
    parameters : list | None, optional
        List of fixed parameters to append to the input grid points, used for
        parametric PINNs. Each point in the domain is evaluated with the same
        parameters.
    filename : str | None, optional
        If provided, the plot will be saved to the specified path as a PDF.
    ax : plt.Axes | None, optional
        Existing 3D axis to draw the surface plot on. If not provided, a new
        figure will be created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are
        shown as spatial ($x, y$).
    adjust_zlim : bool, optional
        If True, adjusts the z-axis limits based on the valid data range.

    Returns
    -------
    None
        Generates a 3D surface plot of the model prediction. If `filename` is
        given, saves the figure. If `ax` is not provided, the plot is displayed
        interactively.
    """
    # Create a grid over the square domain.
    eje1 = torch.linspace(
        domain_kwargs["dim1_min"], domain_kwargs["dim1_max"], 400
    )
    eje2 = torch.linspace(
        domain_kwargs["dim2_min"], domain_kwargs["dim2_max"], 400
    )
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing="ij")
    Z = torch.full(grid_1.shape, float("nan"))

    # Evaluate the model over the grid.
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            input_tensor = [grid_1[i, j], grid_2[i, j]] + (parameters or [])
            Z[i, j] = model_instance.pinn(torch.tensor(input_tensor))

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_figure = True

    # Plot the surface.
    ax.plot_surface(
        grid_1.numpy(),
        grid_2.numpy(),
        Z.detach().numpy(),
        cmap="winter",
        edgecolor="none",
    )

    # Labels and styling.
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$t$" if time_dependent else r"$y$", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)
    ax.set_xticks(np.linspace(eje1[0].item(), eje1[-1].item(), 5))
    ax.set_yticks(np.linspace(eje2[0].item(), eje2[-1].item(), 5))

    if adjust_zlim:
        Z_valid = Z[~torch.isnan(Z)]
        zmin, zmax = Z_valid.min().item(), Z_valid.max().item()
        ax.set_zlim(zmin, zmax)
        ax.set_zticks(torch.linspace(zmin, zmax, 5).tolist())

    # z-axis label depending on mode.
    if time_dependent:
        zlabel = (
            r"$\boldsymbol{\hat{u}}_{w}(x,t;\theta)$"
            if parameters
            else r"$\boldsymbol{\hat{u}}_{w}(x,t)$"
        )
    else:
        zlabel = (
            r"$\boldsymbol{\hat{u}}_{w}(x,y;\theta)$"
            if parameters
            else r"$\boldsymbol{\hat{u}}_{w}(x,y)$"
        )
    ax.set_zlabel(zlabel, fontsize=15)

    # If observed data is provided, plot it as scatter points.
    if (
        domain_kwargs.get("data_x") is not None
        and domain_kwargs.get("data_u") is not None
    ):
        X_data_np = domain_kwargs["data_x"]
        u_data_np = domain_kwargs["data_u"]
        ax.scatter(
            X_data_np[:, 0],
            X_data_np[:, 1],
            u_data_np + 0.1,
            color="red",
            label="Observed Data",
            s=50,
            zorder=30,
            depthshade=False,
        )
        ax.legend()

    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0.45, dpi=500)
    if created_figure:
        plt.show()

def plot_solution_circle(
    model_instance: Callable,
    domain_kwargs: dict,
    parameters: list | None = None,
    filename: str | None = None,
    ax: plt.Axes | None = None,
    time_dependent: bool = False,
    adjust_zlim: bool = False
) -> None:
    """
    Plots the model prediction $\boldsymbol{\hat{u}}_{w}(\mathbf{x}, t; \theta)$
    over a circular domain. This function evaluates a PINN model over a structured
    grid defined by the bounding square of a circular domain and plots the
    prediction as a 3D surface. It supports both stationary and time-dependent
    problems, as well as parametric models.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model for predictions.
    domain_kwargs : dict
        Dictionary defining the circular domain. Must include:
            center : tuple of float
                Coordinates ($x, y$) or ($x, t$) of the circle's center.
            radius : float
                Radius of the circle.
    parameters : list | None, optional
        List of fixed parameters to append to each evaluation point, for parametric
        PINNs.
    filename : str | None, optional
        If provided, the plot will be saved to this path as a PDF.
    ax : plt.Axes | None, optional
        Existing 3D axis object to render the plot. If None, a new figure will
        be created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are
        shown as spatial ($x, y$).
    adjust_zlim : bool, optional
        If True, adjusts the z-axis limits based on the valid data range.

    Returns
    -------
    None
        The function produces a 3D surface plot of the model solution. If
        `filename` is provided, the plot is saved. If no axis is passed, the
        figure is displayed.
    """
    # Extract circle parameters.
    center = domain_kwargs["center"]
    radius = domain_kwargs["radius"]

    # Create a grid over the bounding square of the circle.
    eje1 = torch.linspace(center[0] - radius, center[0] + radius, 400)
    eje2 = torch.linspace(center[1] - radius, center[1] + radius, 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing="ij")
    Z = torch.full(grid_1.shape, float('nan'))

    # Evaluate the model over the grid.
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            r = torch.sqrt(
                (grid_1[i, j] - center[0])**2 + (grid_2[i, j] - center[1])**2
            )
            if r <= radius:
                z_input = torch.tensor(
                    [grid_1[i, j], grid_2[i, j]] + (parameters or [])
                )
                with torch.no_grad():
                    Z[i, j] = model_instance.pinn(z_input).cpu()

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_figure = True

    # Plot the surface.
    ax.plot_surface(
        grid_1.numpy(),
        grid_2.numpy(),
        Z.detach().numpy(),
        cmap="winter",
        edgecolor="none"
    )

    # Labels and styling.
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$t$" if time_dependent else r"$y$", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)
    ax.set_xticks(np.linspace(eje1[0].item(), eje1[-1].item(), 5))
    ax.set_yticks(np.linspace(eje2[0].item(), eje2[-1].item(), 5))

    if adjust_zlim:
        Z_valid = Z[~torch.isnan(Z)]
        zmin, zmax = Z_valid.min().item(), Z_valid.max().item()
        ax.set_zlim(zmin, zmax)
        ax.set_zticks(torch.linspace(zmin, zmax, 5).tolist())
    
    # z-axis label depending on mode.
    if time_dependent:
        zlabel = (
            r"$\boldsymbol{\hat{u}}_{w}(x,t;\theta)$"
            if parameters
            else r"$\boldsymbol{\hat{u}}_{w}(x,t)$"
        )
    else:
        zlabel = (
            r"$\boldsymbol{\hat{u}}_{w}(x,y;\theta)$"
            if parameters
            else r"$\boldsymbol{\hat{u}}_{w}(x,y)$"
        )
    ax.set_zlabel(zlabel, fontsize=15)

    # If observed data is provided, plot it as scatter points.
    if (
        domain_kwargs.get("data_x") is not None
        and domain_kwargs.get("data_u") is not None
    ):
        X_data_np = domain_kwargs["data_x"]
        u_data_np = domain_kwargs["data_u"]
        ax.scatter(
            X_data_np[:, 0],
            X_data_np[:, 1],
            u_data_np + 0.1,
            color="red",
            label="Observed Data",
            s=50,
            zorder=30,
            depthshade=False,
        )
        ax.legend()
    
    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        fig.savefig(path, bbox_inches="tight", pad_inches=0.45, dpi=500)
    if created_figure:
        plt.show()

def plot_comparison_contour_square(
    model_instance: Callable,
    domain_kwargs: dict,
    parameters: list | None = None,
    filename: str | None = None,
    levels: int = 20,
    ax: plt.Axes | None = None,
    time_dependent: bool = False
) -> None:
    """
    Plots a contour comparison between the PINN prediction, the analytical
    solution, and their absolute error over a square domain. This function
    evaluates both the trained PINN model and the reference analytical
    solution over a structured 2D grid, then generates three contour plots:
        (i) predicted solution,
        (ii) true solution, and
        (iii) absolute error.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model for predictions.
    domain_kwargs : dict
        Dictionary specifying the square domain. Must include:
            dim1_min : float
                Lower bound of the first dimension (e.g., $x$ or $t$).
            dim1_max : float
                Upper bound of the first dimension.
            dim2_min : float
                Lower bound of the second dimension (e.g., $y$ or $x$).
            dim2_max : float
                Upper bound of the second dimension.
    parameters : list | None, optional
        Fixed parameters to append to each evaluation point. Used in parametric
        PINNs.
    filename : str | None, optional
        If provided, the resulting plot is saved to the given path in PDF format.
    levels : int, optional
        Number of contour levels for each subplot. Default is 20.
    ax : plt.Axes | None, optional
        Pre-existing array of axes for the three subplots. If None, new subplots
        are created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are
        shown as spatial ($x, y$).

    Returns
    -------
    None
        The function produces a composite figure with three contour plots. If
        `filename` is provided, the figure is saved. If `ax` is not provided,
        the plot is displayed interactively.
    """
    # Create meshgrid over the square domain.
    eje1 = torch.linspace(
        domain_kwargs["dim1_min"], domain_kwargs["dim1_max"], 400
        )
    eje2 = torch.linspace(
        domain_kwargs["dim2_min"], domain_kwargs["dim2_max"], 400
        )
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing="ij")

    # Initialize tensors for storing values.
    Z_pinn = torch.full_like(grid_1, float("nan"))
    Z_true = torch.full_like(grid_1, float("nan"))
    Z_error = torch.full_like(grid_1, float("nan"))

    # Evaluate model and analytical solution.
    for i in range(grid_1.shape[0]):
        for j in range(grid_2.shape[1]):
            z_input = torch.tensor(
                [grid_1[i, j], grid_2[i, j]] + (parameters or [])
            )
            with torch.no_grad():
                pred = model_instance.pinn(z_input).cpu()
                true = model_instance.analytical_solution(z_input.unsqueeze(0)).cpu()
                Z_pinn[i, j] = pred
                Z_true[i, j] = true
                Z_error[i, j] = torch.abs(pred - true)

    # Shared scale for solution plots.
    vmin = min(Z_pinn.min(), Z_true.min()).item()
    vmax = max(Z_pinn.max(), Z_true.max()).item()

    # Plot setup with three subplots and two colorbars.
    created_figure = False
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
        created_figure = True

    # PINN prediction.
    cs1 = axes[0].contourf(
        grid_1, grid_2, Z_pinn, levels=levels, vmin=vmin, vmax=vmax
    )
    axes[0].set_title("PINN", fontsize=20, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].tick_params(axis="both", labelsize=18)

    # Analytical solution.
    cs2 = axes[1].contourf(
        grid_1, grid_2, Z_true, levels=levels, vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Analytical Solution", fontsize=20, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].tick_params(axis="both", labelsize=18)

    # Absolute error.
    cs3 = axes[2].contourf(grid_1, grid_2, Z_error, levels=levels)
    axes[2].set_title("Absolute Error", fontsize=20, fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].tick_params(axis="both", labelsize=18)

    # Set common labels for all subplots.
    fig.supxlabel(r"$x$", fontsize=20, y=0.03)
    if time_dependent:
        fig.supylabel(r"$t$", fontsize=20, x=0.06)
    else:
        fig.supylabel(r"$y$", fontsize=20, x=0.08)

    # Colorbar for solution plots (left two).
    cbar_ax1 = fig.add_axes([0.92, 0.58, 0.015, 0.30])
    fig.colorbar(cs2, cax=cbar_ax1).set_label("Solution Scale", fontsize=15)

    # Colorbar for error plot (right).
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.30])
    fig.colorbar(cs3, cax=cbar_ax2).set_label("Absolute Error", fontsize=15)

    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])),filename
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0.4, dpi=500)
    if created_figure:
        plt.show()

def plot_comparison_contour_circle(
    model_instance: Callable,
    domain_kwargs: dict,
    parameters: list | None = None,
    filename: str | None = None,
    levels: int = 20,
    ax: plt.Axes | None = None,
    time_dependent: bool = False
) -> None:
    """
    Plots a contour comparison between the PINN prediction, the analytical
    solution, and their absolute error over a circular domain. This function
    evaluates both the trained PINN model and the reference analytical solution
    over a structured 2D grid that covers the bounding box of a circular domain.
    Three contour plots are generated:
        (i) predicted solution,
        (ii) true solution, and
        (iii) absolute error.

    Parameters
    ----------
    model_instance : Callable
        Trained PINN model for predictions.
    domain_kwargs : dict
        Dictionary specifying the circular domain. Must include:
            center : tuple of float
                Coordinates ($x, y$) or ($x, t$) of the circle's center.
            radius : float
                Radius of the circular domain.
    parameters : list | None, optional
        Fixed parameters to append to each evaluation point. Used in parametric
        PINNs.
    filename : str | None, optional
        If provided, the resulting plot is saved to the given path in PDF format.
    levels : int, optional
        Number of contour levels for each subplot. Default is 20.
    ax : plt.Axes | None, optional
        Pre-existing array of axes for the three subplots. If None, new subplots
        are created.
    time_dependent : bool, optional
        If True, labels the vertical axis as time ($t$). Otherwise, labels are
        shown as spatial ($x, y$).

    Returns
    -------
    None
        The function produces a figure with three contour plots. If `filename`
        is provided, the figure is saved. If no axis is passed, the plot is
        displayed interactively.
    """
    # Extract circle info.
    center = domain_kwargs["center"]
    radius = domain_kwargs["radius"]

    # Create meshgrid over the bounding square domain.
    eje1 = torch.linspace(center[0] - radius, center[0] + radius, 400)
    eje2 = torch.linspace(center[1] - radius, center[1] + radius, 400)
    grid_1, grid_2 = torch.meshgrid(eje1, eje2, indexing="ij")

    # Initialize tensors for storing values.
    Z_pinn = torch.full_like(grid_1, float("nan"))
    Z_true = torch.full_like(grid_1, float("nan"))
    Z_error = torch.full_like(grid_1, float("nan"))

    # Evaluate the model over the grid.
    for i in range(grid_1.shape[0]):
        for j in range(grid_2.shape[1]):
            r = torch.sqrt(
                (grid_1[i, j] - center[0])**2 + (grid_2[i, j] - center[1])**2
            )
            if r <= radius:
                z_input = torch.tensor(
                    [grid_1[i, j], grid_2[i, j]] + (parameters or [])
                )
                with torch.no_grad():
                    pred = model_instance.pinn(z_input).cpu()
                    true = model_instance.analytical_solution(
                        z_input.unsqueeze(0)
                    ).cpu()
                    Z_pinn[i, j] = pred
                    Z_true[i, j] = true
                    Z_error[i, j] = torch.abs(pred - true)

    # Shared scale for solution plots.
    #vmin = min(
    #    torch.min(Z_pinn[~Z_pinn.isnan()]),
    #    torch.min(Z_true[~Z_true.isnan()])
    #).item()
    #vmax = max(
    #    torch.max(Z_pinn[~Z_pinn.isnan()]),
    #    torch.max(Z_true[~Z_true.isnan()])
    #).item()
    vmin = min(torch.nanmin(Z_pinn), torch.nanmin(Z_true)).item()
    vmax = max(torch.nanmax(Z_pinn), torch.nanmax(Z_true)).item()

    # Plot setup with three subplots and two colorbars.
    created_figure = False
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
        created_figure = True

    # PINN prediction.
    cs1 = axes[0].contourf(
        grid_1, grid_2, Z_pinn, levels=levels, vmin=vmin, vmax=vmax
    )
    axes[0].set_title("PINN", fontsize=20, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].tick_params(axis="both", labelsize=18)

    # Analytical solution.
    cs2 = axes[1].contourf(
        grid_1, grid_2, Z_true, levels=levels, vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Analytical Solution", fontsize=20, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].tick_params(axis="both", labelsize=18)

    # Absolute error.
    cs3 = axes[2].contourf(grid_1, grid_2, Z_error, levels=levels)
    axes[2].set_title("Absolute Error", fontsize=20, fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].tick_params(axis="both", labelsize=18)

    # Set common labels for all subplots.
    fig.supxlabel(r"$x$", fontsize=20, y=0.03)
    fig.supylabel(r"t" if time_dependent else r"y", fontsize=20, x=0.06)

    # Colorbar for solution plots (left two).
    cbar_ax1 = fig.add_axes([0.92, 0.58, 0.015, 0.30])
    fig.colorbar(cs2, cax=cbar_ax1).set_label("Solution Scale", fontsize=15)

    # Colorbar for error plot (right).
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.30])
    fig.colorbar(cs3, cax=cbar_ax2).set_label("Absolute Error", fontsize=15)

    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0.4, dpi=500)
    if created_figure:
        plt.show()

def plot_joint_posteriors(
    samples1: np.ndarray,
    samples2: np.ndarray | None = None,
    par_true: float | Sequence[float] | None = None,
    par_names: str | Sequence[str] | None = None,
    bins: int = 30,
    ax: plt.Axes | None = None,
    filename: str | None = None,
    param_idx: int | None = None,
) -> None:
    """
    Plot posterior histograms for a selected parameter from one or two sets of
    samples. If `samples1` or `samples2` are 2D arrays with shape (N, n_params),
    `param_idx` must be specified to select which parameter column to plot. If
    arrays are 1D, `param_idx` is ignored.

    Parameters
    ----------
    samples1 : np.ndarray
        Samples from the first posterior. Shape (N,) or (N, n_params).
    samples2 : np.ndarray, optional
        Samples from the second posterior for comparison. Shape (N,) or (N, n_params).
    par_true : float or sequence of float, optional
        Ground-truth value(s). If a sequence is provided, the element selected
        by `param_idx` is used.
    par_names : str or sequence of str, optional
        Parameter name(s). If a sequence is provided, the element selected by
        `param_idx` is used for the x-axis label.
    bins : int, default=30
        Number of bins in the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    filename : str, optional
        File name to save the figure (PDF or PNG). Saved in the script directory.
    param_idx : int, optional
        Column index of the parameter to plot when input arrays are 2D.
    """
    def _select_1d(a: np.ndarray, which: str) -> np.ndarray:
        """Function to select 1D array from possibly 2D input."""
        a = np.asarray(a)
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            if param_idx is None:
                raise ValueError(
                    f"{which} has shape {a.shape}; "
                    f"provide param_idx to choose which parameter to plot."
                )
            return a[:, param_idx]
        raise ValueError(f"{which} must be 1D or 2D; got shape {a.shape}.")

    # Extract 1D series to plot.
    s1 = _select_1d(samples1, "samples1")
    s2 = _select_1d(samples2, "samples2") if samples2 is not None else None

    # Resolve true value for the selected parameter.
    true_val = None
    if par_true is not None:
        if isinstance(par_true, (list, tuple, np.ndarray)):
            if param_idx is None and np.ndim(par_true) != 0 and (s1.ndim == 1):
                true_val = float(np.asarray(par_true).ravel()[0])
            else:
                par_true_arr = np.asarray(par_true).ravel()
                if param_idx is None:
                    true_val = float(par_true_arr[0]) 
                else:
                    if param_idx >= par_true_arr.size:
                        raise ValueError(
                            f"par_true has size {par_true_arr.size} "
                            f"but param_idx={param_idx} requested."
                        )
                    true_val = float(par_true_arr[param_idx])
        else:
            true_val = float(par_true)

    # Resolve parameter name.
    xlabel = None
    if par_names is not None:
        if isinstance(par_names, (list, tuple)):
            if param_idx is None:
                xlabel = str(par_names[0])
            else:
                if param_idx >= len(par_names):
                    raise ValueError(
                        f"par_names has length {len(par_names)} "
                        f"but param_idx={param_idx} requested."
                    )
                xlabel = str(par_names[param_idx])
        else:
            xlabel = str(par_names)

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_figure = True

    # Plot histograms.
    ax.hist(
        s1,
        bins=bins,
        alpha=0.8,
        label="Analytical solution",
        color='#1f77b4',
        edgecolor='#1f77b4',
        density=True
    )
    if s2 is not None:
        ax.hist(
            s2,
            bins=bins,
            alpha=0.7,
            label="PINN",
            color='#ff7f0e',
            edgecolor='#ff7f0e',
            density=True
        )

    # True value line.
    if true_val is not None:
        label_str = f"True {xlabel}" if xlabel else "True value"
        ax.axvline(
            x=true_val,
            color="red",
            linestyle="-",
            linewidth=3,
            label=fr"{label_str} = {true_val:.3f}"
        )

    # Labels and styling.
    ax.legend(fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("density", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12, fontweight='bold')
    ax.grid(True)
    plt.tight_layout()

    # Save/show plot.
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        plt.savefig(path, bbox_inches='tight', pad_inches=0.4, dpi=500)
    if created_figure:
        plt.show()

def plot_corner_comparison(
    samples_analytical: np.ndarray | dict,
    samples_pinn: np.ndarray | dict,
    par_names: Sequence[str] | None = None,
    par_true: Sequence[float] | None = None,
    bins: int = 30,
    filename: str | None = None,
    burn_in: int = 250_000,
    ax: plt.Axes | None = None
) -> None:
    """
    Compare posterior samples from an analytical solution and a PINN model by
    generating a corner-plot style visualization. Marginal histograms are
    plotted along the diagonal and joint scatter plots off-diagonal.

    Parameters
    ----------
    samples_analytical : np.ndarray | dict
        Posterior samples from the analytical solution. Must be of shape
        (n_samples, n_params).
    samples_pinn : np.ndarray | dict
        Posterior samples from the PINN solution. Must be of shape
        (n_samples, n_params).
    par_names : Sequence[str] | None, optional
        Names of the parameters. If None, defaults to "par0", "par1", ...
    par_true : Sequence[float] | None, optional
        True parameter values to be plotted as red markers/lines.
    bins : int, optional
        Number of bins for histograms. Default is 30.
    filename : str | None, optional
        If provided, the figure is saved to this path.
    burn_in : int, optional
        Number of initial samples to discard. Default is 250,000.
    ax : plt.Axes | None, optional
        Pre-existing axes object. If None, new axes are created.
    """
    # Extract posterior samples after burn-in.
    samples1 = samples_analytical[burn_in:]
    samples2 = samples_pinn[burn_in:]
    n_params = samples1.shape[1]

    # Validate parameters.
    if n_params < 2:
        raise ValueError("At least 2 parameters are required.")
    
    if par_names is None or len(par_names) != n_params:
        par_names = [f"par{i}" for i in range(n_params)]

    if par_true is None:
        par_true = [None] * n_params

    # Create a new figure and axis if not provided.
    created_figure = False
    if ax is None:
        fig, axes = plt.subplots(
            n_params, n_params, figsize=(3.2 * n_params, 2.6 * n_params)
        )
        created_figure = True

    # Iterate over the parameter pairs.
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            # Skip redundant upper triangle.
            if i < j:
                ax.axis('off')
                continue

            # Diagonal: marginal histograms.
            if i == j:
                
                ax.hist(
                    samples1[:, j],
                    bins=bins,
                    density=True,
                    alpha=0.85,
                    color="#1f77b4",
                    label="Analytical"
                )
                ax.hist(
                    samples2[:, j],
                    bins=bins,
                    density=True,
                    alpha=0.75,
                    color="#ff7f0e",
                    label="PINN"
                )
                ax.axvline(par_true[j], color="red", lw=2, ls="--")
                ax.set_ylabel("density", fontsize=12)
                ax.set_xlabel(par_names[j], fontsize=12)

            # Off-diagonal: joint scatter
            else: 

                ax.scatter(
                    samples1[:, j],
                    samples1[:, i],
                    s=6,
                    alpha=0.02,
                    color="#1f77b4"
                )
                ax.scatter(
                    samples2[:, j],
                    samples2[:, i],
                    s=6,
                    alpha=0.02,
                    color="#ff7f0e"
                )
                ax.plot(
                    par_true[j], par_true[i], marker="x", color="red", ms=8, mew=2
                )
                if i == n_params - 1:
                    ax.set_xlabel(par_names[j], fontsize=12)
                if j == 0:
                    ax.set_ylabel(par_names[i], fontsize=12)
            
            ax.grid(True, alpha=0.4)

    # Labels and styling.
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.85),
        plt.Line2D([0], [0], color="#ff7f0e", lw=6, alpha=0.75),
    ]
    labels = ["Analytical", "PINN"]
    fig.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(0.75, 0.75),
        fontsize=12
    )

    # Save/show plot.
    fig.tight_layout()
    if filename:
        path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), filename
        )
        plt.savefig(path, bbox_inches="tight", dpi=500)
    
    if created_figure:
        plt.show()