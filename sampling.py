"""
sampling.py
-----------
Sampling utilities and synthetic data generation for Physics-Informed Neural
Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 20 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides functions to generate collocation points and synthetic
datasets for training, validation, and evaluation of Physics-Informed Neural
Networks (PINNs). It supports sampling strategies over both square and circular
domains, with options for parametric PINNs and noisy data generation.

Functions
---------
sample_square_uniform :
    Uniform sampling of interior and boundary points over a 2D square domain.
    Supports fixed and randomly sampled parameters.
sample_circle_uniform_center_restriction :
    Sampling strategy for circular domains including interior points, boundary
    points, and auxiliary center points. Useful for Neumann problems or PDEs
    requiring source terms.
generate_synthetic_data_on_square :
    Generate noisy synthetic data inside a square domain from an analytical
    solution, with support for fixed and true parameters.
generate_synthetic_data_on_circle_boundary :
    Generate noisy synthetic data on the boundary of a circle from an analytical
    solution, with support for fixed and true parameters.

Usage
-----
Example: sampling points on a square domain for training
>>> from sampling import sample_square_uniform
>>> points = sample_square_uniform(
...     dim1_min=0.0, dim1_max=1.0,
...     dim2_min=0.0, dim2_max=1.0,
...     interiorSize=500,
...     dim1_minSize=100, dim1_maxSize=100,
...     dim2_minSize=100, dim2_maxSize=100,
...     valSize=200,
...     device="cpu"
... )
>>> points.shape
torch.Size([900, 2])

Example: sampling points on a circular domain
>>> from sampling import sample_circle_uniform_center_restriction
>>> points = sample_circle_uniform_center_restriction(
...     center=(0.0, 0.0),
...     radius=1.0,
...     interiorSize=500,
...     boundarySize=200,
...     auxiliarySize=50,
...     device="cpu"
... )
>>> points.shape
torch.Size([750, 2])

Example: generating synthetic noisy data on a square
>>> from sampling import generate_synthetic_data_on_square
>>> data_x, u_exact, u_noisy = generate_synthetic_data_on_square(
...     0.0, 1.0, 0.0, 1.0, n_points=100,
...     pinn_instance=my_pinn,
...     sigma=0.01
... )
>>> data_x.shape, u_exact.shape
((100, 2), (100,))

Example: generating synthetic noisy data on a circle boundary
>>> from sampling import generate_synthetic_data_on_circle_boundary
>>> data_x, u_exact, u_noisy = generate_synthetic_data_on_circle_boundary(
...     center=(0.0, 0.0),
...     radius=1.0,
...     n_points=50,
...     pinn_instance=my_pinn
... )
>>> data_x.shape, u_noisy[:5]
((50, 2), array([0.12, 0.08, -0.03, ...]))

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.
  Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/torch.html
- SciPy Quasi-Monte Carlo sampling:
  https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
- NumPy documentation: https://numpy.org/doc/stable/
"""
# Necessary libraries.
import numpy as np                        # Arrays and math.
import torch                              # Tensors and autograd.
from scipy.stats import qmc               # Quasi-Monte Carlo sampling.
from typing import Tuple, List, Callable  # Type hints.

def sample_square_uniform(
    dim1_min: float,
    dim1_max: float,
    dim2_min: float,
    dim2_max: float,
    interiorSize: int,
    dim1_minSize: int,
    dim1_maxSize: int,
    dim2_minSize: int,
    dim2_maxSize: int,
    valSize: int,
    fixed_params: Tuple = None,
    param_domains: List[Tuple] = None,
    train: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Samples points uniformly over a square domain, including both interior and
    boundary points, for training or validation in Physics-Informed Neural
    Networks (PINNs). This function generates points inside a 2D domain and on
    its four edges using Latin Hypercube Sampling and uniform random sampling.
    It supports the addition of fixed or randomly sampled parameters for parametric
    PINNs. If `train` is False, it produces validation points with equal count
    per region.

    Parameters
    ----------
    dim1_min : float
        Lower bound of the first input dimension (e.g., $x$ or $t$).
    dim1_max : float
        Upper bound of the first input dimension.
    dim2_min : float
        Lower bound of the second input dimension (e.g., $y$ or $x$).
    dim2_max : float
        Upper bound of the second input dimension.
    interiorSize : int
        Number of points to sample in the interior of the domain.
    dim1_minSize : int
        Number of boundary points on the lower edge of the first dimension
        (dim1 = dim1_min).
    dim1_maxSize : int
        Number of boundary points on the upper edge of the first dimension
        (dim1 = dim1_max).
    dim2_minSize : int
        Number of boundary points on the lower edge of the second dimension
        (dim2 = dim2_min).
    dim2_maxSize : int
        Number of boundary points on the upper edge of the second dimension
        (dim2 = dim2_max).
    valSize : int
        Total number of validation points to sample (used only if `train` is
        False).
    fixed_params : tuple, optional
        Fixed parameters to append to each sampled point. Useful for parametric
        PINNs.
    param_domains : list of tuple, optional
        List of (min, max) tuples specifying uniform sampling ranges for each
        parameter.
        Used to append randomly sampled parameters to each point.
    train : bool, optional
        If True, samples training points including interior and boundaries. If
        False, samples validation points. Default is True.
    device : str, optional
        Device where the resulting tensor will be stored ('cpu' or 'cuda').
        Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2 + n_params) if parameters are used, or (N, 2)
        otherwise. Contains interior and boundary points. Requires gradients
        if `train` is True.
    """
    # Check if the input parameters are valid.
    if dim1_min >= dim1_max or dim2_min >= dim2_max:
        raise ValueError(
            "Invalid domain: dim1_min must be less than dim1_max "
            "and dim2_min must be less than dim2_max."
        )
        
    # Calculate the interval lengths in dim1 and dim2.
    Delta_dim1 = dim1_max - dim1_min  # Interval length in dim1.
    Delta_dim2 = dim2_max - dim2_min  # Interval length in dim2.

    # Training points.
    if train == True:

        # Interior [dim1_min, dim1_max] x [dim2_min, dim2_max].
        X = qmc.LatinHypercube(d=2).random(n=interiorSize)  
        X = torch.tensor(X, dtype=torch.float32)         
        X[:, 0] = dim1_min + Delta_dim1 * X[:, 0]  # Rescale dim1.
        X[:, 1] = dim2_min + Delta_dim2 * X[:, 1]  # Rescale dim2.
        
        # dim1 = dim1_min.
        Y = dim2_min + Delta_dim2 * torch.rand(dim1_minSize, 2)
        Y[:, 0] = dim1_min
        X = torch.vstack((X, Y))

        # dim1 = dim1_max.
        Y = dim2_min + Delta_dim2 * torch.rand(dim1_maxSize, 2) 
        Y[:, 0] = dim1_max
        X = torch.vstack((X, Y))

        # dim2 = dim2_min.
        Y = dim1_min + Delta_dim1 * torch.rand(dim2_minSize, 2)
        Y[:, 1] = dim2_min
        X = torch.vstack((X, Y))

        # dim2 = dim2_max.
        Y = dim1_min + Delta_dim1 * torch.rand(dim2_maxSize, 2)
        Y[:, 1] = dim2_max
        X = torch.vstack((X, Y))

        # Concatenate the fixed parameters.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X), 1)
            X = torch.cat((X, params), dim=1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X), n_params)
            for i in range(n_params):
                params[:, i] = (
                    params[:, i] * (param_domains[i][1] - param_domains[i][0])
                    + param_domains[i][0]
                )
            X = torch.cat((X, params), dim=1)

        return X.requires_grad_(True).to(device)

    # Validation points.
    else:

        points_val_each_region = valSize // 5 # Point per region.

        # Interior [dim1_min, dim1_max] x [dim2_min, dim2_max].
        X_val = qmc.LatinHypercube(d=2).random(n=points_val_each_region)  
        X_val = torch.tensor(X_val, dtype=torch.float32)                  
        X_val[:, 0] = dim1_min + Delta_dim1 * X_val[:, 0]  # Rescale dim1.
        X_val[:, 1] = dim2_min + Delta_dim2 * X_val[:, 1]  # Rescale dim2.

        # dim1 = dim1_min.
        Y_val = dim2_min + Delta_dim2 * torch.rand(points_val_each_region, 2)
        Y_val[:, 0] = dim1_min
        X_val = torch.vstack((X_val, Y_val))

        # dim1 = dim1_max.
        Y_val = dim2_min + Delta_dim2 * torch.rand(points_val_each_region, 2)
        Y_val[:, 0] = dim1_max
        X_val = torch.vstack((X_val, Y_val))

        # dim2 = dim2_min.
        Y_val = dim1_min + Delta_dim1 * torch.rand(points_val_each_region, 2) 
        Y_val[:, 1] = dim2_min
        X_val = torch.vstack((X_val, Y_val))

        # dim2 = dim2_max.
        Y_val = dim1_min + Delta_dim1 * torch.rand(points_val_each_region, 2)
        Y_val[:, 1] = dim2_max
        X_val = torch.vstack((X_val,  Y_val))

        # Concatenate the fixed parameters.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X_val), 1)
            X_val = torch.cat((X_val, params), dim=1)

        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X_val), n_params)
            for i in range(n_params):
                params[:, i] = (
                    params[:, i] * (param_domains[i][1] - param_domains[i][0])
                    + param_domains[i][0]
                )
            X_val = torch.cat((X_val, params), dim=1)

        return X_val.to(device)

def sample_circle_uniform_gauge_restriction(
    center: Tuple,
    radius: float,
    interiorSize: int,
    boundarySize: int,
    auxiliarySize: int,
    valSize = None,
    fixed_params: Tuple = None,
    param_domains: List[Tuple] = None,
    train: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Samples collocation points for a Physics-Informed Neural Network (PINN) in
    a circular domain centered at a given point and mean equals zero at the
    boundary for Neumann boundary conditions. This function generates points
    for training or validation in PINNs over a circular domain:
        - **Interior points** are sampled using Latin Hypercube Sampling (LHS)
        in polar coordinates.
        - **Boundary points** are uniformly distributed along the circle's
        perimeter.
        - **Auxiliary points** are repeated at the boundary and the center of
        the circle (e.g., for Neumann or source terms).

    Optionally, the function can append fixed or randomly sampled parameters
    to each point, useful for parametric PINNs. For validation (`train=False`),
    the total number of points `valSize` is divided evenly among the 4 regions.

    Parameters
    ----------
    center : tuple of float
        Coordinates of the center of the circle (e.g., ($x, y$)).
    radius : float
        Radius of the circular domain.
    interiorSize : int
        Number of interior collocation points.
    boundarySize : int
        Number of points to sample on the boundary of the circle.
    auxiliarySize : int
        Number of auxiliary points for the boundary and center of the circle.
    valSize : int, optional
        Total number of validation points to sample (used only if `train=False`).
    fixed_params : tuple, optional
        Fixed parameter values to append to each sampled point (for parametric
        PINNs).
    param_domains : list of tuple, optional
        List of (min, max) tuples defining the sampling range of each parameter
        to be randomly sampled.
    train : bool, optional
        If True, generates training points (interior, boundary, and center). If
        False, generates validation points.
    device : str, optional
        Target device for the returned tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2 + n_params), where N is the total number of sampled
        points and `n_params` is the number of additional parameters (if any).
        The tensor requires gradients if `train=True`.
    """
    # Check if the input parameters are valid.
    if radius <= 0:
        raise ValueError("Invalid radius: must be greater than zero.")
    if len(center) != 2:
        raise ValueError(
            "Invalid center: must be a tuple of (dim1, dim2) coordinates."
        )

    # Training points.
    if train:
        
        # Interior points in the circle.
        sampler = qmc.LatinHypercube(d=2)                              
        sample = sampler.random(n=interiorSize)                         
        theta = 2 * torch.pi * torch.tensor(sample[:, 0])                  
        r = radius * torch.sqrt(torch.tensor(sample[:, 1]))               
        dim1_interior = r * torch.cos(theta) + center[0]                  
        dim2_interior = r * torch.sin(theta) + center[1]                  
        X_interior = torch.stack((dim1_interior, dim2_interior), dim=1) 

        # Boundary points.
        theta_boundary = torch.linspace(0, 2 * torch.pi, boundarySize)
        dim1_boundary = radius * torch.cos(theta_boundary) + center[0]
        dim2_boundary = radius * torch.sin(theta_boundary) + center[1]
        X_boundary = torch.stack((dim1_boundary, dim2_boundary), dim=1)

        # Gauge points.
        theta_gauge = torch.linspace(0, 2 * torch.pi, auxiliarySize)
        dim1_gauge = radius * torch.cos(theta_gauge) + center[0]
        dim2_gauge = radius * torch.sin(theta_gauge) + center[1]
        X_gauge = torch.stack((dim1_gauge, dim2_gauge), dim=1)

        # Center points.
        X_center = torch.zeros(auxiliarySize, 2)

        # Combine interior and boundary points.
        X = torch.cat((X_interior, X_boundary, X_gauge, X_center), dim=0)

        # Concatenate the fixed parameters.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X), 1)
            X = torch.cat((X, params), dim=1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X), n_params)
            for i in range(n_params):
                params[:, i] = (
                    params[:, i] * (param_domains[i][1] - param_domains[i][0])
                    + param_domains[i][0]
                )
            X = torch.cat((X, params), dim=1)

        X = X.requires_grad_(True).to(device)
        X = X.to(dtype=torch.float32)

        return X

    # Validation points.
    else:
        per_region = valSize // 4  # Points per region.

        # Interior points in the circle.
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(n=per_region)  
        theta = 2 * torch.pi * torch.tensor(sample[:, 0])                  
        r = radius * torch.sqrt(torch.tensor(sample[:, 1]))  
        dim1_interior = r * torch.cos(theta) + center[0]
        dim2_interior = r * torch.sin(theta) + center[1]
        X_interior = torch.stack((dim1_interior, dim2_interior), dim=1) 

        # Boundary points.
        theta_boundary = torch.linspace(0, 2 * torch.pi, per_region)      
        dim1_boundary = radius * torch.cos(theta_boundary) + center[0]
        dim2_boundary = radius * torch.sin(theta_boundary) + center[1]
        X_boundary = torch.stack((dim1_boundary, dim2_boundary), dim=1)

        # Gauge points.
        theta_gauge = torch.linspace(0, 2 * torch.pi, per_region)
        dim1_gauge = radius * torch.cos(theta_gauge) + center[0]
        dim2_gauge = radius * torch.sin(theta_gauge) + center[1]
        X_gauge = torch.stack((dim1_gauge, dim2_gauge), dim=1)

        # Center points.
        X_center = torch.zeros(per_region, 2)

        # Combine interior and boundary points.
        X_val = torch.cat((X_interior, X_boundary, X_gauge, X_center), dim=0)

        # Concatenate the fixed parameters.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(per_region * 4, 1)
            X_val = torch.cat((X_val, params), dim=1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(per_region * 4, n_params)
            for i in range(n_params):
                params[:, i] = (
                    params[:, i] * (param_domains[i][1] - param_domains[i][0])
                    + param_domains[i][0]
                )
            X_val = torch.cat((X_val, params), dim=1)

        X_val = X_val.requires_grad_(True).to(dtype=torch.float32)

        return X_val.to(device)

def generate_synthetic_data_on_square(
    dim1_min: float,
    dim1_max: float,
    dim2_min: float,
    dim2_max: float,
    n_points: int,
    pinn_instance: Callable, 
    fixed_params: tuple[float, ...] | None = None,
    par_true: tuple[float, ...] | None = None,
    sigma: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic data for PINN inference, supporting any number of fixed
    parameters and true parameters to be appended to the input.

    Parameters
    ----------
    dim1_min : float
        Minimum value for the first sampled dimension (e.g., $x$).
    dim1_max : float
        Maximum value for the first sampled dimension (e.g., $x$).
    dim2_min : float
        Minimum value for the second sampled dimension (e.g., $t$).
    dim2_max : float
        Maximum value for the second sampled dimension (e.g., $t$).
    n_points : int
        Number of random data points to generate.
    pinn_instance : object
        Model instance with an `.analytical_solution(torch.Tensor)` method.
    fixed_params : tuple of float, optional
        Fixed parameters to append after (dim1, dim2). Repeated for all points.
    par_true : tuple of float, optional
        True parameter values to append after the fixed parameters. Repeated
        for all points.
    sigma : float, optional
        Standard deviation of the Gaussian noise added to the analytical solution.

    Returns
    -------
    data_x : np.ndarray
        Input locations for the data (without par_true), shape (n_points, D).
    data_u_exact : np.ndarray
        Exact solution values without noise, shape (n_points,).
    data_u : np.ndarray
        Solution values with Gaussian noise, shape (n_points,).
    """
    # Randomly sample dimensions.
    dim1 = np.random.uniform(dim1_min, dim1_max, n_points)
    dim2 = np.random.uniform(dim2_min, dim2_max, n_points)
    data_x_parts = [dim1, dim2]                     

    # Add fixed parameters (if any).
    if fixed_params is not None:
        for p in fixed_params:
            data_x_parts.append(np.full_like(dim1, p, dtype=np.float32))
    data_x = np.column_stack(data_x_parts)

    # Create the input for the analytical solution.
    X_parts = [dim1, dim2]
    if fixed_params is not None:
        for p in fixed_params:
            X_parts.append(np.full_like(dim1, p, dtype=np.float32))
    if par_true is not None:
        for p in par_true:
            X_parts.append(np.full_like(dim1, p, dtype=np.float32))

    # Stack to create the full input for the analytical solution.
    X = torch.tensor(
        np.column_stack(X_parts), dtype=torch.float32,
    )

    # Compute exact solution.
    data_u_exact = (
        pinn_instance.analytical_solution(X).detach().cpu().numpy().reshape(-1)
    )

    # Add Gaussian noise.
    data_u = data_u_exact + np.random.normal(
        loc=0, scale=sigma, size=data_u_exact.shape
    )

    return data_x, data_u_exact, data_u

def generate_synthetic_data_on_circle_boundary(
    center: tuple[float, float],
    radius: float,
    n_points: int,
    pinn_instance: Callable,
    fixed_params: tuple[float, ...] | None = None,
    par_true: tuple[float, ...] | None = None,
    sigma: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic data on the *boundary* of a circle for PINN inference.
    Points are sampled uniformly along the circumference defined by `center`
    and `radius`. Supports optional fixed parameters (appended after ($x, y$))
    and true parameters for the forward model evaluation.

    Parameters
    ----------
    center : tuple of float
        ($x, y$) coordinates of the circle center.
    radius : float
        Circle radius (must be > 0).
    n_points : int
        Number of boundary data points to generate.
    pinn_instance : object
        Model instance with an `.analytical_solution(torch.Tensor)` method.
    fixed_params : tuple of float, optional
        Fixed parameters appended after ($x, y$). Repeated for all points.
    par_true : tuple of float, optional
        True parameter values appended after the fixed parameters for the
        forward evaluation.
    sigma : float, optional
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    data_x : np.ndarray
        Input locations on the boundary (without `par_true`), shape (n_points, D).
    data_u_exact : np.ndarray
        Exact solution values without noise, shape (n_points,).
    data_u : np.ndarray
        Noisy observations, shape (n_points,).
    """
    if radius <= 0:
        raise ValueError("Invalid radius: must be greater than zero.")

    # Randomly sample dimensions.
    theta = np.random.uniform(0.0, 2.0 * np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    data_x_parts = [x, y]

    # Add fixed parameters (if any).
    if fixed_params is not None:
        for p in fixed_params:
            data_x_parts.append(np.full_like(x, p, dtype=np.float32))
    data_x = np.column_stack(data_x_parts)

    # Create the input for the analytical solution.
    X_parts = [x, y]
    if fixed_params is not None:
        for p in fixed_params:
            X_parts.append(np.full_like(x, p, dtype=np.float32))
    if par_true is not None:
        for p in par_true:
            X_parts.append(np.full_like(x, p, dtype=np.float32))

    # Stack to create the full input for the analytical solution.
    X = torch.tensor(
        np.column_stack(X_parts), dtype=torch.float32
    )

    # Compute exact solution.
    data_u_exact = (
        pinn_instance.analytical_solution(X).detach().cpu().numpy().reshape(-1)
    )

    # Add Gaussian noise.
    data_u = data_u_exact + np.random.normal(
        loc=0.0, scale=sigma, size=data_u_exact.shape
    )

    return data_x, data_u_exact, data_u