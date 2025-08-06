"""
sampling.py
-----------
Sampling utilities for collocation, boundary, and auxiliary points in square and circular domains, for
training Physics-Informed Neural Networks (PINNs) in forward and inverse problems.

Author: Ezau Faridh Torres Torres.
Date: 25 June 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module provides functions to generate training and validation points in 2D geometries, with optional
support for fixed or sampled PDE parameters. It includes:
    - Uniform sampling with boundary support in square domains using Latin Hypercube Sampling.
    - Uniform sampling in circular domains with boundary and center points.
    - Optional parameter concatenation for inverse problem settings.
    - Differentiable output tensors for autograd use in PINNs.

Functions
---------
sample_square_uniform :
    Samples collocation and boundary points in a square domain.
sample_circle_uniform_center_restriction :
    Samples interior, boundary, and center points in a circular domain.

Usage
-----
>>> from sampling import sample_square_uniform, sample_circle_uniform_center_restriction
>>> points = sample_square_uniform(dim1_min = 0, dim1_max = 1, dim2_min = 0, dim2_max = 1,
...                                interiorSize = 10000, dim1_minSize = 500, dim1_maxSize = 500,
...                                dim2_minSize = 500, dim2_maxSize = 500, valSize = 2000)
>>> points_circle = sample_circle_uniform_center_restriction(center = (0.0, 0.0), radius = 1.0,
...                                                          interiorSize = 10000, boundarySize = 1000,
...                                                          auxiliarySize = 500)

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
    Journal of Computational Physics, 378, 686-707.
- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting
    values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2),
    239-245. [Latin Hypercube Sampling]
- https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
"""
# Necessary libraries.
import torch                   # PyTorch library for tensor operations.
from scipy.stats import qmc    # Quasi-Monte Carlo sampling methods from SciPy.
from typing import Tuple, List # Type hinting for tuples and lists.

def sample_square_uniform(
        dim1_min: float, dim1_max: float, dim2_min: float, dim2_max: float, interiorSize: int,
        dim1_minSize: int, dim1_maxSize: int, dim2_minSize: int, dim2_maxSize: int, valSize: int,
        fixed_params: Tuple = None, param_domains: List[Tuple] = None,
        train: bool = True, device: str = 'cpu') -> torch.Tensor:
    """
    Samples points uniformly over a square domain, including both interior and boundary points, for
    training or validation in Physics-Informed Neural Networks (PINNs). This function generates points
    inside a 2D domain and on its four edges using Latin Hypercube Sampling and uniform random sampling.
    It supports the addition of fixed or randomly sampled parameters for parametric PINNs. If `train` is
    False, it produces validation points with equal count per region.

    Parameters
    ----------
    dim1_min : float
        Lower bound of the first input dimension (e.g., x or t).
    dim1_max : float
        Upper bound of the first input dimension.
    dim2_min : float
        Lower bound of the second input dimension (e.g., y or x).
    dim2_max : float
        Upper bound of the second input dimension.
    interiorSize : int
        Number of points to sample in the interior of the domain.
    dim1_minSize : int
        Number of boundary points on the lower edge of the first dimension (dim1 = dim1_min).
    dim1_maxSize : int
        Number of boundary points on the upper edge of the first dimension (dim1 = dim1_max).
    dim2_minSize : int
        Number of boundary points on the lower edge of the second dimension (dim2 = dim2_min).
    dim2_maxSize : int
        Number of boundary points on the upper edge of the second dimension (dim2 = dim2_max).
    valSize : int
        Total number of validation points to sample (used only if `train` is False).
    fixed_params : tuple, optional
        Fixed parameters to append to each sampled point. Useful for parametric PINNs.
    param_domains : list of tuple, optional
        List of (min, max) tuples specifying uniform sampling ranges for each parameter.
        Used to append randomly sampled parameters to each point.
    train : bool, optional
        If True, samples training points including interior and boundaries. If False, samples validation
        points. Default is True.
    device : str, optional
        Device where the resulting tensor will be stored ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2 + n_params) if parameters are used, or (N, 2) otherwise. Contains interior
        and boundary points. Requires gradients if `train` is True.
    """
    # Check if the input parameters are valid.
    if dim1_min >= dim1_max or dim2_min >= dim2_max:
        raise ValueError("Invalid domain: dim1_min must be less than dim1_max and dim2_min must be less than dim2_max.")
    
    # Calculate the interval lengths in dim1 and dim2.
    Delta_dim1 = dim1_max - dim1_min # Interval length in dim1.
    Delta_dim2 = dim2_max - dim2_min # Interval length in dim2.

    # Training points.
    if train == True:

        X = qmc.LatinHypercube(d = 2).random(n = interiorSize)  # Interior [dim1_min, dim1_max] x [dim2_min, dim2_max].
        X = torch.tensor(X, dtype = torch.float32)              # Convert to tensor.
        X[:,0] = dim1_min + Delta_dim1 * X[:,0]                 # Rescale dim1.
        X[:,1] = dim2_min + Delta_dim2 * X[:,1]                 # Rescale dim2.
        
        Y = dim2_min + Delta_dim2 * torch.rand(dim1_minSize, 2) # dim1 = dim1_min.
        Y[:,0] = dim1_min
        X = torch.vstack((X,Y))
        Y = dim2_min + Delta_dim2 * torch.rand(dim1_maxSize, 2) # dim1 = dim1_max.
        Y[:,0] = dim1_max
        X = torch.vstack((X,Y))
        Y = dim1_min + Delta_dim1 * torch.rand(dim2_minSize, 2) # dim2 = dim2_min.
        Y[:,1] = dim2_min
        X = torch.vstack((X,Y))
        Y = dim1_min + Delta_dim1 * torch.rand(dim2_maxSize, 2) # dim2 = dim2_max.
        Y[:,1] = dim2_max
        X = torch.vstack((X,Y))

        # If we have fixed parameters or parameter domains, concatenate them to the input.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X), 1)
            X = torch.cat((X, params), dim = 1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X), n_params)
            for i in range(n_params):
                params[:,i] = params[:,i] * (param_domains[i][1] - param_domains[i][0]) + param_domains[i][0]
            X = torch.cat((X, params), dim = 1)

        return X.requires_grad_(True).to(device)

    else: # Validation points.
        points_val_each_region = valSize // 5

        X_val = qmc.LatinHypercube(d = 2).random(n = points_val_each_region)  # Interior [dim1_min, dim1_max] x [dim2_min, dim2_max].
        X_val = torch.tensor(X_val, dtype = torch.float32)                    # Convert to tensor.
        X_val[:,0] = dim1_min + Delta_dim1 * X_val[:,0]                       # Rescale dim1.
        X_val[:,1] = dim2_min + Delta_dim2 * X_val[:,1]                       # Rescale dim2.

        Y_val = dim2_min + Delta_dim2 * torch.rand(points_val_each_region, 2) # dim1 = dim1_min.
        Y_val[:,0] = dim1_min
        X_val = torch.vstack((X_val,Y_val))
        Y_val = dim2_min + Delta_dim2 * torch.rand(points_val_each_region, 2) # dim1 = dim1_max.
        Y_val[:,0] = dim1_max
        X_val = torch.vstack((X_val,Y_val))
        Y_val = dim1_min + Delta_dim1 * torch.rand(points_val_each_region, 2) # dim2 = dim2_min.
        Y_val[:,1] = dim2_min
        X_val = torch.vstack((X_val,Y_val))
        Y_val = dim1_min + Delta_dim1 * torch.rand(points_val_each_region, 2) # dim2 = dim2_max.
        Y_val[:,1] = dim2_max
        X_val = torch.vstack((X_val,Y_val))

        # If we have fixed parameters or parameter domains, concatenate them to the input.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X_val), 1)
            X_val = torch.cat((X_val, params), dim = 1)

        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X_val), n_params)
            for i in range(n_params):
                params[:,i] = params[:,i] * (param_domains[i][1] - param_domains[i][0]) + param_domains[i][0]
            X_val = torch.cat((X_val, params), dim = 1)

        return X_val.to(device)

def sample_circle_uniform_center_restriction(
        center: Tuple, radius: float, interiorSize: int, boundarySize: int, auxiliarySize: int,
        valSize = None, fixed_params: Tuple = None, param_domains: List[Tuple] = None,
        train: bool = True, device: str = 'cpu') -> torch.Tensor:
    """
    Samples collocation points for a Physics-Informed Neural Network (PINN) in a circular domain centered at a given point.

    This function generates points for training or validation in PINNs over a circular domain:
    - **Interior points** are sampled using Latin Hypercube Sampling (LHS) in polar coordinates.
    - **Boundary points** are uniformly distributed along the circle's perimeter.
    - **Auxiliary points** are repeated at the center of the circle (e.g., for Neumann or source terms).
    
    Optionally, the function can append fixed or randomly sampled parameters to each point, useful for parametric PINNs.
    For validation (`train=False`), the total number of points `valSize` is divided evenly among the three regions.

    Parameters
    ----------
    center : tuple of float
        Coordinates of the center of the circle (e.g., (x₀, y₀)).
    radius : float
        Radius of the circular domain.
    interiorSize : int
        Number of interior collocation points.
    boundarySize : int
        Number of points to sample on the boundary of the circle.
    auxiliarySize : int
        Number of auxiliary points at the center of the circle.
    valSize : int, optional
        Total number of validation points to sample (used only if `train=False`).
    fixed_params : tuple, optional
        Fixed parameter values to append to each sampled point (for parametric PINNs).
    param_domains : list of tuple, optional
        List of (min, max) tuples defining the sampling range of each parameter to be randomly sampled.
    train : bool, optional
        If True, generates training points (interior, boundary, and center). If False, generates validation points.
    device : str, optional
        Target device for the returned tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2 + n_params), where N is the total number of sampled points and `n_params` is the number
        of additional parameters (if any). The tensor requires gradients if `train=True`.
    """
    # Check if the input parameters are valid.
    if radius <= 0:
        raise ValueError("Invalid radius: must be greater than zero.")
    if len(center) != 2:
        raise ValueError("Invalid center: must be a tuple of (dim1, dim2) coordinates.")

    # Training points.
    if train:
        
        # Interior points in the circle.
        sampler = qmc.LatinHypercube(d = 2)                               # Generate Latin Hypercube samples in 2D.
        sample = sampler.random(n = interiorSize)                         # Sample points uniformly in the unit square.
        theta = 2 * torch.pi * torch.tensor(sample[:,0])                  # Angle in polar coordinates.
        r = radius * torch.sqrt(torch.tensor(sample[:,1]))                # Radius in polar coordinates, scaled by the circle's radius.
        dim1_interior = r * torch.cos(theta) + center[0]                  # dim1-coordinates of interior points.
        dim2_interior = r * torch.sin(theta) + center[1]                  # dim2-coordinates of interior points.
        X_interior = torch.stack((dim1_interior, dim2_interior), dim = 1) # Stack dim1 and dim2 coordinates to form the interior points tensor.

        # Boundary points.
        theta_boundary = torch.linspace(0, 2 * torch.pi, boundarySize)    # Generate angles for boundary points.
        dim1_boundary = radius * torch.cos(theta_boundary) + center[0]    # dim1-coordinates of boundary points.
        dim2_boundary = radius * torch.sin(theta_boundary) + center[1]    # dim2-coordinates of boundary points.
        X_boundary = torch.stack((dim1_boundary, dim2_boundary), dim = 1) # Stack dim1 and dim2 coordinates to form the boundary points tensor.

        # Center points.
        X_center = torch.zeros(auxiliarySize, 2) # Auxiliary center points (origin).

        # Combine interior and boundary points.
        X = torch.cat((X_interior, X_boundary, X_center), dim = 0)
        
        # If we have fixed parameters or parameter domains, concatenate them to the input.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(len(X), 1)
            X = torch.cat((X, params), dim = 1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(len(X), n_params)
            for i in range(n_params):
                params[:,i] = params[:,i] * (param_domains[i][1] - param_domains[i][0]) + param_domains[i][0]
            X = torch.cat((X, params), dim = 1)
        
        X = X.requires_grad_(True).to(device)
        X = X.to(dtype = torch.float32)
        
        return X

    else: # Validation points.
        per_region = valSize // 3 # Number of points per region (interior, boundary, center).

        # Interior points in the circle.
        sampler = qmc.LatinHypercube(d = 2)                               # Generate Latin Hypercube samples in 2D.
        sample = sampler.random(n = per_region)                           # Sample points uniformly in the unit square.
        theta = 2 * torch.pi * torch.tensor(sample[:,0])                  # Angle in polar coordinates.
        r = radius * torch.sqrt(torch.tensor(sample[:,1]))                # Radius in polar coordinates, scaled by the circle's radius.
        dim1_interior = r * torch.cos(theta) + center[0]                  # dim1-coordinates of interior points.
        dim2_interior = r * torch.sin(theta) + center[1]                  # dim2-coordinates of interior points.
        X_interior = torch.stack((dim1_interior, dim2_interior), dim = 1) # Stack dim1 and dim2 coordinates to form the interior points tensor.

        # Boundary points.
        theta_boundary = torch.linspace(0, 2 * torch.pi, per_region)      # Generate angles for boundary points.
        dim1_boundary = radius * torch.cos(theta_boundary) + center[0]    # dim1-coordinates of boundary points.
        dim2_boundary = radius * torch.sin(theta_boundary) + center[1]    # dim2-coordinates of boundary points.
        X_boundary = torch.stack((dim1_boundary, dim2_boundary), dim = 1) # Stack dim1 and dim2 coordinates to form the boundary points tensor.

        # Center points.
        X_center = torch.zeros(per_region, 2) # Auxiliary center points (origin).

        # Combine interior and boundary points.
        X_val = torch.cat((X_interior, X_boundary, X_center), dim = 0)
        
        # If we have fixed parameters or parameter domains, concatenate them to the input.
        if fixed_params is not None:
            params = torch.tensor(fixed_params).repeat(per_region * 3, 1)
            X_val = torch.cat((X_val, params), dim = 1)
        if param_domains is not None:
            n_params = len(param_domains)
            params = torch.rand(per_region * 3, n_params)
            for i in range(n_params):
                params[:,i] = params[:,i] * (param_domains[i][1] - param_domains[i][0]) + param_domains[i][0]
            X_val = torch.cat((X_val, params), dim = 1)
        
        X_val = X_val.requires_grad_(True).to(device)
        X_val = X_val.to(dtype = torch.float32)

        return X_val
    
def generate_square_grid_points(
    dim1_min: float, dim1_max: float,
    dim2_min: float, dim2_max: float,
    interiorSize: int,
    dim1_minSize: int, dim1_maxSize: int,
    dim2_minSize: int, dim2_maxSize: int,
    valSize: int,
    fixed_params: Tuple = None,
    param_domains: List[Tuple] = None,
    train: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generates a structured grid of (x, y) points over a square domain for CNN-based PINNs.

    Parameters follow the same convention as in `sample_square_uniform` to ensure full compatibility.

    Parameters
    ----------
    [same as in sample_square_uniform]

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2 + n_params) if parametric, or (N, 2) otherwise. Structured grid if train=True,
        or validation grid if train=False.
    """

    # Check domain validity
    if dim1_min >= dim1_max or dim2_min >= dim2_max:
        raise ValueError("Invalid domain bounds.")

    # Determine grid resolution (sqrt of interiorSize)
    resolution = int(interiorSize**0.5)
    if resolution**2 != interiorSize:
        raise ValueError("For structured grid, interiorSize must be a perfect square.")

    # Create regular grid over the domain
    x = torch.linspace(dim1_min, dim1_max, resolution)
    y = torch.linspace(dim2_min, dim2_max, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')  # Shape: (res, res)
    grid = torch.stack((X, Y), dim=2).reshape(-1, 2)  # Shape: (interiorSize, 2)

    # Combine with boundary points if train=True
    if train:
        # Generate random boundary points (same as in sample_square_uniform)
        def boundary_points(num_points, fixed_dim, fixed_value, var_min, var_max, dim_index):
            points = var_min + (var_max - var_min) * torch.rand(num_points, 2)
            points[:, dim_index] = fixed_value
            return points

        boundaries = [
            boundary_points(dim1_minSize, 0, dim1_min, dim2_min, dim2_max, 1),
            boundary_points(dim1_maxSize, 0, dim1_max, dim2_min, dim2_max, 1),
            boundary_points(dim2_minSize, 1, dim2_min, dim1_min, dim1_max, 0),
            boundary_points(dim2_maxSize, 1, dim2_max, dim1_min, dim1_max, 0),
        ]
        boundary_grid = torch.vstack(boundaries)

        X_total = torch.vstack((grid, boundary_grid))

    else:
        # Validation grid
        val_res = int((valSize // 5)**0.5)
        if val_res**2 * 5 != valSize:
            raise ValueError("valSize must be divisible by 5 and yield square regions.")

        X_total = []

        def val_grid(xmin, xmax, ymin, ymax):
            x = torch.linspace(xmin, xmax, val_res)
            y = torch.linspace(ymin, ymax, val_res)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.stack((X, Y), dim=2).reshape(-1, 2)

        X_total.append(val_grid(dim1_min, dim1_max, dim2_min, dim2_max))  # Interior
        X_total.append(val_grid(dim1_min, dim1_min, dim2_min, dim2_max))  # Left
        X_total.append(val_grid(dim1_max, dim1_max, dim2_min, dim2_max))  # Right
        X_total.append(val_grid(dim1_min, dim1_max, dim2_min, dim2_min))  # Bottom
        X_total.append(val_grid(dim1_min, dim1_max, dim2_max, dim2_max))  # Top

        X_total = torch.vstack(X_total)

    # Append parameters if needed
    if fixed_params is not None:
        params = torch.tensor(fixed_params).repeat(len(X_total), 1)
        X_total = torch.cat((X_total, params), dim=1)

    if param_domains is not None:
        n_params = len(param_domains)
        params = torch.rand(len(X_total), n_params)
        for i in range(n_params):
            min_val, max_val = param_domains[i]
            params[:, i] = params[:, i] * (max_val - min_val) + min_val
        X_total = torch.cat((X_total, params), dim=1)

    return X_total.requires_grad_(train).to(device)