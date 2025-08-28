# Poisson Equation

This experiment solves the 2D Poisson equation on the unit square $\Omega:=[0,1]\times[0,1]$ using a Physics-Informed Neural Network. This experiment demonstrates how a Physics-Informed Neural Network (PINN) can solve the 2D Poisson equation, showcasing its ability to approximate PDE solutions without labeled data.

## Problem Description

The following PDE is solved:

$$
\Delta \boldsymbol{u}(x,y) = -2\pi^2 \sin(\pi x)\sin(\pi y), \qquad (x,y)\in\Omega,
$$

with homogeneous Dirichlet boundary conditions:

$$
\boldsymbol{u}(x,y)= 0, \qquad (x,y)\in\partial\Omega.
$$

The analytical solution is:

$$
\boldsymbol{u}(x,y) = \sin(\pi x) \cdot \sin(\pi y).
$$
        
## Model Summary

| Component    | Choice                                  |
|--------------|-----------------------------------------|
| Network      | MLP with 3 hidden layers (100 neurons)  |
| Optimizer    | L-BFGS (strong Wolfe line search)       |
| Activation   | Tanh (with LayerNorm after each Linear) |
| Dropout      | 0.01                                    |
| Domain       | Unit square $\Omega=[0,1]\times[0,1]$   |
| Collocation  | 500 interior, 8000 boundary points      |
| Loss         | PDE residual + boundary condition loss  |
| Weights used | $\lambda_{pde} = \lambda_{bc} = 1.0$    | 
| Error        | $1\times10^{-6}$ @ epoch 881            |
| Time         | 1131.58 s                               |

## Training Losses

<div align="center">
  <img src="loss_plot.png" alt="Training Loss" width="500"/>
</div>

## Solution Predicted by the PINN

<div align="center">
  <img src="solution_plot.png" alt="PINN Solution" width="500"/>
</div>

## Comparison with Analytical Solution

<div align="center">
  <img src="comparison_plot.png" alt="Comparison with Analytical Solution" width="1000"/>
</div>

---

*Author: Ezau Faridh Torres Torres · CIMAT · Aug 2025*