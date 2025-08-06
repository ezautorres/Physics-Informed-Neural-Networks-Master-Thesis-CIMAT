"""
architectures.py
----------------
Neural network architectures for 2D Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 05 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module defines the neural network architectures used within the PINN framework, including:
    - A customizable multi-layer perceptron (MLP) with Xavier initialization, LayerNorm, an activation
    functions.
    - A simple convolutional neural network (SimpleCNN) for grid-based input, useful in structured
    domains.

All architectures are compatible with PINN training pipelines and can be used interchangeably in forward
or inverse PDE problems.

Usage
-----
Example usage for the MLP:
>>> from architectures import MLP
>>> model = MLP(inputSize = 3, hidden_lys = [64, 64, 64])
>>> output = model(torch.randn(128, 3))

Example usage for the CNN:
>>> from architectures import SimpleCNN
>>> model = SimpleCNN(input_channels = 1, output_channels = 1)
>>> output = model(torch.randn(16, 1, 64, 64))

Classes
-------
MLP :
    Feedforward neural network with configurable depth and width.
SimpleCNN :
    Shallow CNN suitable for structured 2D input (e.g., grids from finite differences).

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
    Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/nn.html
"""
# Necessary libraries.
import torch.nn as nn # Neural network module.
import torch          # PyTorch library.

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) architecture for 2D PINNs.

    Parameters
    ----------
    inputSize : int
        Size of the input vector (e.g., 2 + number of parameters for PDEs).
    hidden_lys : list of int
        Sizes of hidden layers.
    outputSize : int, optional
        Size of the output layer (default is 1).

    Attributes
    ----------
    net : torch.nn.Sequential
        The sequential model representing the neural network.
    """
    def __init__(self, inputSize: int, hidden_lys: list, outputSize: int = 1):
        super().__init__()            # Initialize the parent class.
        self.inputSize  = inputSize   # Input size of the neural network. If we have parameters, the input size is 2 + n_params.
        self.hidden_lys = hidden_lys  # Hidden layer sizes of the neural network.
        self.outputSize = outputSize  # Output size of the neural network (default = 1).
        layers = []                   # List to store the layers of the neural network.
        
        # Initialize the neural network layers.
        prev_size = self.inputSize
        for hidden_sz in self.hidden_lys:
            linear = nn.Linear(prev_size, hidden_sz)                                      # Linear layer.
            nn.init.xavier_uniform_(linear.weight, gain = nn.init.calculate_gain("tanh")) # Initialize the weights of the linear layer using Xavier uniform initialization with Tanh gain. 
            nn.init.zeros_(linear.bias)                                                   # Initialize the bias of the linear layer to zero.
            layers.append(linear)                                                         # Append the linear layer to the list of layers.
            layers.append(nn.LayerNorm(hidden_sz))                                        # Layer normalization for the hidden layer.
            layers.append(nn.Tanh())                                                      # Tanh activation function for the hidden layer.
            layers.append(nn.Dropout(0.0))                                                # Dropout layer with a dropout rate of 0.0 to prevent overfitting.
            prev_size = hidden_sz
        output_layer = nn.Linear(prev_size, self.outputSize) # Output layer.
        nn.init.xavier_uniform_(output_layer.weight)         # Initialize the weights of the output layer.
        nn.init.zeros_(output_layer.bias)                    # Initialize the bias of the output layer.
        layers.append(output_layer)                          # Append the output layer to the list of layers.
        self.net = nn.Sequential(*layers)                    # Neural network model.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, inputSize).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, outputSize).
        """
        return self.net(x)

class CNNPINN(nn.Module):
    def __init__(self, input_channels=4, conv_channels=[32, 64, 128], kernel_size=3,
                 hidden_layers=[100], output_size=1):
        super(CNNPINN, self).__init__()

        # Convolutional layers
        convs = []
        in_ch = input_channels
        for out_ch in conv_channels:
            convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            convs.append(nn.ReLU())
            convs.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)

        # Final linear layers after flattening
        conv_out_channels = conv_channels[-1]
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_channels, hidden_layers[0]),
            nn.Tanh(),
            *[layer for i in range(1, len(hidden_layers))
              for layer in (nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.Tanh())],
            nn.Linear(hidden_layers[-1], output_size)
        )

    def forward(self, x):
        """
        x: (B, C, H, W), typically (1, 4, H, W)
        returns: (H, W)
        """
        features = self.conv(x)  # (B, C_out, H, W)
        features = features.permute(0, 2, 3, 1).squeeze(0)  # (H, W, C_out)
        out = self.mlp(features)  # (H, W, output_size)
        return out.squeeze(-1)  # (H, W)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet2D(nn.Module):
    """
    2D Convolutional Neural Network for structured PINNs.

    Assumes that input points lie on a regular grid (e.g., 100x100 = 10,000 points).
    """

    def __init__(self, grid_size: int = 100, hidden_channels: int = 32):
        """
        Parameters
        ----------
        grid_size : int
            Number of points per axis (e.g., 100 for 100x100 grid).
        hidden_channels : int
            Number of hidden channels in the convolutional layers.
        """
        super(ConvNet2D, self).__init__()

        self.grid_size = grid_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )

        # Will be populated in forward_full_grid()
        self.full_grid = None
        self.full_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model assuming x is structured grid (flattened).
        """
        assert x.shape[0] == self.grid_size**2 and x.shape[1] == 2, \
            f"Expected input of shape ({self.grid_size**2}, 2), got {x.shape}"

        # Reshape into grid
        x1 = x[:, 0].reshape(self.grid_size, self.grid_size)
        x2 = x[:, 1].reshape(self.grid_size, self.grid_size)

        # Stack as input channel
        input_tensor = torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)  # Just to test
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        out = self.encoder(input_tensor)  # [1,1,H,W]
        return out.view(-1, 1)  # Flatten back to (N,1)

    def forward_full_grid(self):
        """
        Evaluates the CNN over a structured grid and stores the output.

        This must be called before evaluate_points().
        """
        lin = torch.linspace(0, 1, self.grid_size)
        x, y = torch.meshgrid(lin, lin, indexing='ij')
        grid = torch.stack([x, y], dim=-1).view(-1, 2).to(next(self.parameters()).device)

        self.full_grid = grid  # (N,2)
        self.full_output = self(grid).view(1, 1, self.grid_size, self.grid_size)  # (1,1,H,W)

    def evaluate_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Interpolates arbitrary points from the CNN output on the structured grid.
        """
        if self.full_grid is None or self.full_output is None:
            raise RuntimeError("You must call forward_full_grid() before evaluate_points().")

        # Normalize points to [-1, 1] for grid_sample
        coords = 2.0 * points - 1.0
        coords = coords.view(1, -1, 1, 2)  # [1, N, 1, 2] for grid_sample

        interpolated = F.grid_sample(self.full_output, coords, mode='bilinear', align_corners=True)
        return interpolated.view(-1, 1)