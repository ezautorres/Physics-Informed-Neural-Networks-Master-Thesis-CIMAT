"""
architectures.py
----------------
Neural network architectures for Physics-Informed Neural Networks (PINNs).

Author: Ezau Faridh Torres Torres.
Date: 20 August 2025.
Institution: Centro de Investigación en Matemáticas (CIMAT).

Description
-----------
This module implements several neural network architectures for solving partial
differential equations (PDEs) within the PINN framework. It provides flexible
models for both unstructured and grid-based domains:
    - MLP (Multi-Layer Perceptron) with Xavier initialization, LayerNorm,
      dropout, and customizable activation functions.
    - Sin activation wrapper for experiments with periodic features.
    - CNNPINN, a hybrid architecture combining convolutional encoders with
      fully connected layers, useful for structured PDE data.
    - ConvNet2D, a 2D convolutional encoder-decoder tailored for regular grids.

All architectures are compatible with the PINN training pipelines and can be
applied in forward and inverse PDE problems.

Classes
-------
MLP :
    Flexible fully connected neural network with configurable depth, width,
    dropout, normalization, and activation.
Sin :
    Torch module wrapper for the sine activation function.
CNNPINN :
    Convolutional encoder followed by MLP layers for structured 2D inputs.
ConvNet2D :
    Convolutional neural network designed for PINNs on structured square grids.

Usage
-----
Example usage for the MLP:
>>> from architectures import MLP
>>> model = MLP(inputSize=3, hidden_lys=[64, 64, 64], activation="tanh")
>>> output = model(torch.randn(128, 3))

Using CNNPINN:
>>> from architectures import CNNPINN
>>> model = CNNPINN(
...     input_channels=4, conv_channels=[32, 64], hidden_layers=[128, 64]
... )
>>> output = model(torch.randn(1, 4, 64, 64))

Using ConvNet2D on structured grids:
>>> from architectures import ConvNet2D
>>> model = ConvNet2D(grid_size=50, hidden_channels=16)
>>> model.forward_full_grid()     
>>> preds = model.evaluate_points(torch.rand(100, 2))

References
----------
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.
  Journal of Computational Physics, 378, 686-707.
- PyTorch documentation: https://pytorch.org/docs/stable/nn.html
"""
# Necessary libraries.
import torch                     # Tensors and autograd.
import torch.nn as nn            # Neural network layers.
import torch.nn.functional as F  # Activations and losses.

class MLP(nn.Module):
    def __init__(
        self,
        inputSize: int,
        hidden_lys: list[int],
        outputSize: int = 1,
        activation: str | nn.Module = "tanh",
        dropout: float = 0.0,
        normalization: bool = True,
    ) -> None:
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
        activation : str or torch.nn.Module, optional
            Activation function to use in hidden layers. If a string is provided,
            it must be one of {"tanh", "relu", "sigmoid", "swish"}. If a
            torch.nn.Module instance is passed, it will be used directly (default
            is "tanh").
        dropout : float, optional
            Dropout rate to use in hidden layers (default is 0.0).
        normalization : bool, optional
            Whether to apply layer normalization after each hidden layer (default
            is True).

        Attributes
        ----------
        net : torch.nn.Sequential
            The sequential model representing the neural network.
        """
        super().__init__()                 
        self.inputSize = inputSize         
        self.hidden_lys = hidden_lys       
        self.outputSize = outputSize       
        self.dropout = dropout             
        self.normalization = normalization 

        # Resolve activation.
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == "tanh":
                act_fn = nn.Tanh()
                gain = nn.init.calculate_gain("tanh")
            elif activation == "relu":
                act_fn = nn.ReLU()
                gain = nn.init.calculate_gain("relu")
            elif activation == "sigmoid":
                act_fn = nn.Sigmoid()
                gain = nn.init.calculate_gain("sigmoid")
            elif activation == "swish":
                act_fn = nn.SiLU() 
                gain = 1.0       
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif isinstance(activation, nn.Module):
            act_fn = activation
            gain = 1.0
        else:
            raise TypeError("activation must be str or nn.Module")

        layers = []
        prev_size = self.inputSize

        for hidden_sz in self.hidden_lys:
            linear = nn.Linear(prev_size, hidden_sz)
            nn.init.xavier_uniform_(linear.weight, gain=gain)  # Xavier initialization.
            nn.init.zeros_(linear.bias)  # Start with zero bias.
            layers.append(linear)

            if self.normalization:
                layers.append(nn.LayerNorm(hidden_sz))  # Stabilize training with layer norm.

            layers.append(act_fn)
            layers.append(nn.Dropout(self.dropout))

            prev_size = hidden_sz

        output_layer = nn.Linear(prev_size, self.output_size)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

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
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

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