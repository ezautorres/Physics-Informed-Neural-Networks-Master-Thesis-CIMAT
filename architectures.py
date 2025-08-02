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