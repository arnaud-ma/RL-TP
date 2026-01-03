import itertools
import operator
from collections.abc import Sequence
from functools import reduce

import torch
from torch import nn


class ConvolutionalMDP(nn.Module):
    """Neural network with convolutional and fully connected layers for MDPs
    (maximum distance profile).
    """

    def __init__(  # noqa: PLR0913
        self,
        in_size: int | tuple[int, int, int],
        out_size: int,
        *,
        layers: Sequence[int] = (),
        convs: Sequence[tuple[int, int, int, int]] | None = None,
        final_layer_activation: type[nn.Module] = nn.Identity,
        activation: type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
        max_pool: int | None = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.in_size = in_size
        self.dropout = dropout
        self.batch_norm = batch_norm

        flat_size = in_size

        # Build convolutional part
        conv_layers = []
        if convs is not None:
            if not isinstance(in_size, tuple):
                msg = "in_size must be a tuple of (channels, height, width) when using conv layers"
                raise TypeError(msg)
            conv_layers, flat_size = self._build_conv_layers(
                in_size,
                convs,
                max_pool,
                activation,
            )

        # Build fully connected part
        layer_sizes = [flat_size, *layers]
        fc_layers = [
            layer
            for in_features, out_features in itertools.pairwise(layer_sizes)
            for layer in [
                nn.Linear(in_features, out_features),
                activation(),
                self.dropout_func(dropout),
            ]
        ]

        # Build complete model
        self.model = nn.Sequential(
            *conv_layers,
            nn.Flatten() if conv_layers else nn.Identity(),
            *fc_layers,
            nn.Linear(layer_sizes[-1], out_size),
            final_layer_activation(),
        )

    @staticmethod
    def _build_conv_layers(
        in_size: tuple[int, int, int],
        convs: Sequence[tuple[int, int, int, int]],
        max_pool: int | None,
        activation: type[nn.Module],
    ):
        """Build convolutional layers and return the flattened output size."""
        layers = []
        nb_chans, h_size, w_size = in_size

        for in_chans, out_chans, kernel_size, stride in convs:
            if nb_chans != in_chans:
                msg = f"Incompatible number of channels: expected {nb_chans}, got {in_chans}"
                raise RuntimeError(msg)

            layers.extend((
                nn.Conv2d(in_chans, out_chans, kernel_size, stride=stride),
                activation(),
            ))

            h_size = ((h_size - kernel_size) / stride) + 1
            w_size = ((w_size - kernel_size) / stride) + 1
            nb_chans = out_chans

            if max_pool is not None:
                layers.append(nn.MaxPool2d(max_pool))
                h_size = (h_size - max_pool) + 1
                w_size = (w_size - max_pool) + 1

        return layers, int(nb_chans * h_size * w_size)

    @property
    def dropout_func(self) -> type[nn.Module]:
        return nn.Dropout if self.dropout > 0.0 else nn.Identity

    def forward(self, x: torch.Tensor):

        # need to reshape if in_size is a tuple (for conv layers)
        # (batch_size, channels * height * width) -> (batch_size, channels, height, width)
        if isinstance(self.in_size, tuple):
            features_per_sample = x.numel() // x.size(0)
            expected_features = reduce(operator.mul, self.in_size)
            if features_per_sample != expected_features and len(x.shape) == 2:
                x = x.view(x.size(0), *self.in_size)
        return self.model(x)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: Sequence[int] = (),
        activation: type[nn.Module] = nn.Tanh,
        final_activation: type[nn.Module] = nn.Identity,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout

        layer_sizes = [input_size, *layers]

        # Input ->
        # (Linear -> Activation -> Dropout)*len(layers)
        # -> Linear -> FinalActivation
        self.model = nn.Sequential(
            *(
                layer
                for in_features, out_features in itertools.pairwise(layer_sizes)
                for layer in [
                    nn.Linear(in_features, out_features),
                    activation(),
                    self.dropout_func(dropout),
                ]
            ),
            nn.Linear(layer_sizes[-1], output_size),
            final_activation(),
        )

    @property
    def dropout_func(self) -> type[nn.Module]:
        return nn.Dropout if self.dropout > 0.0 else nn.Identity

    def forward(self, x):
        return self.model(x)
