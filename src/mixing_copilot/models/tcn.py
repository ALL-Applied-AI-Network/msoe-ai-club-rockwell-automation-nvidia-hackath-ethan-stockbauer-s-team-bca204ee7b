from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # Optional dependency.
    torch = None
    nn = None


if nn is not None:

    class CausalConv1d(nn.Conv1d):
        def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
        ):
            self.left_padding = (kernel_size - 1) * dilation
            super().__init__(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.left_padding,
                dilation=dilation,
            )

        def forward(self, values):
            output = super().forward(values)
            return output[..., : -self.left_padding] if self.left_padding else output

    class ResidualTemporalBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.network = nn.Sequential(
                CausalConv1d(channels, channels, kernel_size, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConv1d(channels, channels, kernel_size, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, values):
            return values + self.network(values)

    class CausalTCN(nn.Module):
        """Experimental causal encoder; this is not a trained contamination detector."""

        def __init__(
            self,
            input_features: int,
            channels: int = 32,
            levels: int = 4,
            kernel_size: int = 3,
            output_features: int = 1,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_projection = nn.Conv1d(input_features, channels, 1)
            self.blocks = nn.Sequential(
                *[
                    ResidualTemporalBlock(channels, kernel_size, 2**level, dropout)
                    for level in range(levels)
                ]
            )
            self.output_projection = nn.Conv1d(channels, output_features, 1)

        def forward(self, values):
            return self.output_projection(self.blocks(self.input_projection(values)))

else:

    class CausalTCN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is optional; install the project with the [tcn] extra")
