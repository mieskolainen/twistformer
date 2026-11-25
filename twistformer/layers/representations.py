# Shared attention and representation layers
#
# mikael.mieskolainen@cern.ch (2026)

"""Linear maps on multiplicity channels, never arbitrary spacetime coordinates"""

import math
import torch
from torch import nn

class RepresentationLinear(nn.Module):
    """(..., representation_dimension, channels) -> (..., representation_dimension, out)

    The same complex coefficients act on every representation coordinate, so the
    operation commutes with a common linear group action. No vector bias is allowed
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.real = nn.Parameter(torch.randn(out_channels, in_channels) / math.sqrt(2 * in_channels))
        self.imag = nn.Parameter(torch.randn(out_channels, in_channels) / math.sqrt(2 * in_channels))

    def forward(self, x):
        """
        Mix multiplicity channels without mixing representation coordinates

        Args:
            x:  Complex tensors (...,R,C), with R representation coordinates
        Returns:
            features: Complex tensors (...,R,D)
        """
        weight = torch.complex(self.real, self.imag).to(x.dtype)
        return torch.einsum("...rc,dc->...rd", x, weight)
