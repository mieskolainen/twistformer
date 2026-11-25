# Lorentz and twistor algebra for pairs
#
# mikael.mieskolainen@cern.ch (2026)

"""The 15 linear permutation-equivariant pair-tensor basis operations"""

import torch
from torch import nn
from torch.nn import functional as F

class PairLinear(nn.Module):
    def __init__(self, in_dim, out_dim, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.export_diagonal = False
        self.maps = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(15)])
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, mask):
        """
        Apply the complete 15-term linear equivariant pair basis

        Args:
            x:  Scalar pair channels (B,N,N,C)
            mask:  Valid particle mask (B,N)
        Returns:
            pairs: Masked output pair channels (B,N,N,D)
        """
        b, n, _, _ = x.shape
        valid = mask[:, :, None] & mask[:, None, :]
        x = x.masked_fill(~valid[..., None], 0)
        diag = x.diagonal(dim1=1, dim2=2).transpose(1, 2)
        total, trace = x.sum((1, 2)), diag.sum(1)
        row, col = x.sum(2), x.sum(1)
        if self.normalize:
            count = mask.sum(1).clamp_min(1).to(x.dtype)
            total = total / count[:, None].square()
            trace = trace / count[:, None]
            row, col = row / count[:, None, None], col / count[:, None, None]

        weights = torch.cat((self.maps[8].weight, self.maps[9].weight))
        direct, reverse = F.linear(x, weights).chunk(2, -1)
        out = direct + reverse.transpose(1, 2) + self.bias

        def project(indices, values):
            """
            Keep reduced sums in the pair output precision under AMP

            Args:
                indices:  Equivariant basis indices with a common input
                values:  Reduced particle features
            Returns:
                projected: Scalar or one-particle output channels
            """
            weights = torch.cat([self.maps[index].weight for index in indices])
            return F.linear(values, weights).to(out.dtype).chunk(len(indices), -1)

        # Combine one-particle terms before broadcasting over both particle axes
        total_all, total_diag = project((0, 1), total)
        trace_all, trace_diag = project((2, 3), trace)
        row_receiver, row_sender, row_diag = project((4, 7, 10), row)
        col_sender, col_receiver, col_diag = project((5, 6, 11), col)
        diag_receiver, diag_sender, diag_diag = project((12, 13, 14), diag)
        receiver = row_receiver + col_receiver + diag_receiver
        sender = col_sender + row_sender + diag_sender
        diagonal = (total_diag + trace_diag)[:, None] + row_diag + col_diag + diag_diag
        out = out + (receiver + (total_all + trace_all)[:, None])[:, :, None]
        out = out + sender[:, None]
        if self.export_diagonal:
            out = out + torch.diag_embed(diagonal.transpose(1, 2)).movedim(1, -1)
        else:
            out.diagonal(dim1=1, dim2=2).add_(diagonal.transpose(1, 2))
        return out.masked_fill(~valid[..., None], 0)
