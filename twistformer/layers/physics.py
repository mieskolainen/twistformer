# Shared attention and representation layers
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch import nn
from twistformer.geometry.lorentz import dot
from .common import mlp
from twistformer.api.config import ModelConfig

class JetBias(nn.Module):
    """Jet-frame pair features with per-head projections and masked row centering"""

    def __init__(self, heads, config=None):
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.jet_scale_init == 0 and cfg.jet_block_scale_init == 0:
            raise ValueError("Jet bias scales cannot both be zero at initialization")
        self.features, self.betas = tuple(cfg.jet_features), tuple(cfg.jet_eec_betas)
        self.radius, self.exponent, self.clamp = (cfg.jet_radius, cfg.jet_exponent, cfg.jet_clamp)
        self.grad_geometry = cfg.jet_grad_geometry
        channels = sum(name != "eec" for name in self.features) + (len(self.betas) if "eec" in self.features else 0)
        self.network = mlp(channels, cfg.jet_hidden, cfg.jet_hidden, norm="layer")
        self.to_bias = nn.Linear(cfg.jet_hidden, heads if cfg.jet_head_mode == "per_head" else 1)
        self.head_gain = nn.Parameter(torch.ones(heads)) if cfg.jet_head_mode == "shared" and cfg.jet_head_gains else None
        # A zero scale supplies the unbiased start and needs a nonzero projection
        if cfg.jet_scale_init != 0 and cfg.jet_block_scale_init != 0:
            nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)
        self.scale = nn.Parameter(torch.tensor(cfg.jet_scale_init))

    def forward(self, p, mask):
        """
        Compute invariant, angular, anti-kt, Lund and EEC pair features

        Args:
            p:  Physical four-momenta (B,N,4), ordered E,px,py,pz
            mask:  Valid particle mask (B,N)
        Returns:
            bias: Centered per-head bias, zero on padding and self pairs
        """
        with torch.autocast(device_type=p.device.type, enabled=False):
            p = (p if self.grad_geometry else p.detach()).double().masked_fill(~mask[..., None], 0)
            e, px, py, pz = p.unbind(-1)
            eps = 1e-8
            pt = (px.square() + py.square()).clamp_min(eps**2).sqrt()
            zero_pt = px.square() + py.square() <= eps**2
            eta = torch.asinh(pz / pt)
            phi = torch.atan2(py.masked_fill(zero_pt, 0), px.masked_fill(zero_pt, 1))
            dphi = (phi[:, :, None] - phi[:, None, :] + torch.pi) % (2 * torch.pi) - torch.pi
            dr2 = (eta[:, :, None] - eta[:, None, :]).square() + dphi.square()
            dr = (dr2 + eps).sqrt()
            spatial = p[..., 1:] @ p[..., 1:].transpose(1, 2)
            ab = e[:, :, None] * e[:, None, :] - spatial
            mass = dot(p, p)
            norm = (p[..., 1:].square().sum(-1) + eps).sqrt()
            pti, ptj = pt[:, :, None], pt[:, None, :]
            z = pt / ((pt * mask).sum(-1, keepdim=True) + eps)
            zprod = z[:, :, None] * z[:, None, :]
            channels = {
                "p4dot": ab,
                "m2": mass[:, :, None] + mass[:, None, :] + 2 * ab,
                "dR": dr / self.radius,
                "akt": torch.minimum(pti.pow(2 * self.exponent), ptj.pow(2 * self.exponent)) * dr2 / (self.radius**2 + eps),
                "cos3d": spatial / (norm[:, :, None] * norm[:, None, :]).clamp_min(eps),
                "z_sym": torch.minimum(pti, ptj) / (pti + ptj + eps),
                "lund_log_invDR": -(dr + eps).log(),
                "lund_log_kt": (torch.minimum(pti, ptj) * dr + eps).log(),
            }
            selected = []
            for name in self.features:
                if name == "eec":
                    selected.extend(zprod * (dr + eps).pow(beta) for beta in self.betas)
                else:
                    selected.append(channels[name])
            features = torch.stack(selected, -1)
            valid = mask[:, :, None] & mask[:, None, :]
            valid = valid & ~torch.eye(p.shape[1], dtype=torch.bool, device=p.device)
            features = features.masked_fill(~valid[..., None], 0).float()
        bias = self.to_bias(self.network(features)).float().permute(0, 3, 1, 2) * self.scale
        if self.head_gain is not None:
            bias = bias * self.head_gain[None, :, None, None]
        valid = valid[:, None]
        mean = bias.masked_fill(~valid, 0).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
        bias = bias - mean
        if self.clamp:
            bias = bias.clamp(-self.clamp, self.clamp)
        return bias.masked_fill(~valid, 0)

class MandelstamBias(nn.Module):
    """Per-head bias from s=(p_i+p_j)^2 and t=(p_i-p_j)^2"""

    def __init__(self, heads):
        super().__init__()
        self.network = mlp(2, 32, heads)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, p, mask):
        """
        Map Mandelstam invariants to an additive attention bias

        Args:
            p:  Four-momenta (B,N,4), ordered E,px,py,pz
            mask:  Valid particle mask (B,N)
        Returns:
            bias: Finite scalar biases (B,H,N,N)
        """
        with torch.autocast(device_type=p.device.type, enabled=False):
            p = p.double().masked_fill(~mask[..., None], 0)
            ab = dot(p[:, :, None], p[:, None])
            m = dot(p, p)
            s = m[:, :, None] + m[:, None, :] + 2 * ab
            t = m[:, :, None] + m[:, None, :] - 2 * ab
            features = torch.stack([s.asinh(), t.asinh()], -1).float()
        bias = self.network(features).permute(0, 3, 1, 2)
        valid = mask[:, None, :, None] & mask[:, None, None, :]
        return bias.masked_fill(~valid, 0)
