# Jet constituent kinematics
#
# mikael.mieskolainen@cern.ch (2026)

import torch

KINEMATICS15 = (
    "E",
    "px",
    "py",
    "pz",
    "pt",
    "eta",
    "phi",
    "mass",
    "deta",
    "dphi",
    "log_pt",
    "log_E",
    "log_pt_rel",
    "log_E_rel",
    "dR",
)

def constituent_features(p, feature_set):
    """
    Construct constituent features from selected physical four-momenta

    Args:
        p:  Nonempty selected constituent momenta (N,4)
        feature_set:  p4 or kinematics15
    Returns:
        features: Four-vector channels or the archived 15 kinematic channels
    """
    if feature_set == "p4":
        return p.clone()
    if feature_set != "kinematics15":
        raise ValueError(f"Unknown top-tag feature set: {feature_set}")
    eps = 1e-8
    e, px, py, pz = p.unbind(-1)
    pt = (px.square() + py.square()).clamp_min(eps).sqrt()
    phi = (torch.atan2(py, px) + torch.pi) % (2 * torch.pi) - torch.pi
    eta = torch.asinh(pz / pt.clamp_min(eps))
    mass = (e.square() - (px.square() + py.square() + pz.square())).clamp_min(0).sqrt()
    total = p.sum(0)
    jet_pt = (total[1].square() + total[2].square()).clamp_min(eps).sqrt()
    jet_phi = (torch.atan2(total[2], total[1]) + torch.pi) % (2 * torch.pi) - torch.pi
    jet_eta = torch.asinh(total[3] / jet_pt.clamp_min(eps))
    deta = eta - jet_eta
    dphi = (phi - jet_phi + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack(
        [
            e,
            px,
            py,
            pz,
            pt,
            eta,
            phi,
            mass,
            deta,
            dphi,
            pt.clamp_min(eps).log(),
            e.clamp_min(eps).log(),
            (pt / jet_pt.clamp_min(eps)).clamp_min(eps).log(),
            (e / total[0].clamp_min(eps)).clamp_min(eps).log(),
            (deta.square() + dphi.square()).sqrt(),
        ],
        -1,
    )
