# Momentum transport diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch


def momentum_loss(prediction, target, cartesian=True):
    """
    Compare one or two three-momentum vectors

    Args:
        prediction:  Predicted Cartesian or cylindrical components
        target:  Reference components
        cartesian:  Select Cartesian squared error
    Returns:
        output: Mean component loss per event
    """
    if prediction.shape != target.shape or prediction.shape[-1] not in {3, 6}:
        raise ValueError("Momentum matching requires three or six components")
    if cartesian:
        return (prediction - target).square().mean(-1)
    residual = (prediction - target).reshape(len(target), -1, 3)
    radial = residual[..., 0].abs()
    log_cosh = radial + torch.nn.functional.softplus(-2 * radial) - math.log(2)
    angular = torch.atan2(residual[..., 2].sin(), residual[..., 2].cos())
    return (log_cosh + residual[..., 1].square() + angular.square()).mean(-1) / 3
