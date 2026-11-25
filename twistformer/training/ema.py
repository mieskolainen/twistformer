# Exponential averaging of model parameters
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
import torch

class ModelEMA:
    """Average parameters after successful optimizer steps and copy model buffers"""

    def __init__(self, model, decay):
        self.module = deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        """
        Average parameters and preserve current normalization statistics

        Args:
            model:  Model after a successful optimizer update
        Returns:
            None: Shadow weights and update counter are advanced
        """
        for shadow, source in zip(self.module.parameters(), model.parameters(), strict=True):
            shadow.lerp_(source.detach(), 1 - self.decay)
        for shadow, source in zip(self.module.buffers(), model.buffers(), strict=True):
            shadow.copy_(source)
        self.updates += 1

    def state_dict(self):
        """
        Serialize averaged weights and update count
        Returns:
            state: Portable EMA checkpoint state
        """
        return {"model": self.module.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state):
        """
        Restore the complete averaging state

        Args:
            state:  EMA checkpoint state
        Returns:
            None: Shadow model and counter are restored
        """
        if state["decay"] != self.decay:
            raise ValueError("EMA decay differs from the checkpoint")
        self.module.load_state_dict(state["model"], strict=True)
        self.updates = state["updates"]
