# Muon, AdamW and Lion parameter updates
# Reference: https://arxiv.org/abs/2302.06675
#
# mikael.mieskolainen@cern.ch (2026)

import torch


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups, matrix_parameters, lr, betas, eps, fused=False):
        """
        Apply Muon to dense projections and AdamW to the remaining parameters

        Args:
            groups:  Trainable parameter groups with weight decay
            matrix_parameters:  Dense projection weights for Muon
            lr:  Shared base learning rate
            betas:  AdamW momentum coefficients
            eps:  AdamW denominator offset
            fused:  Use fused AdamW updates on CUDA
        """
        muon_groups, adamw_groups = [], []
        for group in groups:
            matrices = [p for p in group["params"] if p in matrix_parameters]
            others = [p for p in group["params"] if p not in matrix_parameters]
            if matrices:
                muon_groups.append({**group, "params": matrices})
            if others:
                adamw_groups.append({**group, "params": others})

        self.optimizers = []
        if muon_groups:
            self.optimizers.append(torch.optim.Muon(muon_groups, lr=lr, adjust_lr_fn="match_rms_adamw"))
        if adamw_groups:
            self.optimizers.append(torch.optim.AdamW(adamw_groups, lr=lr, betas=betas, eps=eps, fused=fused))
        super().__init__([group for optimizer in self.optimizers for group in optimizer.param_groups], {"lr": lr})
        for optimizer in self.optimizers:
            optimizer.state = self.state

    def load_state_dict(self, state_dict):
        """
        Restore both updates and their scheduled learning rates

        Args:
            state_dict:  Joint optimizer checkpoint
        """
        super().load_state_dict(state_dict)
        # PyTorch replaces group dictionaries and state on restore
        offset = 0
        for optimizer in self.optimizers:
            count = len(optimizer.param_groups)
            optimizer.param_groups = self.param_groups[offset:offset + count]
            optimizer.state = self.state
            offset += count

    def step(self, closure=None):
        """
        Update both parameter sets after joint gradient normalization and clipping

        Args:
            closure:  Optional loss function with gradient computation
        Returns:
            loss:  Closure loss when supplied
        """
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        for optimizer in self.optimizers:
            optimizer.step()
        return loss


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Construct Lion with decoupled weight decay

        Args:
            params:  Parameters or optimizer parameter groups
            lr:  Learning rate
            betas:  Update interpolation and momentum decay
            weight_decay:  Decoupled decay coefficient
        """
        if lr <= 0 or weight_decay < 0 or len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid Lion learning rate, decay or momentum")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Apply the sign update and retain the exponential gradient average

        Args:
            closure:  Optional loss function with gradient computation
        Returns:
            loss: Closure loss when supplied
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or parameter.is_complex():
                    raise ValueError("Lion requires dense real gradients")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                update = momentum.mul(beta1).add(gradient, alpha=1 - beta1).sign_()
                parameter.add_(update, alpha=-group["lr"])
                momentum.mul_(beta2).add_(gradient, alpha=1 - beta2)
        return loss
