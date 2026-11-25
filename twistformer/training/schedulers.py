# Epoch and optimizer-step learning-rate schedules
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch

def step_scheduler(scheduler, metric):
    """
    Advance an epoch schedule using the configured validation metric

    Args:
        scheduler:  Optional epoch scheduler
        metric:  Validation monitor value
    Returns:
        None: Optimizer learning rates are updated
    """
    if isinstance(scheduler, (WarmupScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step(metric)
    elif scheduler is not None:
        scheduler.step()

class WarmupScheduler:
    """Linear epoch warmup followed by a standard Torch scheduler"""

    def __init__(self, optimizer, scheduler, epochs):
        self.optimizer, self.scheduler, self.epochs = optimizer, scheduler, epochs
        self.completed = 0
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self._set_fraction(1 / epochs)

    def _set_fraction(self, fraction):
        """
        Set each parameter group's fraction of its base rate

        Args:
            fraction:  Warmup multiplier
        Returns:
            None: Optimizer rates are replaced
        """
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * fraction

    def step(self, metric):
        """
        Finish an epoch and prepare the rate for the next epoch

        Args:
            metric:  Validation monitor value for plateau scheduling
        Returns:
            None: Warmup or the downstream schedule advances
        """
        self.completed += 1
        if self.completed < self.epochs:
            self._set_fraction((self.completed + 1) / self.epochs)
        elif self.completed == self.epochs:
            self._set_fraction(1.0)
        else:
            step_scheduler(self.scheduler, metric)

    def state_dict(self):
        """
        Serialize warmup progress and the downstream schedule
        Returns:
            state: Epoch-boundary scheduler state
        """
        return {
            "completed": self.completed,
            "epochs": self.epochs,
            "base_lrs": self.base_lrs,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state):
        """
        Restore schedule progress after restoring optimizer rates

        Args:
            state:  Saved warmup and scheduler state
        Returns:
            None: Internal counters are restored without advancing the rate
        """
        if state["epochs"] != self.epochs:
            raise ValueError("Warmup duration differs from the checkpoint")
        self.completed, self.base_lrs = state["completed"], state["base_lrs"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])

class CosineRestartScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine restart cycles followed by exponential decay"""

    def __init__(self, optimizer, cfg):
        """
        Set the post-warmup cosine and decay intervals

        Args:
            optimizer:  Training optimizer
            cfg:  Training configuration
        Returns:
            None: The first cosine epoch starts at the base rate
        """
        self.cosine_epochs = cfg.epochs - cfg.warmup_epochs - cfg.decay_epochs
        self.restart_period = cfg.restart_period
        self.restart_mult = cfg.restart_mult
        self.decay_gamma = cfg.decay_gamma
        self.min_lr = cfg.min_learning_rate
        super().__init__(optimizer)

    def get_lr(self):
        """
        Evaluate cosine cycles and decay from the last cosine rate

        Returns:
            rates: Learning rates for the next training epoch
        """
        epoch = min(self.last_epoch, self.cosine_epochs - 1)
        period = self.restart_period
        while epoch >= period:
            epoch -= period
            period *= self.restart_mult
        fraction = (1 + math.cos(math.pi * epoch / period)) / 2
        decay = self.decay_gamma ** max(0, self.last_epoch - self.cosine_epochs + 1)
        return [(self.min_lr + (base - self.min_lr) * fraction) * decay for base in self.base_lrs]

def make_scheduler(optimizer, cfg):
    """
    Construct epoch or optimizer-step learning rates

    Args:
        optimizer:  Training optimizer
        cfg:  Training configuration
    Returns:
        scheduler: Optional schedule advanced at the configured interval
    """
    span = cfg.max_steps if cfg.scheduler_interval == "step" else cfg.epochs - cfg.warmup_epochs
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=span, eta_min=cfg.min_learning_rate)
    elif cfg.scheduler == "cosine_restarts":
        scheduler = CosineRestartScheduler(optimizer, cfg)
    elif cfg.scheduler == "thirds":

        def fraction(epoch):
            """
            Apply two decade reductions over the post-warmup epoch budget

            Args:
                epoch:  Completed post-warmup epochs
            Returns:
                factor: Learning-rate multiplier bounded by the configured floor
            """
            reductions = int(epoch >= max(1, span // 3)) + int(epoch >= max(2, 2 * (span // 3)))
            return max(0.1**reductions, cfg.min_learning_rate / cfg.learning_rate)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, fraction)
    elif cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            threshold=cfg.plateau_threshold,
            cooldown=cfg.plateau_cooldown,
            min_lr=cfg.min_learning_rate,
        )
    else:
        scheduler = None
    if cfg.warmup_epochs:
        return WarmupScheduler(optimizer, scheduler, cfg.warmup_epochs)
    return scheduler
