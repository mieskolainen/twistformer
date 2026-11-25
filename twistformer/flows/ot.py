# Minibatch optimal transport
#
# mikael.mieskolainen@cern.ch (2026)

import torch


class OTPlanSampler:
    def __init__(self, method="exact", reg=0.05, reg_m=1.0, normalize_cost=False, **options):
        """
        Configure an optional POT solver

        Args:
            method:  Exact, Sinkhorn, unbalanced or partial transport
            reg:  Entropy regularization
            reg_m:  Marginal regularization
            normalize_cost:  Scale the cost by its maximum
            options:  Additional solver options
        """
        if method not in {"exact", "sinkhorn", "unbalanced", "partial"}:
            raise ValueError("Unknown OT solver")
        self.method, self.reg, self.reg_m = method, reg, reg_m
        self.normalize_cost, self.options = normalize_cost, options

    def get_map(self, x0, x1, weights0=None, weights1=None):
        """
        Solve transport with explicit nonnegative marginal weights

        Args:
            x0:  Source coordinates
            x1:  Target coordinates
            weights0:  Optional source masses
            weights1:  Optional target masses
        Returns:
            result: Normalized transport matrix on the input device
        """
        try:
            import ot
        except ImportError as error:
            raise ImportError("OT paths require the twistformer[flow] extra with POT") from error
        cost = torch.cdist(x0.detach().double().flatten(1), x1.detach().double().flatten(1)).square()
        if self.normalize_cost:
            cost = cost / cost.max().clamp_min(torch.finfo(cost.dtype).eps)
        marginals = []
        for size, weights in [(len(x0), weights0), (len(x1), weights1)]:
            mass = torch.ones(size, device=cost.device, dtype=cost.dtype) if weights is None else weights.to(cost)
            if mass.shape != (size,) or not torch.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
                raise ValueError("OT masses must be finite, nonnegative and have positive sum")
            marginals.append((mass / mass.sum()).cpu().numpy())
        a, b = marginals
        matrix = cost.cpu().numpy()
        if self.method == "exact":
            result = ot.emd(a, b, matrix, numThreads=self.options.get("num_threads", 1))
        elif self.method == "sinkhorn":
            result = ot.sinkhorn(a, b, matrix, self.reg)
        elif self.method == "unbalanced":
            result = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, matrix, self.reg, self.reg_m)
        else:
            result = ot.partial.entropic_partial_wasserstein(a, b, matrix, self.reg)
        result = torch.as_tensor(result, device=x0.device, dtype=torch.float64)
        if not torch.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
            raise FloatingPointError("OT solver returned an invalid transport plan")
        return result / result.sum()

    def indices(self, x0, x1, *, generator=None, replace=True):
        """
        Draw aligned source and target indices from a transport plan

        Args:
            x0:  Source coordinates
            x1:  Target coordinates
            generator:  Optional Torch random generator
            replace:  Sample joint cells with replacement
        Returns:
            result: Source and target event indices
        """
        plan = self.get_map(x0, x1)
        flat = self.sample_map(plan, len(x1), generator=generator, replace=replace)
        return flat

    def sample_map(self, plan, count, *, generator=None, replace=True):
        """
        Sample a finite joint transport matrix

        Args:
            plan:  Nonnegative joint masses
            count:  Number of event pairs
            generator:  Optional Torch random generator
            replace:  Allow repeated joint cells
        Returns:
            result: Source and target index tensors
        """
        plan = torch.as_tensor(plan)
        if plan.ndim != 2 or not torch.isfinite(plan).all() or (plan < 0).any() or plan.sum() <= 0:
            raise ValueError("Invalid joint transport matrix")
        if not replace and count > int((plan > 0).sum()):
            raise ValueError("Too few positive OT cells for sampling without replacement")
        flat = torch.multinomial(plan.flatten(), count, replacement=replace, generator=generator)
        return flat // plan.shape[1], flat % plan.shape[1]

    def sample_plan(self, x0, x1, *, generator=None, replace=True):
        """
        Draw source and target event pairs

        Args:
            x0:  Source coordinates
            x1:  Target coordinates
            generator:  Optional Torch random generator
            replace:  Allow repeated joint cells
        Returns:
            result: Paired coordinates
        """
        source, target = self.indices(x0, x1, generator=generator, replace=replace)
        return x0[source], x1[target]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, *, generator=None, replace=True):
        """
        Preserve labels when sampling transport pairs

        Args:
            x0:  Source coordinates
            x1:  Target coordinates
            y0:  Optional source labels
            y1:  Optional target labels
            generator:  Optional Torch random generator
            replace:  Allow repeated joint cells
        Returns:
            result: Coordinates and aligned labels
        """
        source, target = self.indices(x0, x1, generator=generator, replace=replace)
        return x0[source], x1[target], None if y0 is None else y0[source], None if y1 is None else y1[target]

    def sample_trajectory(self, populations, *, generator=None):
        """
        Sample Markov trajectories through successive OT plans

        Args:
            populations:  Event, time and coordinate tensor
            generator:  Optional Torch random generator
        Returns:
            output: Aligned populations with the same shape
        """
        indices = torch.arange(len(populations), device=populations.device)
        output = [populations[:, 0]]
        for index in range(1, populations.shape[1]):
            plan = self.get_map(populations[:, index - 1], populations[:, index])[indices]
            if (plan.sum(-1) <= 0).any():
                raise ValueError("Trajectory has a source with zero transported mass")
            indices = torch.multinomial(plan, 1, generator=generator).squeeze(-1)
            output.append(populations[indices, index])
        return torch.stack(output, 1)


def wasserstein(x0, x1, *, method="exact", reg=0.05, power=2, weights0=None, weights1=None):
    """
    Compute an empirical Wasserstein distance

    Args:
        x0:  Source events
        x1:  Target events
        method:  Exact or entropy-regularized transport
        reg:  Positive entropy regularization
        power:  Distance order one or two
        weights0:  Optional source probability masses
        weights1:  Optional target probability masses
    Returns:
        result: Nonnegative scalar distance
    """
    import ot

    if method not in {"exact", "sinkhorn"} or power not in {1, 2} or reg <= 0:
        raise ValueError("Invalid Wasserstein distance settings")
    cost = torch.cdist(x0.detach().double().flatten(1), x1.detach().double().flatten(1)).pow(power).cpu().numpy()
    masses = []
    for size, weights in ((len(x0), weights0), (len(x1), weights1)):
        value = torch.ones(size, dtype=torch.float64) if weights is None else torch.as_tensor(weights).detach().double().cpu()
        if value.shape != (size,) or not torch.isfinite(value).all() or (value < 0).any() or value.sum() <= 0:
            raise ValueError("Wasserstein distances need finite nonnegative masses")
        masses.append((value / value.sum()).numpy())
    result = ot.emd2(*masses, cost) if method == "exact" else ot.sinkhorn2(*masses, cost, reg)
    return max(float(result), 0.0) ** (1 / power)
