# Conditional flow tasks with event encoders
#
# mikael.mieskolainen@cern.ch (2026)

import math
from dataclasses import replace
import torch
from torch import nn
from twistformer.api.contracts import Prediction, TaskSpec, LossResult
from .continuous import ContinuousFlow
from .discrete import DiscreteFlow
from .networks import VectorField, Codec


class GraphConditioner(nn.Module):
    def __init__(self, input_dim, width, depth):
        """
        Construct the Supraflow graph attention conditioner

        Args:
            input_dim:  Node feature count
            width:  Hidden width
            depth:  Message layer count
        """
        super().__init__()
        try:
            from torch_geometric.nn import TransformerConv
        except ImportError as error:
            raise ImportError("Graph conditioning requires twistformer[flow-graph]") from error
        self.projection = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(TransformerConv(width, width, heads=4, concat=False) for _ in range(depth))

    def forward(self, batch):
        """
        Encode disjoint complete event graphs with masked pooling

        Args:
            batch:  Padded node features and valid-node masks
        Returns:
            output: Concatenated mean and maximum graph features
        """
        count, tokens = batch.mask.shape
        valid = batch.mask.flatten()
        event = torch.arange(count, device=valid.device).repeat_interleave(tokens)[valid]
        full_indices = torch.arange(count * tokens, device=valid.device).reshape(count, tokens)
        compact = torch.full((count * tokens,), -1, dtype=torch.long, device=valid.device)
        compact[valid] = torch.arange(int(valid.sum()), device=valid.device)
        edges = batch.mask[:, :, None] & batch.mask[:, None, :]
        row = full_indices[:, :, None].expand(-1, -1, tokens)[edges]
        column = full_indices[:, None, :].expand(-1, tokens, -1)[edges]
        edge_index = torch.stack((compact[row], compact[column]))
        value = self.projection(batch.features.flatten(0, 1)[valid])
        for layer in self.layers:
            value = value + torch.nn.functional.silu(layer(value, edge_index))
        from torch_geometric.nn import global_mean_pool, global_max_pool

        return torch.cat((global_mean_pool(value, event, size=count), global_max_pool(value, event, size=count)), -1)


class FlowModel(nn.Module):
    def __init__(self, input_spec, task_spec, config):
        """
        Combine a flow, optional event conditioner and latent codec

        Args:
            input_spec:  Driver input schema
            task_spec:  Driver target schema
            config:  Experiment configuration
        """
        super().__init__()
        self.config, self.task_name = config.flow, task_spec.name
        f = self.config
        self.shape = tuple(f.shape or [len(task_spec.outputs)])
        if math.prod(self.shape) != len(task_spec.outputs):
            raise ValueError("Flow shape must contain one coordinate per task output")
        self.requires_p4 = False
        self.conditioner = None
        conditional = f.mode in {"conditional", "conditional-flip", "morph-conditional", "morph-conditional-shuffle"}
        condition_dim = 0
        if conditional:
            if f.condition == "encoder":
                from twistformer.models.factory import Model

                if config.model.gatr_readout == "logits":
                    raise ValueError("Flow conditioning requires an invariant encoder readout")
                self.conditioner = Model(input_spec, TaskSpec("gaussian_regression", ("embedding",)),
                                         config.model, config.task)
                self.conditioner.head = nn.Identity()
                self.requires_p4 = self.conditioner.requires_p4
                condition_dim = config.model.width
            elif f.condition == "graph":
                self.conditioner = GraphConditioner(len(input_spec.features), f.width, f.depth)
                condition_dim = 2 * f.width
            elif f.condition == "parameters":
                condition_dim = len(input_spec.parameters)
                if not condition_dim:
                    raise ValueError("Conditional flows require declared parameters or an event encoder")
        self.conditional = conditional
        self.codec = Codec(self.shape, f) if f.space == "latent" else None
        flow_shape = (f.latent_dim,) if self.codec is not None else self.shape
        if f.type == "discrete":
            self.flow = DiscreteFlow(flow_shape, condition_dim, f)
        else:
            if f.network == "unet":
                from .unet.unet import UNetModelWrapper

                if len(flow_shape) != 3 or flow_shape[-1] != flow_shape[-2]:
                    raise ValueError("U-Net requires square images with shape [channels, height, width]")
                if conditional and f.condition != "image":
                    raise ValueError("U-Net conditional generation requires flow.condition=image")
                field = UNetModelWrapper(dim=flow_shape, num_channels=f.width, num_res_blocks=f.depth,
                                         superreso=conditional, channel_mult=(1, 2), attention_resolutions="16")
            else:
                field = VectorField(flow_shape, condition_dim, f)
            self.flow = ContinuousFlow(field, flow_shape, steps=f.steps, integrator=f.integrator, base=f.base,
                                       path=f.path, sigma=f.sigma, warp=f.warp, trace=f.trace, trace_samples=f.trace_samples,
                                       finite_difference=f.finite_difference, ot_method=f.ot_method,
                                       ot_regularization=f.ot_regularization, ot_mass_regularization=f.ot_mass_regularization)

    @property
    def has_density(self):
        """
        Report whether the observation density has a known base measure

        Returns:
            result: True for direct generative flows
        """
        return self.codec is None and "morph" not in self.config.mode

    def condition(self, batch):
        """
        Encode external measurements without using target coordinates

        Args:
            batch:  Preprocessed event batch
        Returns:
            output: Optional condition tensor
        """
        if not self.conditional:
            return None
        if self.config.condition == "encoder":
            return self.conditioner.encode(batch)
        if self.config.condition == "graph":
            return self.conditioner(batch)
        if self.config.condition == "image":
            if batch.flow_condition is None:
                raise ValueError("Image conditioning requires a flow_condition tensor")
            return batch.flow_condition
        return batch.parameters

    def forward(self, batch):
        """
        Compute per-event flow objectives through the DDP forward path

        Args:
            batch:  Flow targets, sources, conditions and weights
        Returns:
            prediction: Per-event loss and optional log density
        """
        if batch.targets is None:
            raise ValueError("Flow training requires target coordinates")
        target, source, weights = batch.targets.float(), batch.source, batch.sample_weights
        if not torch.isfinite(target).all() or (batch.target_mask is not None and not batch.target_mask.all()):
            raise ValueError("Flow targets must be complete finite observations")
        if tuple(target.shape[1:]) != self.shape:
            raise ValueError("Flow targets do not match the configured shape")
        if ("morph" in self.config.mode) != (source is not None):
            raise ValueError("Morphing requires source coordinates and generation requires a known base")
        condition = self.condition(batch)
        reconstruction = target.new_zeros(len(target))
        if self.codec is not None:
            latent, kl = self.codec.encode(target, stochastic=self.training)
            reconstructed = self.codec.decode(latent)
            reconstruction = self.config.reconstruction_weight * (reconstructed - target).square().flatten(1).mean(-1)
            reconstruction = reconstruction + self.config.kl_weight * kl
            target = latent
            source = None if source is None else self.codec.encode(source, stochastic=self.training)[0]
        if "shuffle" in self.config.mode:
            permutation = torch.randperm(len(target), device=target.device)
            if condition is not None:
                if condition.ndim != 2:
                    raise ValueError("Conditional shuffle requires explicit vector conditions")
                _, groups = torch.unique(condition.detach(), dim=0, return_inverse=True)
                permutation = torch.arange(len(target), device=target.device)
                for group in groups.unique():
                    indices = (groups == group).nonzero().flatten()
                    permutation[indices] = indices[torch.randperm(len(indices), device=target.device)]
            target, weights, reconstruction = target[permutation], weights[permutation], reconstruction[permutation]
        if self.task_name == "flow_nll":
            if self.training and self.config.noise_regularization:
                variance = self.config.noise_regularization
                target = math.sqrt(1 - variance) * target + math.sqrt(variance) * torch.randn_like(target)
            log_prob = self.flow.log_prob(target, condition, create_graph=self.training)
            return Prediction(flow_loss=-log_prob, flow_weights=weights, log_prob=log_prob)
        loss_function = None
        if self.config.matching_loss == "momentum":
            from .physics import momentum_loss

            loss_function = momentum_loss
        loss, aligned_weights = self.flow.matching_loss(target, source, condition, weights, loss=loss_function,
                                                       auxiliary=reconstruction)
        if self.config.lipschitz_weight:
            loss = loss + self.config.lipschitz_weight * self.flow.vector_field.get_lipschitz_loss()
        return Prediction(flow_loss=loss, flow_weights=aligned_weights)

    def sample(self, batch, *, count=1, temperature=None, generator=None):
        """
        Generate conditional replicas for each event

        Args:
            batch:  Conditions and optional morphing sources
            count:  Replica count per event
            temperature:  Optional base temperature override
            generator:  Optional Torch random generator
        Returns:
            output: Events with axes batch, replica and coordinate
        """
        condition = self.condition(batch)
        size = len(batch.ids)
        condition = None if condition is None else condition.repeat_interleave(count, 0)
        if "morph" in self.config.mode:
            if batch.source is None:
                raise ValueError("Transport prediction requires source coordinates")
            source = batch.source.repeat_interleave(count, 0)
            if self.codec is not None:
                source = self.codec.encode(source)[0]
            output = self.flow.transport(source, condition)
        else:
            output = self.flow.sample(size * count, condition,
                                      temperature=self.config.temperature if temperature is None else temperature,
                                      generator=generator)
        if self.codec is not None:
            output = self.codec.decode(output)
        return output.reshape(size, count, *self.shape)

    def log_prob(self, batch):
        """
        Evaluate a direct observation-space density

        Args:
            batch:  Target coordinates and conditions
        Returns:
            output: Log density per event
        """
        if not self.has_density:
            raise ValueError("Latent and empirical-source morphing models have no exact observation density")
        if batch.targets is None:
            raise ValueError("Density evaluation requires target coordinates")
        return self.flow.log_prob(batch.targets, self.condition(batch))


class FlowTask:
    def __init__(self, spec, config):
        """
        Configure a common weighted flow objective

        Args:
            spec:  Flow output schema
            config:  Task configuration
        """
        self.spec, self.config = spec, config

    def compute_loss(self, prediction, batch):
        """
        Reduce event losses with aligned nonnegative probability weights

        Args:
            prediction:  Per-event flow losses
            batch:  Event weights
        Returns:
            result: Differentiable loss numerator and weight normalizer
        """
        weights = prediction.flow_weights if prediction.flow_weights is not None else batch.sample_weights
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Flow objectives require finite nonnegative probability weights")
        if prediction.flow_loss is None:
            raise ValueError("Prediction does not contain a flow objective")
        losses = torch.where(weights > 0, prediction.flow_loss, 0.0)
        return LossResult((losses * weights).sum(), weights.sum())
