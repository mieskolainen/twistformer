# Flow normalization and density units
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import torch
from twistformer.data.preprocessing import Preprocessor
from .transforms import transform_chain


class FlowPreprocessor(Preprocessor):
    def __init__(self, state=None, *, flow_transforms=None, **options):
        """
        Restore the physical coordinate transform chain

        Args:
            state:  Optional saved preprocessing state
            flow_transforms:  Explicit physical coordinate transforms
            options:  Shared affine normalization settings
        """
        super().__init__(state, **options)
        self.transforms = list(flow_transforms or [])
        if state is not None and state.get("flow_transforms", []) != self.transforms:
            raise ValueError("Flow coordinate transforms differ from checkpoint preprocessing")

    def fit(self, dataset, regression=True, class_count=0):
        """
        Fit source-compatible target coordinates on training events

        Args:
            dataset:  Training Sample records
            regression:  Shared trainer argument
            class_count:  Must be zero for density training
        Returns:
            preprocessor: Fitted flow preprocessor
        """
        if class_count:
            raise ValueError("Flow normalization cannot balance class labels")
        shape = None
        def checked():
            nonlocal shape
            for sample in dataset:
                if sample.target is None or not sample.target.is_floating_point() or not torch.isfinite(sample.target).all():
                    raise ValueError("Flows require complete finite continuous targets")
                if sample.target_mask is not None and not sample.target_mask.all():
                    raise ValueError("Missing target coordinates require an explicit marginal density model")
                if shape is None:
                    shape = tuple(sample.target.shape)
                if tuple(sample.target.shape) != shape:
                    raise ValueError("Flow target shapes must agree")
                if sample.source is not None and sample.source.shape != sample.target.shape:
                    raise ValueError("Morphing source and target shapes must agree")
                target, _ = transform_chain(sample.target.unsqueeze(0), self.transforms)
                yield replace(sample, target=target[0])
        super().fit(checked(), regression=True)
        self.state["flow_shape"] = torch.tensor(shape, dtype=torch.long)
        self.state["flow_transforms"] = self.transforms
        if getattr(self, "quantization_levels", 0):
            self.state["dequantization_levels"] = self.quantization_levels
        return self

    def transform(self, sample):
        """
        Normalize flow tensors while retaining their physical shape

        Args:
            sample:  Raw event
        Returns:
            output: Normalized event
        """
        shape = tuple(self.state["flow_shape"].tolist())
        if sample.target is not None and tuple(sample.target.shape) != shape:
            raise ValueError("Flow target shape differs from checkpoint preprocessing")
        determinant = None
        flat = None
        if sample.target is not None:
            target, determinant = transform_chain(sample.target.unsqueeze(0), self.transforms)
            flat, determinant = target.flatten(), determinant[0]
        mask = sample.target_mask
        if mask is not None and (sample.target is None or mask.shape != sample.target.shape or not mask.all()):
            raise ValueError("Flows require a complete target mask matching the target shape")
        output = super().transform(replace(sample, target=flat, target_mask=None))
        source = sample.source
        if source is not None:
            if tuple(source.shape) != shape:
                raise ValueError("Flow source shape differs from checkpoint preprocessing")
            source = transform_chain(source.unsqueeze(0), self.transforms)[0][0]
            source = ((source.flatten() - self.state["target_mean"]) / self.state["target_std"]).reshape(shape)
        return replace(output, target=None if output.target is None else output.target.reshape(shape), target_mask=mask, source=source,
                       flow_log_det=determinant)

    def inverse_target(self, target):
        """
        Restore physical coordinates with any leading sample axes

        Args:
            target:  Standardized coordinates
        Returns:
            output: Physical coordinates
        """
        shape = tuple(self.state["flow_shape"].tolist())
        scale = self.state["target_std"].to(target).reshape(shape)
        shift = self.state["target_mean"].to(target).reshape(shape)
        value = target * scale + shift
        return transform_chain(value.reshape(-1, *shape), self.transforms, inverse=True)[0].reshape_as(value)

    def inverse_prediction(self, prediction):
        """
        Restore generated coordinates and the physical density measure

        Args:
            prediction:  Flow Prediction record
        Returns:
            output: Prediction in physical units
        """
        shape = tuple(self.state["flow_shape"].tolist())
        scale = self.state["target_std"].reshape(shape)
        samples = None if prediction.samples is None else self.inverse_target(prediction.samples)
        return replace(
            prediction,
            mean=samples.mean(1) if samples is not None else None if prediction.mean is None else self.inverse_target(prediction.mean),
            std=samples.std(1, unbiased=False) if samples is not None else None if prediction.std is None else prediction.std * scale.to(prediction.std),
            samples=samples,
            log_prob=None if prediction.log_prob is None else prediction.log_prob - scale.to(prediction.log_prob).log().sum(),
        )
