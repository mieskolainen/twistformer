# Particle batches and training normalization
#
# mikael.mieskolainen@cern.ch (2026)

"""Train-fitted affine transforms. Geometry is never componentwise standardized"""

from dataclasses import replace
import torch
from torch.utils.data import Dataset
from .loading import fetch_samples

class Preprocessor:
    def __init__(
        self, state=None, *, normalize_features=True, normalize_targets=True, normalize_globals=True, global_dim=None,
        normalize_parameters=True, parameter_dim=None,
    ):
        self.state = state
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.normalize_globals = normalize_globals
        self.global_dim = global_dim
        self.normalize_parameters = normalize_parameters
        self.parameter_dim = parameter_dim
        self.balance_observable = ""
        self.balance_edges = None

    def configure_class_bins(self, observable, edges):
        """
        Balance one physical observable within each class using training yields

        Args:
            observable:  Sample observable name, excluded from model inputs
            edges:  Increasing boundaries for half-open intervals
        Returns:
            preprocessor: This configured instance
        """
        edges = torch.as_tensor(edges, dtype=torch.float64)
        if not observable or edges.ndim != 1 or len(edges) < 2 or not torch.isfinite(edges).all() or (edges.diff() <= 0).any():
            raise ValueError("Class-bin balancing requires an observable and finite increasing edges")
        if self.state is not None and (
            "class_bin_weights" not in self.state or "class_bin_edges" not in self.state
            or not torch.equal(self.state["class_bin_edges"], edges)
        ):
            raise ValueError("Class-bin boundaries must match checkpoint preprocessing")
        self.balance_observable, self.balance_edges = observable, edges
        return self

    def _class_bins(self, samples, edges):
        """
        Assign physical observable values to configured intervals

        Args:
            samples:  Labeled raw events
            edges:  Bin boundaries in physical units
        Returns:
            bins: Integer bin index for each event
        """
        values = torch.tensor([sample.observables.get(self.balance_observable, float("nan")) for sample in samples], dtype=torch.float64)
        return self._balance_bins(values, edges)

    def _balance_bins(self, values, edges):
        """Assign the physical balancing values to half-open bins"""
        bins = torch.searchsorted(edges, values, right=True) - 1
        if not torch.isfinite(values).all() or ((bins < 0) | (bins >= len(edges) - 1)).any():
            raise ValueError(f"Class-bin balancing requires finite {self.balance_observable} within the configured edges")
        return bins

    def _descriptor_statistics(self, dataset, counts, totals, bin_totals):
        """
        Reduce physical shower rows before adding reflected training views

        Args:
            dataset:     Single-row classification events
            counts:      Per-class event counts
            totals:      Per-class physical weight sums
            bin_totals:  Optional per-class and energy-bin weight sums
        Returns:
            moments:  Feature and global sums and squared sums
        """
        def accumulate(values, normalize, total, square):
            if values.ndim != 2 or not values.numel() or not torch.isfinite(torch.stack(torch.aminmax(values))).all():
                raise ValueError("Descriptor inputs must be finite nonempty rows")
            if total is None:
                total = torch.zeros(values.size(1), dtype=torch.float64)
                square = torch.zeros_like(total)
            if values.size(1) != total.numel():
                raise ValueError("Descriptor widths must agree")
            if normalize:
                # Retain event-order sums for nearly constant shower observables
                values = values.to(dtype=torch.float64, copy=True)
                squares = values.square()
                values[0] += total
                squares[0] += square
                total = values.cumsum(0)[-1].clone()
                square = squares.cumsum(0)[-1].clone()
            return total, square

        total = square = global_total = global_square = None
        for start in range(0, len(dataset), 2048):
            batch = dataset.classification_batch(slice(start, start + 2048), self.balance_observable)
            total, square = accumulate(batch.features, self.normalize_features, total, square)
            globals_ = batch.global_features
            if self.global_dim is not None and self.global_dim != (0 if globals_ is None else globals_.size(1)):
                raise ValueError("Global features must match the input schema")
            if globals_ is not None:
                global_total, global_square = accumulate(globals_, self.normalize_globals, global_total, global_square)
            if counts.numel():
                labels, weights = batch.labels, batch.weights.double()
                if ((labels < 0) | (labels >= counts.numel())).any():
                    raise ValueError("Invalid class balancing label")
                if not torch.isfinite(weights.float()).all() or (weights < 0).any():
                    raise ValueError("Class balancing requires finite nonnegative weights")
                counts += torch.bincount(labels, minlength=counts.numel())
                totals.scatter_add_(0, labels, weights)
                if bin_totals is not None:
                    bins = self._balance_bins(batch.balance_values.double(), self.balance_edges)
                    bin_totals.view(-1).scatter_add_(0, labels * bin_totals.size(1) + bins, weights)
        return total, square, global_total, global_square

    def fit(self, dataset, regression=False, class_count=0):
        """
        Fit feature and target normalization on the training split

        Args:
            dataset:  Training Sample records
        regression:  Fit target statistics using observed regression labels
        class_count:  Fit class balancing for this many classes, or zero to disable
        Returns:
            preprocessor: This fitted instance
        """
        count = 0
        global_count = 0
        global_total = global_square = None
        global_dim = self.global_dim
        parameter_dim = self.parameter_dim
        parameter_count = 0
        parameter_total = parameter_square = None
        total = square = None
        yt = ys = yc = None
        counts = torch.zeros(class_count, dtype=torch.long)
        totals = torch.zeros(class_count, dtype=torch.float64)
        bin_totals = None
        if self.balance_edges is not None:
            if not class_count or regression:
                raise ValueError("Class-bin balancing requires classification with class balancing")
            bin_totals = torch.zeros(class_count, len(self.balance_edges) - 1, dtype=torch.float64)
        samples = dataset
        if hasattr(dataset, "classification_batch") and not regression and not parameter_dim:
            total, square, global_total, global_square = self._descriptor_statistics(dataset, counts, totals, bin_totals)
            count = len(dataset)
            global_count = count if global_total is not None else 0
            samples = ()
        for sample in samples:
            p = sample.parameters
            actual_dim = 0 if p is None else p.numel()
            if parameter_dim is None:
                parameter_dim = actual_dim
            if actual_dim != parameter_dim or (p is not None and (p.ndim != 1 or actual_dim == 0 or not torch.isfinite(p).all())):
                raise ValueError("Parameters must be finite vectors matching the input schema")
            if p is not None:
                p = p.double()
                parameter_total = p if parameter_total is None else parameter_total + p
                parameter_square = p.square() if parameter_square is None else parameter_square + p.square()
                parameter_count += 1
            g = sample.global_features
            actual_dim = 0 if g is None else g.numel()
            if global_dim is None:
                global_dim = actual_dim
            if actual_dim != global_dim or (g is not None and (g.ndim != 1 or actual_dim == 0 or not torch.isfinite(g).all())):
                raise ValueError("Global features must be finite vectors matching the input schema")
            if g is not None:
                g = g.double()
                global_total = g if global_total is None else global_total + g
                global_square = g.square() if global_square is None else global_square + g.square()
                global_count += 1
            x = sample.features.double()
            if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
                raise ValueError("Training features must be finite, nonempty token arrays")
            if total is not None and x.shape[1] != total.numel():
                raise ValueError("Training feature widths must agree")
            total = x.sum(0) if total is None else total + x.sum(0)
            square = x.square().sum(0) if square is None else square + x.square().sum(0)
            count += len(x)
            if class_count:
                if sample.target is None or sample.target.numel() != 1:
                    raise ValueError("Class balancing requires scalar class labels")
                label = int(sample.target)
                if not 0 <= label < class_count or label != float(sample.target):
                    raise ValueError("Invalid class balancing label")
                if not torch.isfinite(torch.tensor(sample.weight)) or sample.weight < 0:
                    raise ValueError("Class balancing requires finite nonnegative weights")
                counts[label] += 1
                totals[label] += sample.weight
                if bin_totals is not None:
                    index = int(self._class_bins([sample], self.balance_edges)[0])
                    bin_totals[label, index] += sample.weight
            if regression:
                if sample.target is None:
                    raise ValueError("Regression training requires targets")
                y = sample.target.double().reshape(-1)
                if yt is not None and yt.shape != y.shape:
                    raise ValueError("Training target widths must agree")
                valid = torch.isfinite(y)
                if sample.target_mask is not None:
                    if sample.target_mask.shape != sample.target.shape:
                        raise ValueError("target_mask must match regression targets")
                    valid &= sample.target_mask.bool().reshape(-1)
                clean = y.masked_fill(~valid, 0)
                yt = clean if yt is None else yt + clean
                ys = clean.square() if ys is None else ys + clean.square()
                yc = valid.double() if yc is None else yc + valid
        if not count:
            raise ValueError("Training data contain no tokens")
        mean = total / count
        std = (square / count - mean.square()).clamp_min(0).sqrt()
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        self.state = {
            "feature_mean": (mean.float() if self.normalize_features else torch.zeros_like(mean).float()),
            "feature_std": (std.float() if self.normalize_features else torch.ones_like(std).float()),
        }
        if global_count:
            mean = global_total / global_count
            std = (global_square / global_count - mean.square()).clamp_min(0).sqrt()
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            self.state.update(
                global_mean=(mean.float() if self.normalize_globals else torch.zeros_like(mean).float()),
                global_std=(std.float() if self.normalize_globals else torch.ones_like(std).float()),
            )
        if class_count:
            if (totals <= 0).any():
                raise ValueError("Every balanced class needs positive training weight")
            self.state.update(
                class_counts=counts, class_weight_totals=totals, class_weights=(totals.sum() / (class_count * totals)).float()
            )
            if bin_totals is not None:
                if (bin_totals <= 0).any():
                    raise ValueError("Every class and observable bin needs positive training weight")
                self.state.update(
                    class_bin_edges=self.balance_edges.clone(), class_bin_weight_totals=bin_totals,
                    class_bin_weights=(totals[:, None] / (bin_totals.shape[1] * bin_totals)).float(),
                )
        if parameter_count:
            mean = parameter_total / parameter_count
            std = (parameter_square / parameter_count - mean.square()).clamp_min(0).sqrt()
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            self.state.update(
                parameter_mean=(mean.float() if self.normalize_parameters else torch.zeros_like(mean).float()),
                parameter_std=(std.float() if self.normalize_parameters else torch.ones_like(std).float()),
            )
        if regression:
            if (yc == 0).any():
                raise ValueError("Every regression target needs training observations")
            mean = yt / yc
            std = (ys / yc - mean.square()).clamp_min(0).sqrt()
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            self.state.update(
                target_mean=(mean.float() if self.normalize_targets else torch.zeros_like(mean).float()),
                target_std=(std.float() if self.normalize_targets else torch.ones_like(std).float()),
            )
        return self

    def transform(self, sample):
        """
        Standardize learned features and targets, preserving physical p4

        Args:
            sample:  Raw Sample record
        Returns:
            sample: New record with normalized features and targets
        """
        if self.state is None:
            raise RuntimeError("Preprocessor has not been fitted")
        y = sample.target
        mask = sample.target_mask
        if y is not None and "target_mean" in self.state:
            y, mask = self._target_vector(y, mask)
            y = (y - self.state["target_mean"]) / self.state["target_std"]
        g = sample.global_features
        if "global_mean" in self.state:
            if g is None or g.shape != self.state["global_mean"].shape or not torch.isfinite(g).all():
                raise ValueError("Global features must match checkpoint preprocessing")
        elif g is not None:
            raise ValueError("Checkpoint preprocessing does not declare global features")
        x, g = self.transform_inputs(sample.features, g)
        self.validate_parameters(sample.parameters)
        parameters = self.transform_parameters(sample.parameters)
        weight = sample.weight
        if sample.target is not None and "class_weights" in self.state:
            weight *= float(self.state["class_weights"][int(sample.target)])
            if "class_bin_weights" in self.state:
                index = int(self._class_bins([sample], self.state["class_bin_edges"])[0])
                weight *= float(self.state["class_bin_weights"][int(sample.target), index])
        return replace(sample, features=x, target=y, target_mask=mask, weight=weight, global_features=g, parameters=parameters)

    def _target_vector(self, target, mask):
        """
        Align a regression observation and its mask with the fitted coordinates

        Args:
            target:  Raw event target
            mask:  Optional observed coordinates
        Returns:
            result: Target vector and aligned mask
        """
        if target.numel() != self.state["target_mean"].numel():
            raise ValueError("Regression target width must match checkpoint preprocessing")
        if mask is not None and mask.shape != target.shape:
            raise ValueError("target_mask must match regression targets")
        return target.reshape(-1), None if mask is None else mask.reshape(-1)

    def validate_parameters(self, parameters):
        """
        Check finite hypothesis values against the fitted parameter schema

        Args:
            parameters:  Raw parameter tensor or None
        Returns:
            None: Invalid or missing hypothesis values raise ValueError
        """
        if self.state is None:
            raise RuntimeError("Preprocessor has not been fitted")
        if "parameter_mean" in self.state:
            if (
                parameters is None or parameters.ndim != 1
                or parameters.shape[-1:] != self.state["parameter_mean"].shape
                or not torch.isfinite(parameters).all()
            ):
                raise ValueError("Parameters must match checkpoint preprocessing")
        elif parameters is not None:
            raise ValueError("Checkpoint preprocessing does not declare parameters")

    def transform_parameters(self, parameters):
        """
        Normalize validated hypotheses with training-event statistics

        Args:
            parameters:  Raw parameter tensor with any leading batch axes
        Returns:
            parameters: Normalized tensor or None
        """
        if parameters is None:
            return None
        return (parameters - self.state["parameter_mean"]) / self.state["parameter_std"]

    def inverse_parameters(self, parameters):
        """
        Restore the physical hypothesis values used for prediction

        Args:
            parameters:  Normalized parameter tensor
        Returns:
            parameters: Hypotheses in declared physical units
        """
        return parameters * self.state["parameter_std"].to(parameters) + self.state["parameter_mean"].to(parameters)

    def transform_inputs(self, features, global_features=None):
        """
        Normalize validated particle and event tensors with fitted statistics

        Args:
            features:  Raw particle features with any leading batch axes
            global_features:  Optional raw event features
        Returns:
            inputs: Normalized particle and event features
        """
        if features.ndim == 0 or features.shape[-1] != self.state["feature_mean"].numel():
            raise ValueError("Feature width must match checkpoint preprocessing")
        features = (features - self.state["feature_mean"]) / self.state["feature_std"]
        if global_features is not None:
            global_features = (global_features - self.state["global_mean"]) / self.state["global_std"]
        return features, global_features

    def transform_many(self, samples):
        """
        Normalize a batch of ragged events with shared tensor operations

        Args:
            samples:  Raw events in sampler order
        Returns:
            samples: Independent normalized features with physical geometry retained
        """
        if not samples:
            return []
        if getattr(self.transform, "__func__", None) is not Preprocessor.transform:
            return [self.transform(sample) for sample in samples]
        if self.state is None:
            raise RuntimeError("Preprocessor has not been fitted")
        parameters = [sample.parameters for sample in samples]
        for value in parameters:
            self.validate_parameters(value)
        parameters = self.transform_parameters(torch.stack(parameters)).unbind() if parameters[0] is not None else parameters
        globals_ = [sample.global_features for sample in samples]
        if "global_mean" in self.state:
            if any(g is None or g.shape != self.state["global_mean"].shape for g in globals_):
                raise ValueError("Global features must match checkpoint preprocessing")
            globals_ = torch.stack(globals_)
            if not torch.isfinite(globals_).all():
                raise ValueError("Global features must match checkpoint preprocessing")
        elif any(g is not None for g in globals_):
            raise ValueError("Checkpoint preprocessing does not declare global features")
        else:
            globals_ = None
        lengths = [len(sample.features) for sample in samples]
        features, globals_ = self.transform_inputs(torch.cat([sample.features for sample in samples]), globals_)
        features = features.split(lengths)
        globals_ = globals_.unbind() if globals_ is not None else [None] * len(samples)
        targets = [sample.target for sample in samples]
        masks = [sample.target_mask for sample in samples]
        weights = [sample.weight for sample in samples]
        labeled = [i for i, target in enumerate(targets) if target is not None]
        if labeled and ("target_mean" in self.state or "class_weights" in self.state):
            if "target_mean" in self.state:
                for i in labeled:
                    targets[i], masks[i] = self._target_vector(targets[i], masks[i])
            values = torch.stack([targets[i] for i in labeled])
            if "target_mean" in self.state:
                values = (values - self.state["target_mean"]) / self.state["target_std"]
                for i, target in zip(labeled, values, strict=True):
                    targets[i] = target
            if "class_weights" in self.state:
                labels = values.long().reshape(-1)
                factors = self.state["class_weights"][labels]
                if "class_bin_weights" in self.state:
                    bins = self._class_bins([samples[i] for i in labeled], self.state["class_bin_edges"])
                    factors = factors * self.state["class_bin_weights"][labels, bins]
                for i, factor in zip(labeled, factors.tolist(), strict=True):
                    weights[i] *= factor
        return [
            replace(sample, features=x, target=y, target_mask=mask, weight=w, global_features=g, parameters=p)
            for sample, x, y, mask, w, g, p in zip(samples, features, targets, masks, weights, globals_, parameters, strict=True)
        ]

    def inverse_prediction(self, prediction):
        """
        Restore regression predictions and optional uncertainties to physical units

        Args:
            prediction:  Model Prediction in standardized target coordinates
        Returns:
            prediction: Restored Prediction, or unchanged classification logits
        """
        if prediction.mean is None:
            return prediction
        scale = self.state["target_std"].to(prediction.mean)
        shift = self.state["target_mean"].to(prediction.mean)
        components = {}
        if prediction.component_mean is not None:
            components = dict(
                component_mean=prediction.component_mean * scale + shift, component_std=prediction.component_std * scale
            )
        return replace(
            prediction,
            mean=prediction.mean * scale + shift,
            std=prediction.std * scale if prediction.std is not None else None,
            **components,
        )

    def inverse_target(self, target):
        """
        Restore regression targets to physical units

        Args:
            target:  Standardized targets
        Returns:
            target: Physical targets, or unchanged classification labels
        """
        if "target_mean" not in self.state:
            return target
        return target * self.state["target_std"].to(target) + self.state["target_mean"].to(target)

class PreparedDataset(Dataset):
    def __init__(self, dataset, preprocessor, augment=None):
        self.dataset, self.preprocessor, self.augment = dataset, preprocessor, augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.augment is not None:
            sample = self.augment(sample)
        return self.preprocessor.transform(sample)

    def __getitems__(self, indices):
        """
        Fetch and normalize one batch without changing sampler order

        Args:
            indices:  Event indices selected by DataLoader
        Returns:
            samples: Augmented and normalized event records
        """
        samples = fetch_samples(self.dataset, indices)
        if self.augment is not None:
            samples = [self.augment(sample) for sample in samples]
        transform = getattr(self.preprocessor, "transform_many", None)
        return transform(samples) if transform else [self.preprocessor.transform(sample) for sample in samples]
