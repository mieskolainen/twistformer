# Classifier density ratios for kinematic matching
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
from twistformer.randomness import torch_generator
from twistformer.data.loading import fetch_samples


def event_metadata(dataset, classes):
    """
    Validate event identity, class labels and positive-measure weights

    Args:
        dataset:  Labeled raw events
        classes:  Number of classes
    Returns:
        metadata: Event IDs, integer labels and physical weights
    """
    ids, labels, weights = [], [], []
    for start in range(0, len(dataset), 1024):
        for sample in fetch_samples(dataset, list(range(start, min(start + 1024, len(dataset))))):
            if sample.target is None or sample.target.numel() != 1:
                raise ValueError("Reweighting requires scalar class labels")
            label = float(sample.target)
            if not math.isfinite(label) or label != int(label) or not 0 <= label < classes:
                raise ValueError("Invalid reweighting class label")
            if not math.isfinite(sample.weight) or sample.weight < 0:
                raise ValueError("Density ratios require finite nonnegative event weights")
            ids.append(sample.id)
            labels.append(int(label))
            weights.append(sample.weight)
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Reweighting requires nonempty unique event IDs")
    return ids, torch.tensor(labels), torch.tensor(weights, dtype=torch.float64)


class WeightedDataset(Dataset):
    def __init__(self, dataset, factors):
        """
        Multiply training weights without modifying source events

        Args:
            dataset:  Raw event dataset
            factors:  Finite nonnegative density ratios in event order
        Returns:
            None: Independent weight factors are stored
        """
        self.dataset = dataset
        self.factors = torch.as_tensor(factors, dtype=torch.float64).detach().cpu().clone()
        if self.factors.shape != (len(dataset),) or not torch.isfinite(self.factors).all() or (self.factors < 0).any():
            raise ValueError("Weight factors must be finite nonnegative event vectors")

    def __len__(self):
        """
        Count source events

        Returns:
            size: Source event count
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Apply the density ratio to one physical weight

        Args:
            index:  Event index
        Returns:
            sample: Event with a weighted training loss
        """
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """
        Preserve batched reads and source event contents

        Args:
            indices:  Event indices in sampler order
        Returns:
            samples: Events with weighted training losses
        """
        return [
            replace(sample, weight=sample.weight * float(self.factors[index]))
            for index, sample in zip(indices, fetch_samples(self.dataset, indices), strict=True)
        ]


class ClassifierReweighter:
    def __init__(self, classifier, priors, reference_class=0, *, probability_floor=1e-6, max_weight=100.0):
        """
        Convert any predict_prob classifier into a density-ratio estimator

        Args:
            classifier:  Fitted classifier with predict_prob(samples, batch_size)
            priors:  Effective class priors in the classifier loss
            reference_class:  Target density class index
            probability_floor:  Lower bound on posterior probabilities
            max_weight:  Upper bound on each density ratio
        Returns:
            None: Validated classifier and ratio controls are stored
        """
        self.classifier = classifier
        self.priors = torch.as_tensor(priors, dtype=torch.float64).detach().cpu().clone()
        if self.priors.ndim != 1 or len(self.priors) < 2 or not torch.isfinite(self.priors).all() or (self.priors <= 0).any():
            raise ValueError("Density-ratio priors must be finite and positive for every class")
        if type(reference_class) is not int or not 0 <= reference_class < len(self.priors):
            raise ValueError("reference_class must index the classifier outputs")
        if not 0 < probability_floor < 0.5 or not math.isfinite(max_weight) or max_weight < 1:
            raise ValueError("Require 0 < probability_floor < 0.5 and finite max_weight >= 1")
        if not callable(getattr(classifier, "predict_prob", None)):
            raise TypeError("A reweighting classifier must implement predict_prob")
        self.priors /= self.priors.sum()
        self.reference_class, self.probability_floor, self.max_weight = reference_class, probability_floor, max_weight

    def factors(self, samples, batch_size=1024):
        """
        Estimate p_reference(x) divided by p_class(x)

        Args:
            samples:  Labeled classifier input events
            batch_size:  Inference batch size
        Returns:
            factors: CPU density ratios in event order
        """
        _, labels, _ = event_metadata(samples, len(self.priors))
        probability = torch.as_tensor(self.classifier.predict_prob(samples, batch_size=batch_size)).detach().double().cpu()
        if probability.shape != (len(samples), len(self.priors)) or not torch.isfinite(probability).all():
            raise ValueError("predict_prob must return finite probabilities [events,classes]")
        if (probability < 0).any() or (probability > 1).any() or not torch.allclose(
            probability.sum(1), torch.ones(len(samples), dtype=torch.float64), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("predict_prob rows must be normalized class probabilities")
        probability = probability.clamp_min(self.probability_floor)
        reference = self.reference_class
        factors = probability[:, reference] / probability[torch.arange(len(labels)), labels]
        factors *= self.priors[labels] / self.priors[reference]
        factors = factors.clamp_max(self.max_weight)
        factors[labels == reference] = 1
        return factors


def file_digest(path):
    """
    Hash a saved classifier or ratio manifest

    Args:
        path:  Input file
    Returns:
        digest: SHA256 hex string
    """
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_reweighter(config, run_dir, *, reference_class=0, folds=5, validation_fraction=0.2,
                   probability_floor=1e-6, max_weight=100.0):
    """
    Train density classifiers with the shared Trainer on stratified folds

    Args:
        config:  Classifier config using the observables driver
        run_dir:  New output directory for fold runs and the ratio manifest
        reference_class:  Target density class index
        folds:  Number of disjoint held-out training folds
        validation_fraction:  Inner validation fraction for checkpoint selection
        probability_floor:  Lower posterior bound for ratios
        max_weight:  Upper density-ratio bound
    Returns:
        manifest_path: JSON manifest path for final training reweighting.path
    """
    from twistformer import build_experiment, parse_config, Trainer, load_predictor
    from drivers.observables.driver import Driver

    if config.reweighting.path or config.training.strategy != "single":
        raise ValueError("Fit density-ratio folds with single-process training and no nested reweighting")
    if config.task.name != "classification" or config.task.label_smoothing:
        raise ValueError("Density-ratio training requires classification with label_smoothing=0")
    if type(folds) is not int or folds < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Require folds >= 2 and 0 < validation_fraction < 1")
    experiment = build_experiment(config)
    app = experiment.driver
    if not isinstance(app, Driver) or app.options.event_ids or app.options.splits:
        raise ValueError("fit_reweighter requires an observables driver without fold selections")
    classes = len(app.task_spec.outputs)
    if type(reference_class) is not int or not 0 <= reference_class < classes:
        raise ValueError("reference_class must index the classifier outputs")
    if not 0 < probability_floor < 0.5 or not math.isfinite(max_weight) or max_weight < 1:
        raise ValueError("Require 0 < probability_floor < 0.5 and finite max_weight >= 1")
    if config.task.class_weights and min(config.task.class_weights) <= 0:
        raise ValueError("Density-ratio training needs positive class weights")
    dataset = app.dataset("train")
    ids, labels, weights = event_metadata(dataset, classes)
    generator = torch_generator(config.training.seed)
    assignment = torch.empty(len(dataset), dtype=torch.long)
    for label in range(classes):
        indices = torch.where(labels == label)[0]
        if len(indices) < max(folds, 4) or weights[indices].sum() <= 0:
            raise ValueError("Each class needs positive weight and at least max(folds,4) training events")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        assignment[indices] = torch.arange(len(indices)) % folds
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for fold in range(folds):
        train, val = [], []
        for label in range(classes):
            indices = torch.where((labels == label) & (assignment != fold))[0]
            indices = indices[torch.randperm(len(indices), generator=generator)]
            count = max(1, min(len(indices) - 1, round(validation_fraction * len(indices))))
            val.extend(indices[:count].tolist())
            train.extend(indices[count:].tolist())
        totals = torch.stack([weights[train][labels[train] == label].sum() for label in range(classes)])
        if (totals <= 0).any():
            raise ValueError("Every classifier training fold needs positive weight in each class")
        steering = config.to_dict()
        steering["training"]["seed"] += fold
        steering["driver"]["splits"] = {"train": "train", "val": "train"}
        steering["driver"]["event_ids"] = {"train": [ids[i] for i in train], "val": [ids[i] for i in val]}
        checkpoint = Trainer(root / f"fold_{fold}").fit(build_experiment(parse_config(steering)))
        predictor = load_predictor(checkpoint, config.training.device)
        if config.task.balance_classes:
            totals *= predictor.preprocessor.state["class_weights"].double()
        if config.task.class_weights:
            totals *= torch.tensor(config.task.class_weights, dtype=torch.float64)
        ratio = ClassifierReweighter(predictor, totals, reference_class,
                                    probability_floor=probability_floor, max_weight=max_weight)
        records.append({"checkpoint": str(checkpoint.relative_to(root)), "sha256": file_digest(checkpoint),
                        "priors": ratio.priors.tolist()})
        del predictor, ratio
    manifest = {
        "version": 1, "classes": list(app.task_spec.outputs), "variables": app.options.variables,
        "reference_class": reference_class, "probability_floor": probability_floor, "max_weight": max_weight,
        "held_out": {event_id: int(fold) for event_id, fold in zip(ids, assignment, strict=True)},
        "folds": records,
    }
    manifest_path = root / "reweighting.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


class FoldReweighter:
    def __init__(self, path, device="cpu"):
        """
        Open a frozen set of density-ratio classifiers

        Args:
            path:  Manifest written by fit_reweighter
            device:  Torch inference device
        Returns:
            None: Manifest and checkpoint hashes are validated
        """
        self.path, self.device = Path(path).resolve(), device
        self.state = json.loads(self.path.read_text())
        if self.state["version"] != 1 or len(self.state["folds"]) < 2:
            raise ValueError("Unsupported reweighting manifest")
        if any(type(fold) is not int or not 0 <= fold < len(self.state["folds"]) for fold in self.state["held_out"].values()):
            raise ValueError("Invalid held-out fold assignment")
        self.digest = torch.tensor(list(bytes.fromhex(file_digest(self.path))), dtype=torch.uint8)
        for record in self.state["folds"]:
            if file_digest(self.path.parent / record["checkpoint"]) != record["sha256"]:
                raise ValueError("Reweighting checkpoint changed after fitting")

    def apply(self, dataset, input_spec, task_spec, batch_size=1024):
        """
        Apply held-out ratios to known events and mean ratios to new events

        Args:
            dataset:  Raw final-classifier events
            input_spec:  Final classifier input schema
            task_spec:  Final classifier class order
            batch_size:  Density classifier inference batch size
        Returns:
            dataset: Original inputs with density-ratio loss weights
        """
        from twistformer.api.predictor import load_predictor
        from drivers.observables.driver import ObservableDataset

        if task_spec.name != "classification" or list(task_spec.outputs) != self.state["classes"]:
            raise ValueError("Density-ratio and final classifier class orders must agree")
        ids, _, _ = event_metadata(dataset, len(task_spec.outputs))
        assignment = torch.tensor([self.state["held_out"].get(event_id, -1) for event_id in ids])
        factors = torch.zeros(len(dataset), dtype=torch.float64)
        view = ObservableDataset(dataset, self.state["variables"], input_spec)
        for fold, record in enumerate(self.state["folds"]):
            indices = torch.where((assignment == fold) | (assignment == -1))[0]
            if not len(indices):
                continue
            predictor = load_predictor(self.path.parent / record["checkpoint"], self.device)
            ratio = ClassifierReweighter(predictor, record["priors"], self.state["reference_class"],
                                        probability_floor=self.state["probability_floor"], max_weight=self.state["max_weight"])
            values = ratio.factors(Subset(view, indices.tolist()), batch_size)
            values[assignment[indices] == -1] /= len(self.state["folds"])
            factors[indices] += values
            del predictor, ratio
        return WeightedDataset(dataset, factors)
