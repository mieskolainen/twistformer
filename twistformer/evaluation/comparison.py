# Model comparisons on common physics events
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import lru_cache
import csv
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from twistformer.api.config import EvaluationConfig, load_yaml_mapping, strict_dataclass, parse_config
from twistformer.api.contracts import TaskSpec
from twistformer.api.experiment import build_experiment
from twistformer.api.predictor import load_predictor
from twistformer.training.checkpoint import read_checkpoint
from twistboard.outputs import write_json
from twistboard.experiments import timestamp, unique_directory
from .results import compute_metrics, evaluation_weights, write_results
from .categories import category_bins, category_reference, validate_categories, yield_summary
from .runner import collect_predictions
from .comparison_stats import paired_metrics, prediction_spec

def comparison_config(path):
    """
    Validate comparison steering and shared evaluation options

    Args:
        path: YAML card with optional relative includes
    Returns:
        document: Validated configuration mapping
    """
    doc = load_yaml_mapping(path)
    if not isinstance(doc, dict) or set(doc) - {"comparison", "evaluation"} or "comparison" not in doc:
        raise ValueError("Comparison YAML requires comparison and optional evaluation")
    cfg = doc["comparison"]
    allowed = {
        "name",
        "dataset",
        "models",
        "reference_model",
        "output",
        "device",
        "bootstrap_samples",
        "bootstrap_seed",
        "bootstrap_confidence",
        "plots",
    }
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError("Unknown comparison configuration key")
    if not isinstance(cfg.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cfg["name"]):
        raise ValueError("Comparison name must be a path-safe identifier")
    dataset = cfg.get("dataset", {})
    if (
        not isinstance(dataset, dict)
        or set(dataset) - {"reference_checkpoint", "split", "input"}
        or not dataset.get("reference_checkpoint")
    ):
        raise ValueError("Dataset requires reference_checkpoint and optional split/input")
    if dataset.get("split", "test") not in {"train", "val", "test"}:
        raise ValueError("Unknown comparison split")
    models = cfg.get("models", [])
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("Comparison needs at least two models")
    labels = set()
    for model in models:
        if not isinstance(model, dict) or set(model) - {"label", "checkpoint", "evaluation", "batch_size"}:
            raise ValueError("Unknown comparison model key")
        label = model.get("label")
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError("Model labels must be nonempty and unique")
        labels.add(label)
        if bool(model.get("checkpoint")) == bool(model.get("evaluation")):
            raise ValueError("Each model needs exactly one checkpoint or evaluation")
        if "batch_size" in model and (type(model["batch_size"]) is not int or model["batch_size"] < 1):
            raise ValueError("Model batch_size must be positive")
    cfg.setdefault("reference_model", models[0]["label"])
    if cfg["reference_model"] not in labels:
        raise ValueError("reference_model must name a selected model")
    cfg.setdefault("bootstrap_samples", 200)
    cfg.setdefault("bootstrap_seed", 101)
    cfg.setdefault("bootstrap_confidence", 0.6826894921370859)
    n, seed = cfg["bootstrap_samples"], cfg["bootstrap_seed"]
    if type(n) is not int or n < 0 or n == 1 or type(seed) is not int or seed < 0 or not 0 < cfg["bootstrap_confidence"] < 1:
        raise ValueError("Invalid comparison bootstrap settings")
    if type(cfg.get("plots", True)) is not bool:
        raise ValueError("Comparison plots must be Boolean")
    settings = doc.get("evaluation")
    if settings is None:
        settings = read_checkpoint(dataset["reference_checkpoint"])["config"].get("evaluation", {})
    evaluation = strict_dataclass(EvaluationConfig, settings)
    if not evaluation.efficiencies or any(not 0 < x <= 1 for x in evaluation.efficiencies):
        raise ValueError("Efficiencies must be in (0,1]")
    validate_categories(evaluation.categories)
    doc["evaluation"] = asdict(evaluation)
    return doc

@lru_cache(maxsize=256)
def canonical_source(source):
    """
    Resolve a repeated source path once per comparison

    Args:
        source: File part of an event ID
    Returns:
        path: Absolute source path or None for an opaque ID
    """
    return str(Path(source).resolve()) if Path(source).is_file() else None

def event_id(value):
    """
    Normalize file-backed event IDs without changing source entry numbers

    Args:
        value: Reader event ID
    Returns:
        identity: Canonical source path and entry, or original opaque ID
    """
    value = str(value)
    source, sep, entry = value.rpartition(":")
    if sep and canonical_source(source):
        return f"{canonical_source(source)}:{entry}"
    return value

def event_metadata(samples, observables, spec):
    """
    Freeze labels, raw event weights and physical category observables

    Args:
        samples: Reference driver dataset
        observables: Observable schema
        spec: Common task schema
    Returns:
        data: Ordered event metadata arrays
    """
    ids, targets, weights, masks = [], [], [], []
    values = {f"observable_{key}": [] for key in observables}
    for sample in tqdm(samples, desc="Common events", unit="event", bar_format="{l_bar}{bar:20}{r_bar}"):
        if not sample.id or sample.target is None:
            raise ValueError("Comparison requires stable event IDs and labels")
        ids.append(event_id(sample.id))
        target = sample.target.detach().cpu().numpy()
        if spec.name == "classification":
            if target.shape != () or target != int(target):
                raise ValueError("Classification requires integer scalar labels")
            targets.append(int(target))
        else:
            target = target.reshape(-1)
            if target.shape != (len(spec.outputs),):
                raise ValueError("Regression target shape does not match the common schema")
            targets.append(target.copy())
            masks.append(np.isfinite(target) & (sample.target_mask.cpu().numpy().reshape(-1) if sample.target_mask is not None else True))
        weights.append(sample.weight)
        for key in observables:
            values[f"observable_{key}"].append(sample.observables.get(key, np.nan))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Comparison event IDs must be nonempty and unique")
    data = {"ids": np.asarray(ids), "targets": np.asarray(targets), "weights": np.asarray(weights, dtype=float)}
    data.update({key: np.asarray(value, dtype=float) for key, value in values.items()})
    if masks:
        data["target_mask"] = np.asarray(masks, dtype=bool)
    if not np.isfinite(data["weights"]).all() or not np.abs(data["weights"]).sum():
        raise ValueError("Comparison requires finite nonzero physical weights")
    return data

def validate_schemas(source, target):
    """
    Require common task meaning, output names and physical units

    Args:
        source: Model output schema
        target: Reference output schema
    Returns:
        None: Incompatible tasks or units raise ValueError
    """
    if (source.name == "classification") != (target.name == "classification") or set(source.outputs) != set(target.outputs):
        raise ValueError("Incompatible model task or output definitions")
    if source.name != "classification":
        source_units = dict(zip(source.outputs, source.units or ("",) * len(source.outputs)))
        target_units = dict(zip(target.outputs, target.units or ("",) * len(target.outputs)))
        if source_units != target_units:
            raise ValueError("Regression target units differ between models")

class ModelEvents:
    """Model input view with targets mapped to its saved output order"""

    def __init__(self, samples, reference_spec, model_spec):
        """
        Define the mapping from common targets to saved model outputs

        Args:
            samples: Model-specific inputs labeled in the reference order
            reference_spec: Common output schema
            model_spec: Saved model output schema
        """
        self.samples = samples
        validate_schemas(model_spec, reference_spec)
        self.classification = model_spec.name == "classification"
        self.indices = [model_spec.outputs.index(name) for name in reference_spec.outputs]
        self.columns = [reference_spec.outputs.index(name) for name in model_spec.outputs]

    def __len__(self):
        """
        Count events in the model input view

        Returns:
            count: Number of source events
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Map classification labels or regression targets to the saved output order

        Args:
            index: Event index
        Returns:
            sample: Model inputs and reordered targets
        """
        sample = self.samples[index]
        if self.classification:
            return replace(sample, target=torch.tensor(self.indices[int(sample.target)]))
        return replace(
            sample,
            target=sample.target.flatten()[self.columns].reshape_as(sample.target),
            target_mask=(sample.target_mask.flatten()[self.columns].reshape_as(sample.target_mask) if sample.target_mask is not None else None),
        )

def align_predictions(data, common, source_spec, target_spec):
    """
    Align event rows and class columns with the common physics sample

    Args:
        data: Model prediction arrays with raw physical weights
        common: Reference event arrays
        source_spec: Saved model class schema
        target_spec: Common class schema
    Returns:
        aligned: Predictions in the common event and class order
    """
    validate_schemas(source_spec, target_spec)
    ids = [event_id(value) for value in data["ids"]]
    if len(set(ids)) != len(ids) or set(ids) != set(common["ids"]):
        raise ValueError("Model event IDs differ from the common sample or contain duplicates")
    lookup = {value: i for i, value in enumerate(ids)}
    order = np.asarray([lookup[value] for value in common["ids"]])
    columns = [source_spec.outputs.index(name) for name in target_spec.outputs]
    if target_spec.name == "classification":
        targets = np.asarray([target_spec.outputs.index(source_spec.outputs[int(y)]) for y in data["targets"][order]])
        equal = np.array_equal(targets, common["targets"])
    else:
        targets = data["targets"][order].reshape(len(order), -1)[:, columns]
        mask = np.isfinite(targets) & data.get("target_mask", np.ones_like(data["targets"], dtype=bool))[order].reshape(len(order), -1)[:, columns]
        equal = np.array_equal(mask, common["target_mask"]) and np.allclose(
            targets[mask], common["targets"][mask], rtol=1e-5, atol=1e-6
        )
    if not equal:
        raise ValueError("Model labels or regression targets differ from the common sample")
    if "physical_weights" not in data or not np.allclose(data["physical_weights"][order], common["weights"], rtol=1e-7, atol=0):
        raise ValueError("Missing or conflicting physical event weights")
    result = {
        key: value[order]
        for key, value in data.items()
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(order)
    }
    if "weights" in result and "loss_weights" not in result:
        result["loss_weights"] = result["weights"]
    result.update(common)
    if target_spec.name == "classification":
        result["probabilities"] = data["probabilities"][order][:, columns]
    else:
        for key in ("mean", "std"):
            if key in result:
                result[key] = result[key].reshape(len(order), -1)[:, columns]
        for key in ("component_mean", "component_std", "samples"):
            if key in result:
                result[key] = result[key].reshape(len(order), result[key].shape[1], -1)[..., columns]
    result["physical_weights"] = common["weights"]
    if "logits" in result:
        if result["logits"].shape[1] == len(columns):
            result["logits"] = result["logits"][:, columns]
        elif columns == [1, 0]:
            result["logits"] = -result["logits"]
    return result

def verify_sources(sources):
    """
    Reject source files changed during common-event inference

    Args:
        sources: Source paths and initial file statistics
    Returns:
        None: Changed source files raise ValueError
    """
    for path, initial in sources.items():
        current = Path(path).stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (initial.st_ino, initial.st_size, initial.st_mtime_ns):
            raise ValueError(f"Comparison source changed during evaluation: {path}")

def dataset_manifest(common, driver, split, input_path, spec):
    """
    Identify source contents and the fixed physics event selection

    Args:
        common: Reference event metadata
        driver: Reference reader
        split: Data split
        input_path: Optional source override
        spec: Common task schema
    Returns:
        manifest: Dataset identity and source checksums
    """
    candidates = {value.rpartition(":")[0] for value in common["ids"]}
    identity = driver.comparison_identity(split, input_path)
    candidates.update(identity["sources"])
    sources = sorted({str(Path(path).resolve()) for path in candidates if Path(path).is_file()})
    files = []
    for source in sources:
        print(f"Checking dataset: {source}", flush=True)
        with open(source, "rb") as stream:
            files.append({"path": source, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    metadata = {
        "driver": f"{type(driver).__module__}:{type(driver).__name__}",
        "reader": identity,
        "split": split,
        "input": input_path,
        "schema": asdict(spec),
        "sources": files,
        "observables": driver.evaluation_observables,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key, value in sorted(common.items()):
        digest.update(key.encode())
        digest.update(value.tobytes())
    return {**metadata, "fingerprint": digest.hexdigest(), "events": len(common["ids"])}

def write_comparison_table(selection, spec, output):
    """
    Export scalar metrics and paired intervals as flat CSV tables

    Args:
        selection: Common-event comparison result
        spec: Class schema
        output: Existing selection directory
    Returns:
        None: Tables are saved without curve arrays
    """
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "output", "metric", "value"])
        for model in selection["models"]:
            metrics = model["metrics"]
            for name in spec.outputs:
                row = metrics["classes"][name] if spec.name == "classification" else metrics[name]
                for key, value in row.items():
                    if not isinstance(value, (list, dict)):
                        writer.writerow([model["label"], name, key, value])
                for point in row.get("operating_points", []):
                    writer.writerow(
                        [
                            model["label"],
                            name,
                            f"rejection@{point['requested_efficiency']:g}",
                            point.get("rejection") if point.get("physical") else None,
                        ]
                    )
            for key in ("macro_auc", "cross_entropy", "brier"):
                if key in metrics:
                    writer.writerow([model["label"], "all", key, metrics[key]])
    with (output / "paired.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reference",
                "model",
                "metric",
                "kind",
                "estimate",
                "lower",
                "upper",
                "valid_resamples",
                "requested_resamples",
                "confidence",
            ]
        )
        for reference, models in selection.get("paired", {}).items():
            for model, values in models.items():
                for metric, entry in values.items():
                    writer.writerow(
                        [reference, model, metric]
                        + [
                            entry[key]
                            for key in (
                                "kind",
                                "estimate",
                                "lower",
                                "upper",
                                "valid_resamples",
                                "requested_resamples",
                                "confidence",
                            )
                        ]
                    )

def compare(document, output=None):
    """
    Evaluate saved models sequentially on one fixed physics sample

    Args:
        document: Validated comparison card
        output: Optional empty destination directory
    Returns:
        output: Completed comparison directory
    """
    cfg, evaluation = document["comparison"], document["evaluation"]
    canonical_source.cache_clear()
    output = Path(output or cfg.get("output") or unique_directory(Path("runs") / "comparisons", cfg["name"]))
    if output.exists() and any(output.iterdir()):
        names = {path.name for path in output.iterdir()}
        queued = json.loads((output / "comparison.json").read_text()) if "comparison.json" in names else {}
        if (
            names != {"comparison.json", "config.json"}
            or queued.get("status") != "queued"
            or queued.get("launched_by") != "twistboard"
        ):
            raise ValueError(f"Comparison output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "name": cfg["name"],
        "status": "evaluating",
        "started_at": timestamp(),
        "reference_model": cfg["reference_model"],
        "models": [],
        "selections": [],
    }
    write_json(output / "comparison.json", record)
    write_json(output / "config.json", document)
    try:
        with ExitStack() as stack:
            temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="twistformer-compare-")))
            snapshots = {}
            paths = [cfg["dataset"]["reference_checkpoint"]] + [
                model["checkpoint"] for model in cfg["models"] if "checkpoint" in model
            ]
            for index, path in enumerate(dict.fromkeys(str(Path(p).resolve()) for p in paths)):
                print(f"Snapshot checkpoint: {path}", flush=True)
                snapshot = temporary / f"{index}.pt"
                with open(path, "rb") as source, snapshot.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                with snapshot.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                snapshots[path] = (snapshot, digest)
            reference_path = snapshots[str(Path(cfg["dataset"]["reference_checkpoint"]).resolve())][0]
            saved = read_checkpoint(reference_path)
            reference = build_experiment(parse_config(saved["config"])).driver
            del saved
            spec = reference.task_spec
            observables = reference.evaluation_observables
            validate_categories(evaluation["categories"], observables)
            split, input_path = cfg["dataset"].get("split", "test"), cfg["dataset"].get("input")
            sources = reference.comparison_identity(split, input_path)["sources"]
            source_stats = {str(Path(path).resolve()): Path(path).stat() for path in sources}
            record["stage"] = "Reading common events"
            write_json(output / "comparison.json", record)
            samples = reference.dataset(split, input_path)
            common = event_metadata(samples, observables, spec)
            metric_events = {**common, "weights": evaluation_weights(common, observables, evaluation["driver"])}
            if spec.name == "classification" and np.any((common["targets"] < 0) | (common["targets"] >= len(spec.outputs))):
                raise ValueError("Reference labels must index the common classes")
            del samples
            manifest = dataset_manifest(common, reference, split, input_path, spec)
            verify_sources(source_stats)
            manifest["reference_checkpoint"] = str(Path(cfg["dataset"]["reference_checkpoint"]).resolve())
            manifest["reference_sha256"] = snapshots[manifest["reference_checkpoint"]][1]
            write_json(output / "dataset.json", manifest)
            np.savez_compressed(output / "events.npz", **metric_events)
            record["schema"] = asdict(spec)
            record["events"] = len(common["ids"])
            predictions = []
            for index, model in enumerate(cfg["models"]):
                verify_sources(source_stats)
                print(f"Model {index + 1}/{len(cfg['models'])}: {model['label']}", flush=True)
                folder = output / "models" / f"model_{index:03d}"
                folder.mkdir(parents=True)
                record["stage"] = f"Predicting {model['label']} ({index + 1}/{len(cfg['models'])})"
                write_json(output / "comparison.json", record)
                loss = None
                if "checkpoint" in model:
                    path = str(Path(model["checkpoint"]).resolve())
                    snapshot, digest = snapshots[path]
                    predictor = load_predictor(snapshot, cfg.get("device", "auto"))
                    from twistformer.cli.evaluate import report_predictor

                    report_predictor(predictor)
                    app = predictor.experiment.driver
                    view = app.comparison_dataset(reference, split, input_path)
                    view = ModelEvents(view, spec, app.task_spec)
                    data, loss = collect_predictions(
                        predictor,
                        view,
                        model.get("batch_size") or evaluation["batch_size"] or predictor.experiment.config.training.eval_batch_size,
                        require_targets=True,
                        progress=True,
                    )
                    data = align_predictions(data, common, app.task_spec, spec)
                    del predictor, app, view
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    source = Path(model["evaluation"])
                    cached = json.loads((source / "dataset.json").read_text())
                    if cached["fingerprint"] != manifest["fingerprint"]:
                        raise ValueError("Cached evaluation dataset does not match the comparison")
                    saved = json.loads((source / "model.json").read_text())
                    source_spec = TaskSpec(**json.loads((source / "schema.json").read_text()))
                    with np.load(source / "predictions.npz", allow_pickle=False) as arrays:
                        data = align_predictions(dict(arrays), common, source_spec, spec)
                    path, digest = saved["checkpoint"], saved["sha256"]
                    loss = saved.get("preprocessed_loss")
                verify_sources(source_stats)
                entry = {
                    "label": model["label"],
                    "directory": folder.relative_to(output).as_posix(),
                    "checkpoint": path,
                    "sha256": digest,
                    "preprocessed_loss": loss,
                }
                data["weights"] = metric_events["weights"]
                np.savez_compressed(folder / "predictions.npz", **data)
                write_json(folder / "model.json", entry)
                write_json(folder / "dataset.json", manifest)
                write_json(folder / "schema.json", asdict(spec))
                predictions.append(
                    {
                        key: data[key]
                        for key in ("probabilities", "mean", "std", "component_mean", "component_std", "mixture_logits",
                                    "samples", "log_prob", "dequantization_levels")
                        if key in data
                    }
                )
                record["models"].append(entry)
                write_json(output / "comparison.json", record)
            selections = [("inclusive", "Inclusive", np.ones(len(common["ids"]), dtype=bool), True)]
            for category in evaluation["categories"]:
                definition = observables[category["observable"]]
                reference_mask = category_reference(definition, spec, common)
                bins = category_bins(category, common[f"observable_{category['observable']}"], reference_mask)
                for index, (low, high, mask) in enumerate(bins):
                    label = f"{definition.get('label', category['observable'])} [{low if low is not None else '-∞'}, {high if high is not None else '∞'}) {definition.get('unit', '')}".strip()
                    selections.append((f"{category['name']}/bin_{index:03d}", label, mask, (mask & ~reference_mask).any()))
            for name, label, mask, populated in tqdm(
                selections, desc="Comparison selections", unit="selection", bar_format="{l_bar}{bar:20}{r_bar}"
            ):
                record["stage"] = f"Metrics and paired bootstrap: {label}"
                write_json(output / "comparison.json", record)
                selected = {key: value[mask] for key, value in metric_events.items()}
                entry = {"id": name, "label": label, "support": yield_summary(selected, spec), "status": "completed", "models": []}
                if not populated or not np.abs(selected["weights"]).sum():
                    entry.update(status="unavailable", reason="Empty selection or zero absolute weight")
                else:
                    for index, model in enumerate(record["models"]):
                        data = {**selected, **{key: value[mask] for key, value in predictions[index].items()}}
                        metrics, curves = compute_metrics(
                            data, prediction_spec(spec, data), efficiencies=evaluation["efficiencies"], bootstrap_samples=0
                        )
                        folder = output / "selections" / name / f"model_{index:03d}"
                        write_results(data, prediction_spec(spec, data), metrics, curves, folder, plots=False)
                        entry["models"].append(
                            {
                                "label": model["label"],
                                "metrics": metrics,
                                "curves": {
                                    str(i): {
                                        key: value.tolist()
                                        for key, value in curve.items()
                                        if key in {"fpr", "tpr", "precision", "recall"}
                                    }
                                    for i, curve in curves.items()
                                    if "fpr" in curve
                                },
                            }
                        )
                        if spec.name == "classification" and any(row.get("auc") is None for row in metrics["classes"].values()):
                            entry.update(
                                status="partial", reason="AUC unavailable for a class with zero net signal or background weight"
                            )
                        elif spec.name != "classification" and any(metrics[target].get("rmse") is None for target in spec.outputs):
                            entry.update(status="partial", reason="Some targets have no defined RMSE")
                    entry["paired"] = paired_metrics(
                        selected,
                        [{key: value[mask] for key, value in p.items()} for p in predictions],
                        spec,
                        [m["label"] for m in record["models"]],
                        evaluation["efficiencies"],
                        cfg,
                    )
                    if spec.name != "classification":
                        from .comparison_stats import regression_curves

                        views = [{**selected, **{key: value[mask] for key, value in p.items()}} for p in predictions]
                        curves = regression_curves(views, spec, [model["metrics"] for model in entry["models"]])
                        for model, curve in zip(entry["models"], curves):
                            model["curves"] = curve
                    if cfg.get("plots", True):
                        from .comparison_plots import plot_comparison

                        plot_comparison(entry, spec, output / "selections" / name / "plots")
                selection_path = f"selections/{name}/comparison.json"
                (output / selection_path).parent.mkdir(parents=True, exist_ok=True)
                # Browser curves are sampled, exported curve arrays retain every threshold
                for model in entry["models"]:
                    for curve in model["curves"].values():
                        if spec.name != "classification":
                            continue
                        for xkey, ykey in (("fpr", "tpr"), ("recall", "precision")):
                            indices = np.unique(np.linspace(0, len(curve[xkey]) - 1, min(1200, len(curve[xkey])), dtype=int))
                            for key in (xkey, ykey):
                                values = np.asarray(curve[key], dtype=float)[indices]
                                curve[key] = np.where(np.isfinite(values), values, None).tolist()
                write_comparison_table(entry, spec, (output / selection_path).parent)
                write_json(output / selection_path, entry)
                record["selections"].append(
                    {key: entry[key] for key in ("id", "label", "status", "support")} | {"path": selection_path}
                )
                write_json(output / "comparison.json", record)
            record["status"] = "completed"
            record.pop("stage", None)
    except BaseException as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        record["finished_at"] = timestamp()
        write_json(output / "comparison.json", record)
    return output
