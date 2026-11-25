# Evaluation selections in physical observables
#
# mikael.mieskolainen@cern.ch (2026)

import math
import re
import time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from twistboard.outputs import write_json
from twistboard.timing import format_duration
from twistformer.logging import print

def validate_categories(categories, observables=None):
    """
    Validate bin steering and optional driver observable definitions

    Args:
        categories:  List of category mappings
        observables:  Driver observable schema or None
    Returns:
        None: Invalid steering raises ValueError before inference
    """
    if not isinstance(categories, list):
        raise ValueError("Categories must be a list")
    names = set()
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("Each category must be a mapping")
        allowed = {"name", "observable", "edges", "underflow", "overflow"}
        if set(category) - allowed or not {"name", "observable", "edges"} <= set(category):
            raise ValueError("Categories require name, observable and edges with optional underflow and overflow")
        name = category["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Category names must be unique path-safe identifiers")
        names.add(name)
        observable = category["observable"]
        if not isinstance(observable, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", observable):
            raise ValueError("Observable names must be identifiers")
        if observables is not None and observable not in observables:
            raise ValueError(f"Unknown observable {observable}. Available: {', '.join(sorted(observables))}")
        edges = category["edges"]
        if (
            not isinstance(edges, list)
            or len(edges) < 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in edges)
        ):
            raise ValueError("Bin edges must contain at least two finite numbers")
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("Bin edges must increase strictly")
        for flag in ("underflow", "overflow"):
            if flag in category and type(category[flag]) is not bool:
                raise ValueError(f"Category {flag} must be Boolean")

def category_reference(definition, spec, data):
    """
    Keep the other classes inclusive for a class-specific observable

    Args:
        definition:  Driver observable schema
        spec:  Common task and class order
        data:  Event-aligned labels and weights
    Returns:
        reference:  Events shared by all bins of this observable
    """
    name = definition.get("binned_class")
    if name is None:
        return np.zeros(len(data["weights"]), dtype=bool)
    if spec.name != "classification" or name not in spec.outputs:
        raise ValueError(f"Unknown binned class {name} for this task")
    return data["targets"] != spec.outputs.index(name)


def category_bins(category, values, reference=False):
    """
    Select half-open observable bins with an optional common reference sample

    Args:
        category:  Validated category definition
        values:  One physical scalar per event
        reference:  Events included in every bin
    Returns:
        bins: Lower edge, upper edge and event mask for each bin
    """
    edges = category["edges"]
    bounds = list(zip(edges, edges[1:]))
    if category.get("underflow", False):
        bounds.insert(0, (None, edges[0]))
    if category.get("overflow", False):
        bounds.append((edges[-1], None))
    for low, high in bounds:
        mask = np.isfinite(values)
        if low is not None:
            mask &= values >= low
        if high is not None:
            mask &= values < high
        yield low, high, mask | reference

def yield_summary(data, spec):
    """
    Count selected events and signed weighted support

    Args:
        data:  Prediction arrays for one selection
        spec:  Classification or regression schema
    Returns:
        summary: Counts, net yields and effective sample size
    """

    def summary(weights):
        """
        Summarize one signed event-weight array

        Args:
            weights:  Selected event weights
        Returns:
            values: Yield and statistical support
        """
        weights = np.asarray(weights, dtype=float)
        square = float(np.square(weights).sum())
        total = float(weights.sum())
        return {
            "events": len(weights),
            "sum_weights": total,
            "sum_abs_weights": float(np.abs(weights).sum()),
            "sum_weights_squared": square,
            "effective_events": total**2 / square if square else 0.0,
        }

    result = summary(data["weights"])
    if spec.name == "classification":
        result["classes"] = {name: summary(data["weights"][data["targets"] == index]) for index, name in enumerate(spec.outputs)}
    return result

def evaluate_categories(data, spec, categories, observables, output, *, plots=True, progress=False, **metric_options):
    """
    Reuse saved predictions for metrics and curves in physical bins

    Args:
        data:  Event-aligned prediction arrays including observable_NAME arrays
        spec:  Task output schema
        categories:  YAML bin definitions
        observables:  Observable labels and units
        output:  Evaluation directory containing inclusive results
        plots:  Write category figures and ROC overlays
        progress:  Show category and bootstrap progress
        metric_options:  Shared classification metric options
    Returns:
        manifest: Category definitions, coverage and result entries
    """
    from .results import compute_metrics, write_results

    validate_categories(categories, observables)
    output = Path(output)
    size = len(data["weights"])
    for category in categories:
        name = category["observable"]
        values = data.get(f"observable_{name}")
        if values is None or np.asarray(values).shape != (size,):
            raise ValueError(f"Missing or misaligned evaluation observable: {name}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "interval": "[low, high)", "thresholds": "within_category", "categories": []}
    for category in categories:
        name = category["name"]
        values = data[f"observable_{category['observable']}"]
        definition = observables[category["observable"]]
        reference = category_reference(definition, spec, data)
        group = {
            **category,
            "definition": definition,
            "missing_events": int((~np.isfinite(values) & ~reference).sum()),
            "out_of_range_events": 0,
            "bins": [],
        }
        if "binned_class" in definition:
            group["reference_events"] = int(reference.sum())
        selected = np.zeros(size, dtype=bool)
        overlays = []
        bins = list(category_bins(category, values, reference))
        for index, (low, high, mask) in enumerate(
            tqdm(
                bins,
                desc=f"Category {name}",
                unit="bin",
                bar_format="{l_bar}{bar:20}{r_bar}",
                disable=not progress,
                dynamic_ncols=True,
            )
        ):
            selected |= mask
            subset = {
                key: value[mask]
                for key, value in data.items()
                if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == size
            }
            folder = f"categories/{name}/bin_{index:03d}"
            interval = f"[{low if low is not None else '-∞'}, {high if high is not None else '∞'})"
            label = f"{definition.get('label', category['observable'])} {interval} {definition.get('unit', '')}".strip()
            entry = {
                "id": f"{name}/bin_{index:03d}",
                "directory": folder,
                "label": label,
                "low": low,
                "high": high,
                "support": yield_summary(subset, spec),
                "status": "completed",
            }
            if not (mask & ~reference).any():
                entry.update(status="unavailable", reason="Empty bin")
            elif not np.abs(subset["weights"]).sum():
                entry.update(status="unavailable", reason="Zero absolute event weight")
            if entry["status"] == "completed":
                metrics, curves = compute_metrics(subset, spec, progress=progress, **metric_options)
                if spec.name == "classification" and any(value.get("auc") is None for value in metrics["classes"].values()):
                    entry.update(status="partial", reason="ROC unavailable: zero net signal or background weight")
                if spec.name == "classification" and any(
                    np.any((curve[key] < 0) | (curve[key] > 1))
                    for curve in curves.values()
                    for key in ("fpr", "tpr")
                    if key in curve
                ):
                    entry.update(status="partial", reason="Signed-weight ROC contains nonphysical efficiencies")
                write_results(
                    subset,
                    spec,
                    metrics,
                    curves,
                    output / folder,
                    plots=plots and (spec.name != "classification" or bool(curves)),
                    selection_label=label,
                )
                bin_label = f"{interval.replace(', ', ',')} {definition.get('unit', '')}".strip()
                overlays.append((bin_label, curves))
            else:
                (output / folder).mkdir(parents=True, exist_ok=True)
                write_json(output / folder / "metrics.json", {"status": entry["status"], "reason": entry["reason"]})
            group["bins"].append(entry)
        group["out_of_range_events"] = int((np.isfinite(values) & ~selected).sum())
        if plots and spec.name == "classification" and overlays:
            from .plots import plot_category_rocs

            plot_category_rocs(
                overlays,
                spec,
                output / "categories" / name / "plots",
                selection_label=definition.get("label", category["observable"]),
                signed=bool((np.asarray(data["weights"]) < 0).any()),
            )
        manifest["categories"].append(group)
    write_json(output / "categories.json", manifest)
    return manifest

def rebin_evaluation(source, output, config, *, plots=True, progress=False, **metric_options):
    """
    Evaluate new categories from saved predictions without loading a model

    Args:
        source:  Completed evaluation with saved observable arrays
        output:  Empty destination directory
        config:  Validated evaluation steering
        plots:  Write diagnostic figures
        progress:  Show bin and bootstrap progress
        metric_options:  Bootstrap steering overrides
    Returns:
        manifest: Category results for the new evaluation
    """
    import json
    import shutil
    from dataclasses import asdict
    from twistformer.api.contracts import TaskSpec
    from twistboard.experiments import timestamp
    from .results import compute_metrics, evaluation_weights, write_results

    metric_options.setdefault("bootstrap_samples", config.bootstrap_samples)
    metric_options.setdefault("bootstrap_seed", config.bootstrap_seed)
    source, output = Path(source).resolve(), Path(output).resolve()
    record = json.loads((source / "evaluation.json").read_text())
    if record.get("status") != "completed":
        raise ValueError("Rebinning requires a completed source evaluation")
    observables = json.loads((source / "observables.json").read_text())
    validate_categories(config.categories, observables)
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        data = dict(saved)
    source_config_file = source / "evaluation_config.json"
    source_config = json.loads(source_config_file.read_text()) if source_config_file.is_file() else {}
    source_driver = source_config.get("driver", {})
    driver = {**source_driver, **config.driver}
    change_weights = driver.get("energy_weights") != source_driver.get("energy_weights")
    spec = TaskSpec(**json.loads((source / "schema.json").read_text()))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record = {key: record[key] for key in ("schema_version", "checkpoint", "sha256", "split", "input") if key in record}
    record.update(status="evaluating", started_at=timestamp(), source_evaluation=str(source))
    started = time.monotonic()
    write_json(output / "evaluation.json", record)
    try:
        shutil.copyfile(source / "observables.json", output / "observables.json")
        settings = {**asdict(config), **metric_options, "driver": driver}
        if change_weights:
            data["weights"] = evaluation_weights(data, observables, driver)
            metrics, curves = compute_metrics(data, spec, efficiencies=config.efficiencies, progress=progress, **metric_options)
            metrics["preprocessed_loss"] = json.loads((source / "metrics.json").read_text()).get("preprocessed_loss")
            np.savez_compressed(output / "predictions.npz", **data)
            write_results(data, spec, metrics, curves, output, plots=plots)
        else:
            for name in ("predictions.npz", "metrics.json", "curves.npz", "schema.json"):
                shutil.copyfile(source / name, output / name)
            if (source / "plots").is_dir() and plots:
                shutil.copytree(source / "plots", output / "plots")
            settings["inclusive_source"] = str(source)
        write_json(output / "evaluation_config.json", settings)
        result = evaluate_categories(
            data,
            spec,
            config.categories,
            observables,
            output,
            plots=plots,
            progress=progress,
            efficiencies=config.efficiencies,
            **metric_options,
        )
    except BaseException as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}"
        )
        raise
    else:
        record["status"] = "completed"
        return result
    finally:
        record["finished_at"] = timestamp()
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "evaluation.json", record)
        print(f"Evaluation time: {format_duration(record['elapsed_seconds'])}", flush=True)
