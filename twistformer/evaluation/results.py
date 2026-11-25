# Shared inclusive and category evaluation files
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict
from pathlib import Path
import numpy as np
from twistboard.outputs import write_json
from .metrics import classification_metrics, regression_metrics
from twistformer.data.spectrum import normalize_energy_weights, report_energy_weights, spectrum_weights


def evaluation_weights(data, observables, driver):
    """
    Select physical event weights for evaluation metrics

    Args:
        data:  Prediction arrays with event weights and truth observables
        observables:  Observable definitions with the original gun energy edges
        driver:  Evaluation driver overrides
    Returns:
        weights:  Physical weights for the requested spectrum
    """
    weights = driver.get("energy_weights")
    if weights is None:
        return data.get("physical_weights", data["weights"])
    edges = observables.get("truth_energy", {}).get("gun_energy_edges")
    if not edges or "observable_generator_weight" not in data:
        raise ValueError("Overriding energy_weights requires pi0gamma gun bins and generator weights. Evaluate from a checkpoint to record them")
    fractions = normalize_energy_weights(weights, edges)
    report_energy_weights("evaluation.driver.energy_weights", weights, fractions)
    bins = np.searchsorted(edges, data["observable_truth_energy"], side="right") - 1
    if ((bins < 0) | (bins >= len(fractions))).any():
        raise ValueError("Truth energies must lie within energy_edges")
    return spectrum_weights(data["observable_generator_weight"], data["targets"], bins, fractions)

def compute_metrics(data, spec, **options):
    """
    Evaluate a physical event selection with the shared task metrics

    Args:
        data:  Selected prediction arrays
        spec:  Task schema
        options:  Classification uncertainty and operating-point settings
    Returns:
        result: Metric mapping and curve arrays
    """
    if spec.name == "classification":
        return classification_metrics(data, spec, **options)
    if spec.name.startswith("flow_"):
        from twistformer.flows.evaluation import flow_metrics

        settings = {key: options[key] for key in ("bootstrap_samples", "bootstrap_seed", "bootstrap_confidence") if key in options}
        return flow_metrics(data, spec, **settings), {}
    return regression_metrics(data, spec), {}

def write_results(data, spec, metrics, curves, output, *, plots=True, selection_label=None):
    """
    Save common metric and curve files without duplicating predictions

    Args:
        data:  Selected predictions used for plots
        spec:  Task schema
        metrics:  Computed metrics
        curves:  Class-indexed curve arrays
        output:  Destination directory
        plots:  Write diagnostic PNG and PDF figures
        selection_label:  Optional bin label on exported figures
    Returns:
        None: Portable evaluation files are written
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_json(output / "schema.json", asdict(spec))
    np.savez_compressed(
        output / "curves.npz", **{f"class_{i}_{key}": value for i, curve in curves.items() for key, value in curve.items()}
    )
    if plots:
        if spec.name.startswith("flow_"):
            from twistformer.flows.evaluation import plot_flow_results

            plot_flow_results(data, spec, output / "plots", selection_label)
            return
        from .plots import plot_results

        plot_results(data, spec, metrics, curves, output / "plots", selection_label=selection_label)
