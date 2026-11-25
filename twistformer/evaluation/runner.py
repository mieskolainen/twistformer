# Prediction diagnostics and evaluation
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
from dataclasses import fields, asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from twistformer.training.engine import validate_batch
from twistformer.api.contracts import Prediction
from .results import compute_metrics, evaluation_weights, write_results
from .categories import validate_categories, evaluate_categories
from twistboard.outputs import write_json

@torch.inference_mode()
def collect_predictions(predictor, samples, batch_size=128, require_targets=False, *, progress=False, parameters=None):
    """
    Collect predictions with event IDs and original target units

    Args:
        predictor:  Restored Predictor
        samples:  Raw Sample records
        batch_size:  Maximum events per batch
        progress:  Show inference batch progress
        require_targets:  Require labels and evaluate the training-coordinate loss
        parameters:  Optional named hypotheses in physical units
    Returns:
        result: Prediction arrays and optional mean loss
    """
    samples = predictor.condition(samples, parameters)
    arrays = {}
    total_loss = total_weight = 0.0
    offset = 0
    task = predictor.experiment.task
    classification_loss = (getattr(task, "classification_loss_terms", None)
                           if require_targets and task.spec.name == "classification" else None)
    loader = predictor.loader(samples, batch_size)
    for batch in tqdm(
        loader,
        desc="Evaluation" if require_targets else "Prediction",
        unit="batch",
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
        disable=not progress,
    ):
        batch = batch.to(predictor.device, non_blocking=loader.pin_memory)
        validate_batch(batch, predictor.model.requires_p4)
        prediction = predictor.forward(batch)
        loss_values = {}
        if require_targets and not (task.spec.name.startswith("flow_") and (batch.sample_weights < 0).any()):
            if classification_loss is None:
                loss = task.compute_loss(prediction, batch)
                total_loss += float(loss.numerator)
                total_weight += float(loss.normalizer)
            else:
                losses, weights = classification_loss(prediction, batch)
                weighted_loss = losses * weights
                total_loss += float(weighted_loss.sum())
                total_weight += float(weights.sum())
                loss_values = {"evaluation_loss_numerator": weighted_loss.cpu().numpy(),
                               "evaluation_loss_normalizer": weights.cpu().numpy()}
        physical = predictor.preprocessor.inverse_prediction(prediction)
        values = {
            **loss_values,
            "ids": np.asarray(batch.ids),
            "weights": batch.sample_weights.cpu().numpy(),
            "physical_weights": np.asarray([samples[i].weight for i in range(offset, offset + len(batch.ids))], dtype=float),
        }
        offset += len(batch.ids)
        if batch.targets is not None:
            target = predictor.preprocessor.inverse_target(batch.targets)
            values["targets"] = target.cpu().numpy()
        if batch.target_mask is not None:
            values["target_mask"] = batch.target_mask.cpu().numpy()
        if batch.parameters is not None:
            raw_parameters = predictor.preprocessor.inverse_parameters(batch.parameters).cpu().numpy()
            for index, name in enumerate(predictor.experiment.driver.input_spec.parameters):
                values[f"parameter_{name}"] = raw_parameters[:, index]
        for key in (field.name for field in fields(Prediction)):
            if key in {"flow_loss", "flow_weights"}:
                continue
            tensor = getattr(physical, key)
            if tensor is not None:
                valid = ~torch.isnan(tensor) & ~torch.isposinf(tensor) if key == "log_prob" else torch.isfinite(tensor)
                if not valid.all():
                    raise FloatingPointError(f"Nonfinite prediction: {key}")
                values[key] = tensor.cpu().numpy()
        levels = predictor.preprocessor.state.get("dequantization_levels")
        if levels is not None:
            values["dequantization_levels"] = np.full(len(batch.ids), int(levels))
        if physical.logits is not None:
            values["probabilities"] = physical.probabilities.cpu().numpy()
        if physical.mixture_logits is not None:
            values["mixture_probabilities"] = physical.mixture_logits.softmax(-1).cpu().numpy()
        for name, value in batch.observables.items():
            values[f"observable_{name}"] = value.cpu().numpy()
        if arrays and set(values) != set(arrays):
            raise ValueError("Prediction fields and observables must agree across batches")
        for key, value in values.items():
            arrays.setdefault(key, []).append(value)
    if not arrays:
        raise ValueError("No evaluation samples")
    data = {key: np.concatenate(parts) for key, parts in arrays.items()}
    export = getattr(predictor.experiment.driver, "prediction_extra", None)
    if export is not None:
        data.update(export(data))
    signed_flow = predictor.experiment.task.spec.name.startswith("flow_") and (data["weights"] < 0).any()
    return data, total_loss / total_weight if total_weight and not signed_flow else None

def evaluate(
    predictor,
    samples,
    output,
    batch_size=None,
    plots=True,
    efficiencies=None,
    *,
    bootstrap_samples=None,
    bootstrap_seed=None,
    bootstrap_confidence=0.6826894921370859,
    progress=False,
    evaluation_config=None,
    parameters=None,
):
    """
    Write task metrics, prediction arrays and optional diagnostic plots

    Args:
        predictor:  Restored Predictor
        samples:  Labeled raw Sample records
        output:  Evaluation output directory
        batch_size:  Maximum events per batch
        plots:  Write PNG and PDF diagnostics
        efficiencies:  Signal efficiencies for descriptive ROC operating points
        bootstrap_samples:  Replica override, None uses evaluation steering, zero disables bands
        bootstrap_seed:  Bootstrap seed override, None uses evaluation steering
        bootstrap_confidence:  Central percentile coverage
        progress:  Show inference and bootstrap progress
        evaluation_config:  Optional validated evaluation-only steering override
        parameters:  Optional named hypotheses in physical units
    Returns:
        metrics: Task diagnostics in physical target units
    """
    config = evaluation_config or predictor.experiment.config.evaluation
    if bootstrap_samples is None:
        bootstrap_samples = config.bootstrap_samples
    if bootstrap_seed is None:
        bootstrap_seed = config.bootstrap_seed
    if batch_size is None:
        batch_size = config.batch_size or predictor.experiment.config.training.eval_batch_size
    observables = getattr(predictor.experiment.driver, "evaluation_observables", {})
    validate_categories(config.categories, observables)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, loss = collect_predictions(
        predictor, samples, batch_size, require_targets=True, progress=progress, parameters=parameters,
    )
    data["loss_weights"] = data["weights"]
    data["weights"] = evaluation_weights(data, observables, config.driver)
    spec = predictor.experiment.driver.task_spec
    if efficiencies is None:
        efficiencies = config.efficiencies
    options = dict(
        efficiencies=efficiencies,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
    )
    metrics, curves = compute_metrics(data, spec, progress=progress, **options)
    metrics["preprocessed_loss"] = loss
    np.savez_compressed(output / "predictions.npz", **data)
    write_json(output / "observables.json", observables)
    write_json(output / "evaluation_config.json", {**asdict(config), **options, "batch_size": batch_size})
    if plots and progress:
        print("Saving evaluation plots...", flush=True)
    write_results(data, spec, metrics, curves, output, plots=plots)
    if plots:
        extra_plots = getattr(predictor.experiment.driver, "plot_extra", None)
        if extra_plots is not None:
            extra_plots(data, output / "plots")
    if config.categories:
        evaluate_categories(data, spec, config.categories, observables, output, plots=plots, progress=progress, **options)
    diagnostic = getattr(predictor.experiment.driver, "evaluate_extra", None)
    if diagnostic is not None:
        diagnostic(data, output)
    return metrics
