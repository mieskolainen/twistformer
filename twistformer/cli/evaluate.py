# Command-line experiment steering
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
import argparse
import json
from pathlib import Path
import numpy as np
from twistformer.api.predictor import load_predictor
from twistformer.evaluation.runner import evaluate, collect_predictions
from twistboard.experiments import unique_directory, evaluation_run

def evaluation_override(path, base):
    """
    Merge evaluation-only YAML without modifying checkpoint model steering

    Args:
        path:  YAML path or None
        base:  Saved evaluation configuration
    Returns:
        config: Validated evaluation settings
    """
    from dataclasses import asdict
    from twistformer.api.config import EvaluationConfig, load_yaml_mapping, strict_dataclass
    from twistformer.evaluation.categories import validate_categories

    if path is None:
        return base
    document = load_yaml_mapping(path)
    if not document or document.keys() - {"evaluation", "driver"} or not isinstance(document.get("evaluation", {}), dict):
        raise ValueError("Evaluation YAML must contain only an evaluation mapping and optional driver overrides")
    settings = {**asdict(base), **document.get("evaluation", {})}
    if "driver" in document:
        if "driver" in document.get("evaluation", {}):
            raise ValueError("Specify evaluation driver overrides in one mapping")
        settings["driver"] = document["driver"]
    config = strict_dataclass(EvaluationConfig, settings)
    if not config.efficiencies or any(not 0 < value <= 1 for value in config.efficiencies):
        raise ValueError("Evaluation efficiencies must be nonempty and in (0,1]")
    validate_categories(config.categories)
    return config

def report_predictor(predictor):
    """
    Print the restored model size and active inference device

    Args:
        predictor:  Loaded inference model and resolved device
    Returns:
        None: Startup details are flushed to the terminal and evaluation log
    """
    import torch
    from twistformer.api.experiment import parameter_summary

    device = predictor.device
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Active device: cuda:{index} ({torch.cuda.get_device_name(index)})", flush=True)
    else:
        print(f"Active device: {device}", flush=True)
    print(parameter_summary(predictor.model), flush=True)

def main(argv=None, *, prediction_only=False):
    """
    Evaluate or export predictions from a portable checkpoint

    Args:
        argv:  Command-line arguments, or None for process arguments
        prediction_only:  Export arrays without requiring target labels
    Returns:
        None: Evaluation files are written to disk
    """
    p = argparse.ArgumentParser(description="Load a checkpoint and evaluate or predict in a fresh process")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--from-evaluation", help="Rebin saved predictions without loading a checkpoint")
    p.add_argument("--evaluation-config", help="YAML containing evaluation steering only")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--input", help="Driver-specific input path. overrides the configured split")
    p.add_argument("--parameter", action="append", default=[], metavar="NAME=VALUE", help="Set a conditioning hypothesis in physical units")
    p.add_argument(
        "--output", required=prediction_only, help="Evaluation directory (default: unique directory next to the run checkpoints)"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--root-workers", type=int, help="Override pi0gamma ROOT readers, 0 uses all available logical CPUs")
    p.add_argument("--batch-size", type=int, help="Override evaluation.batch_size (fallback: training.eval_batch_size)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--efficiencies", type=float, nargs="+", help="Default: checkpoint evaluation.efficiencies")
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Override evaluation.bootstrap_samples, zero disables bands (default: card or 200)",
    )
    p.add_argument("--bootstrap-seed", type=int, help="Override evaluation.bootstrap_seed")
    p.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.6826894921370859,
        help="Central bootstrap coverage (default: approximately 68.27 percent)",
    )
    a = p.parse_args(argv)
    if a.root_workers is not None and (a.root_workers < 0 or a.from_evaluation):
        p.error("--root-workers needs a nonnegative count and checkpoint input")

    def load_for_reading(checkpoint):
        """Restore the model and apply runtime ROOT reader settings"""
        predictor = load_predictor(checkpoint, a.device)
        if a.root_workers is not None:
            options = getattr(predictor.experiment.driver, "options", None)
            if not hasattr(options, "root_workers"):
                p.error("The checkpoint driver does not support --root-workers")
            options.root_workers = a.root_workers
        report_predictor(predictor)
        return predictor

    parameters = {}
    for item in a.parameter:
        name, separator, value = item.partition("=")
        try:
            number = float(value)
        except ValueError:
            p.error("Each --parameter must be NAME=VALUE with a finite number")
        if not separator or not name or name in parameters or not np.isfinite(number):
            p.error("Parameter names must be unique and values must be finite")
        parameters[name] = number
    parameters = parameters or None
    if ((a.bootstrap_samples is not None and (a.bootstrap_samples < 0 or a.bootstrap_samples == 1))
            or not 0 < a.bootstrap_confidence < 1 or (a.bootstrap_seed is not None and a.bootstrap_seed < 0)):
        p.error("Bootstrap needs zero or >=2 replicas, a nonnegative seed and coverage in (0,1)")
    if (a.batch_size is not None and a.batch_size < 1) or any(not 0 < v <= 1 for v in (a.efficiencies or [])):
        p.error("Positive batch size and efficiencies in (0,1] are required")
    if a.from_evaluation:
        if prediction_only or a.input or parameters:
            p.error("--from-evaluation cannot be combined with prediction export, --input or --parameter")
        from twistformer.api.config import EvaluationConfig
        from twistformer.evaluation.categories import rebin_evaluation

        source = Path(a.from_evaluation)
        saved_config = source / "evaluation_config.json"
        saved = json.loads(saved_config.read_text()) if saved_config.is_file() else {}
        config = evaluation_override(a.evaluation_config, EvaluationConfig(
            efficiencies=saved.get("efficiencies", [0.5, 0.8]),
            bootstrap_samples=saved.get("bootstrap_samples", 200),
            bootstrap_seed=saved.get("bootstrap_seed", 101),
            categories=saved.get("categories", []),
        ))
        if a.efficiencies is not None:
            config.efficiencies = a.efficiencies
        if not config.categories and "energy_weights" not in config.driver:
            p.error("--from-evaluation requires categories or a gun energy weight override")
        output = Path(a.output) if a.output else unique_directory(source.resolve().parent, "rebinned")
        print(f"Rebinning saved predictions: {source}", flush=True)
        rebin_evaluation(
            source,
            output,
            config,
            plots=not a.no_plots,
            progress=True,
            bootstrap_samples=config.bootstrap_samples if a.bootstrap_samples is None else a.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed if a.bootstrap_seed is None else a.bootstrap_seed,
            bootstrap_confidence=a.bootstrap_confidence,
        )
        print(f"Evaluation: {output}", flush=True)
        return
    if prediction_only:
        if a.evaluation_config:
            p.error("--evaluation-config applies to evaluation, not prediction export")
        print(f"Loading checkpoint: {a.checkpoint}", flush=True)
        predictor = load_for_reading(a.checkpoint)
        driver = predictor.experiment.driver
        reader = getattr(driver, "prediction_dataset", None)
        print(f"Loading prediction data: {a.input or a.split}", flush=True)
        samples = reader(a.input) if a.input and reader else driver.dataset(a.split, a.input)
        batch_size = (
            a.batch_size
            or predictor.experiment.config.evaluation.batch_size
            or predictor.experiment.config.training.eval_batch_size
        )
        print(f"Loaded {len(samples):,} events", flush=True)
        print(f"Data loader: prediction batch {batch_size}, workers {predictor.experiment.config.training.workers}", flush=True)
        print("Running inference...", flush=True)
        data, _ = collect_predictions(predictor, samples, batch_size, progress=True, parameters=parameters)
        destination = Path(a.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as f:
            np.savez_compressed(f, **data)
        print(f"Saved {len(data['ids'])} predictions to {destination}", flush=True)
    else:
        checkpoint = Path(a.checkpoint).resolve()
        parent = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
        output = Path(a.output) if a.output else unique_directory(parent / "evaluations", f"{a.split}-{checkpoint.stem}")
        print(f"Evaluation checkpoint: {checkpoint}", flush=True)
        print(f"Evaluation output: {output}", flush=True)
        print("Reading checkpoint and computing checksum...", flush=True)
        with evaluation_run(checkpoint, output, a.split, a.input) as snapshot:
            print("Loading model and saved preprocessing...", flush=True)
            predictor = load_for_reading(snapshot)
            config = evaluation_override(a.evaluation_config, predictor.experiment.config.evaluation)
            from twistformer.evaluation.categories import validate_categories

            validate_categories(config.categories, getattr(predictor.experiment.driver, "evaluation_observables", {}))
            print(f"Loading evaluation data: {a.input or a.split}", flush=True)
            samples = predictor.experiment.driver.dataset(a.split, a.input)
            batch_size = a.batch_size or config.batch_size or predictor.experiment.config.training.eval_batch_size
            print(f"Loaded {len(samples):,} events", flush=True)
            print(f"Data loader: evaluation batch {batch_size}, workers {predictor.experiment.config.training.workers}", flush=True)
            replicas = config.bootstrap_samples if a.bootstrap_samples is None else a.bootstrap_samples
            print(f"Running inference and metrics with {replicas} bootstrap replicas...", flush=True)
            metrics = evaluate(
                predictor,
                samples,
                output,
                batch_size,
                not a.no_plots,
                a.efficiencies,
                bootstrap_samples=config.bootstrap_samples if a.bootstrap_samples is None else a.bootstrap_samples,
                bootstrap_seed=a.bootstrap_seed,
                evaluation_config=config,
                bootstrap_confidence=a.bootstrap_confidence,
                progress=True,
                parameters=parameters,
            )
            print(json.dumps(metrics, indent=2))
            print(f"Evaluation: {output}", flush=True)

def predict(argv=None):
    """
    Export checkpoint predictions without requiring labels

    Args:
        argv:  Command-line arguments, or None for the process arguments
    Returns:
        None: Prediction arrays are written to the requested output file
    """
    main(argv, prediction_only=True)

if __name__ == "__main__":
    main()
