# Fit a post-training classifier temperature and save a portable model
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from twistformer.api.config import CalibrationConfig, load_yaml_mapping, strict_dataclass
from twistformer.api.predictor import load_predictor
from twistformer.evaluation.runner import collect_predictions
from twistformer.evaluation.temperature import fit_temperature
from twistformer.training.checkpoint import read_checkpoint


def main(argv=None):
    """
    Fit a temperature on the checkpoint validation split in a separate procedure

    Args:
        argv:  Command line arguments or None for process arguments
    Returns:
        None:  A new inference checkpoint contains the fitted temperature
    """
    parser = argparse.ArgumentParser(description="Fit validation temperature and save a calibrated classifier")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Steering card containing a calibration section")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output checkpoint path")
    with args.checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        state = read_checkpoint(source)
        settings = state["config"].get("calibration", {})
        if args.config:
            card = load_yaml_mapping(args.config)
            if "calibration" not in card:
                parser.error("The steering card requires a calibration section")
            settings = {**settings, **card["calibration"]}
        steering = strict_dataclass(CalibrationConfig, settings)
        if not steering.temperature_scaling:
            parser.error("Set calibration.temperature_scaling to true for this procedure")
        if state["task_spec"]["name"] != "classification" or state.get("calibration"):
            parser.error("Select an uncalibrated classification checkpoint")
        source.seek(0)
        predictor = load_predictor(source, args.device)
    samples = predictor.experiment.driver.dataset("val")
    config = predictor.experiment.config
    batch_size = config.evaluation.batch_size or config.training.eval_batch_size
    data, _ = collect_predictions(predictor, samples, batch_size, require_targets=True, progress=True)
    record = fit_temperature(data["logits"], data["targets"], data["weights"], steering)
    record["source_sha256"] = digest
    state.update(calibration=record, inference_only=True)
    state["config"]["calibration"] = vars(steering)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".pt") as temporary:
        torch.save(state, temporary)
        temporary.flush()
        os.link(temporary.name, args.output)
    print(json.dumps(record, indent=2))
    print(f"Calibrated checkpoint {args.output}")


if __name__ == "__main__":
    main()
