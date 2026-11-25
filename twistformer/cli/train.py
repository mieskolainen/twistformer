# Command-line experiment steering
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
import argparse
import os
import sys
from pathlib import Path
from twistformer.api.config import load_config, parse_config
from twistformer.api.experiment import build_experiment, parameter_summary
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.engine import Trainer

def main(argv=None):
    """
    Run training and exit quietly on a keyboard interrupt

    Args:
        argv: Command-line arguments or None for process arguments
    Returns:
        None: Training completes or exits with status 130 on interruption
    """
    try:
        return train(argv)
    except KeyboardInterrupt:
        raise SystemExit(130) from None

def train(argv=None):
    """
    Run configuration validation, new training or checkpoint resume

    Args:
        argv:  Command-line arguments, or None for process arguments
    Returns:
        None: Training files are written to the run directory
    """
    p = argparse.ArgumentParser(description="Train any registered driver/model/task")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume")
    p.add_argument("--init-from")
    p.add_argument("--run-name", default="experiment")
    p.add_argument("--output", help="Run directory (default: runs/RUN_NAME)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--device")
    p.add_argument("--strategy", choices=("single", "ddp"), help="Training strategy (default: single)")
    p.add_argument("--workers", type=int)
    p.add_argument("--validate", action="store_true", help="Validate configuration without reading data or training")
    p.add_argument("--reweight-folds", type=int, default=0, help="Fit held-out density classifiers with this many folds")
    p.add_argument("--reference-class", type=int, default=0, help="Target density class for --reweight-folds")
    a = p.parse_args(argv)
    if a.reweight_folds and (a.reweight_folds < 2 or a.resume or a.init_from):
        p.error("--reweight-folds requires at least two folds and a new --config run")
    if a.resume:
        if a.set or a.init_from:
            p.error("Resume accepts runtime --device/--workers only. use --config --init-from for a new run")
        cfg = parse_config(read_checkpoint(a.resume)["config"])
    else:
        cfg = load_config(a.config, a.set)
    if a.device:
        cfg.training.device = a.device
    if a.strategy:
        cfg.training.strategy = a.strategy
    if a.workers is not None:
        cfg.training.workers = a.workers
    cfg = parse_config(cfg.to_dict())
    experiment = build_experiment(cfg)
    if a.validate:
        print(f"Valid: {cfg.driver['name']} / {cfg.model.family} / {cfg.task.name}")
        print(parameter_summary(experiment.model))
        return
    root = a.output or (str(Path(a.resume).resolve().parent.parent) if a.resume else f"runs/{a.run_name}")
    if a.reweight_folds:
        from twistformer.api.reweighting import fit_reweighter

        manifest_path = fit_reweighter(cfg, root, folds=a.reweight_folds, reference_class=a.reference_class)
        print(f"Reweighting manifest: {manifest_path}")
        return
    command = list(sys.orig_argv) if argv is None else [sys.executable, "-m", "twistformer.cli.train", *argv]
    best = Trainer(root, resume=a.resume, init_from=a.init_from, command=command).fit(experiment)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Best checkpoint: {best}")

if __name__ == "__main__":
    main()
