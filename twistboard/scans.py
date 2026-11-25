# Parameter scans and synthetic dashboard fixtures
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import datetime, timedelta, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
from uuid import uuid4
from .events import EventLogger
from .outputs import write_json
from .commands import run_command


def run_scan(card, root="runs", execute=False):
    """
    Expand a grid into recorded trials and optionally train each trial

    Args:
        card:  YAML scan card path
        root:  Experiment root
        execute:  Launch sequential training when true
    Returns:
        manifest: Scan manifest path
    """
    import yaml
    from twistformer.api.config import load_config

    card = Path(card).resolve()
    spec = yaml.safe_load(card.read_text())
    base = (card.parent / spec["config"]).resolve()
    grid = spec.get("parameters", {})
    if not isinstance(grid, dict) or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Scan parameters must contain nonempty lists")
    seeds = spec.get("seeds", [101])
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("Scan seeds must be a nonempty integer list")
    if "training.seed" in grid:
        raise ValueError("Specify scan seeds in seeds only")
    count = math.prod(len(values) for values in grid.values()) * len(seeds)
    if count > 10000:
        raise ValueError("One scan supports at most 10000 trials")
    name = str(spec.get("name", card.stem))
    scan_id = "scan-" + uuid4().hex[:10]
    root = Path(root).resolve()
    trials = []
    for values in itertools.product(*grid.values(), seeds):
        parameters = dict(zip([*grid, "training.seed"], values))
        overrides = [f"{key}={json.dumps(value)}" for key, value in parameters.items()]
        resolved = load_config(base, overrides).to_dict()
        trial_id = f"{len(trials) + 1:05d}"
        directory = root / f"{scan_id}-{trial_id}"
        command = [sys.executable, "-u", "-m", "twistformer.cli.train", "--config", str(base), "--output", str(directory)]
        for override in overrides:
            command += ["--set", override]
        trials.append({"id": trial_id, "directory": str(directory), "parameters": parameters, "configuration": resolved, "command": command, "status": "planned"})
    manifest = root / ".scans" / f"{scan_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "id": scan_id, "name": name, "card": str(card), "objective": spec.get("objective", {"metric": "val_loss", "mode": "min"}), "trials": trials}
    write_json(manifest, record)
    if execute:
        for trial in trials:
            group = {"group_id": name, "scan_id": scan_id, "trial_id": trial["id"], "parameters": trial["parameters"]}
            env = dict(os.environ, TWISTBOARD_GROUP=json.dumps(group))
            trial["status"] = "training"
            write_json(manifest, record)
            try:
                result = run_command(trial["command"], env=env, check=False)
                trial.update(status="trained" if result.returncode == 0 else "failed", returncode=result.returncode)
            except KeyboardInterrupt:
                trial["status"] = "interrupted"
                write_json(manifest, record)
                raise
            write_json(manifest, record)
    return manifest


def synthetic_scans(root="runs", trials=576, epochs=40, seed=19):
    """
    Generate reproducible scan observables without physics data or model weights

    Args:
        root:  Experiment root
        trials:  Total number of synthetic experiments
        epochs:  Epochs per successful trial
        seed:  Fixture random seed
    Returns:
        manifest: Manifest containing only the created run directories
    """
    if trials < 1 or epochs < 1:
        raise ValueError("Trials and epochs must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex[:8]
    rng = random.Random(seed)
    started = datetime.now(timezone.utc) - timedelta(days=1)
    paths = []
    drivers = ["tau3_classification", "toptag", "tau3_regression"]
    grid = list(itertools.product([0.00003, 0.0001, 0.0003, 0.001], [32, 64, 128], [2, 4], ["adamw", "lion"], [1, 2, 3, 4]))
    for index in range(trials):
        driver = drivers[(index // len(grid)) % len(drivers)]
        lr, width, depth, optimizer, run_seed = grid[index % len(grid)]
        scan_id = f"synthetic-{token}-{driver}"
        name = f"{scan_id}-{index + 1:05d}"
        path = root / name
        path.mkdir()
        paths.append(name)
        created = started + timedelta(seconds=index)
        config = {"driver": {"name": driver}, "model": {"family": "gatr_conformal" if driver == "tau3_classification" else "transformer", "width": width, "depth": depth}, "task": {"name": "regression" if driver == "tau3_regression" else "classification"}, "training": {"learning_rate": lr, "optimizer": optimizer, "seed": run_seed, "epochs": epochs}}
        failed = index % 29 == 0
        stopped = index % 31 == 0 and not failed
        length = max(1, epochs // 3) if failed or stopped else epochs
        elapsed = 0.0
        logger = EventLogger(path, "synthetic-1", lambda: elapsed)
        floor = 0.20 + 0.055 * abs(math.log10(lr / 0.0001)) + 1.4 / width + (0.025 if optimizer == "lion" else 0) + 0.01 * (4 - depth)
        latest = None
        for epoch in range(1, length + 1):
            elapsed += width * depth * 0.035 + rng.uniform(0.5, 1.5)
            loss = floor + 0.8 * math.exp(-epoch / max(3, epochs / 5)) + rng.gauss(0, 0.009)
            if failed and epoch > length * 0.7:
                loss += (epoch / length) ** 4
            latest = loss
            values = {"train_loss": loss * 0.86, "val_loss": loss, "lr": lr * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / epochs)) / 2), "diagnostics/grad_norm": abs(rng.gauss(0.6, 0.12)) + (3 if failed else 0), "physics/response": 1 - loss * 0.2}
            if epoch > epochs // 2:
                values["validation/late_metric"] = 1 / (1 + loss)
            if index % 47 == 0 and epoch == length // 2:
                values["diagnostics/grad_norm"] = float("nan")
            logger.log_scalars(values, step=epoch * 100, epoch=epoch)
        logger.log_text("fixture", "Synthetic scan for dashboard testing. No model was trained", step=length * 100)
        logger.log_text("converged", not failed and not stopped, step=length * 100)
        finished = created + timedelta(seconds=elapsed)
        record = {"schema_version": 1, "id": name, "synthetic": True, "status": "failed" if failed else "interrupted" if stopped else "trained", "created_at": created.isoformat(), "updated_at": finished.isoformat(), "driver": driver, "model": config["model"]["family"], "task": config["task"]["name"], "configuration": config, "elapsed_seconds": elapsed, "attempts": [{"started_at": created.isoformat(), "finished_at": finished.isoformat(), "elapsed_seconds": elapsed, "command": f"python -m twistboard synthetic --root {root} --trials {trials} --epochs {epochs} --seed {seed}"}]}
        write_json(path / "experiment.json", record)
        write_json(path / "group.json", {"group_id": f"Synthetic {driver} scan {token}", "scan_id": scan_id, "trial_id": index + 1})
    manifest = root / ".scans" / f"synthetic-{token}.json"
    manifest.parent.mkdir(exist_ok=True)
    write_json(manifest, {"synthetic": True, "seed": seed, "trials": paths, "epochs": epochs})
    return manifest
