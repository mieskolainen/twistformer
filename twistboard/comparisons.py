# Local model comparison jobs
#
# mikael.mieskolainen@cern.ch (2026)

import json
from pathlib import Path
from functools import partial
from .jobs import checkpoint_path, launch_job, finish_job, threading
from .outputs import write_json
from .experiments import timestamp, unique_directory

_lock = threading.Lock()
_jobs = {}
_finish_job = partial(finish_job, kind="comparison")

def start_comparison(root, payload):
    """
    Launch one local comparison using selected experiment checkpoints

    Args:
        root: Dashboard output directory
        payload: Selected labels and checkpoint paths relative to root
    Returns:
        job: New comparison ID and queued status
    """
    from .server import contained

    root = Path(root).resolve()
    if not isinstance(payload, dict) or set(payload) != {"models"}:
        raise ValueError("Select model checkpoints to compare")
    requested = payload["models"]
    if not isinstance(requested, list) or not 2 <= len(requested) <= 16:
        raise ValueError("Select between 2 and 16 checkpoints")
    models, labels, paths = [], set(), set()
    for entry in requested:
        if not isinstance(entry, dict) or set(entry) != {"label", "checkpoint"}:
            raise ValueError("Each model needs a label and checkpoint")
        label, value = entry["label"], entry["checkpoint"]
        if not isinstance(label, str) or not label.strip() or len(label) > 200 or label in labels:
            raise ValueError("Model labels must be nonempty and unique")
        path = checkpoint_path(root, value)
        if path in paths:
            raise ValueError("Select distinct checkpoints")
        labels.add(label)
        paths.add(path)
        models.append({"label": label, "checkpoint": str(path)})
    with _lock:
        active = _jobs.get(str(root))
        if active is not None and active.poll() is None:
            raise RuntimeError("A comparison is already running on this dashboard host")
        output = unique_directory(contained(root, "comparisons"), "models")
        output.mkdir(parents=True)
        steering = {
            "comparison": {
                "name": "model-comparison",
                "dataset": {"reference_checkpoint": models[0]["checkpoint"], "split": "test"},
                "models": models,
                "reference_model": models[0]["label"],
            }
        }
        cards = contained(root, "comparisons/cards")
        cards.mkdir(parents=True, exist_ok=True)
        card = cards / f"{output.name}.yml"
        card.write_text(json.dumps(steering, indent=2) + "\n")
        record = {
            "schema_version": 1,
            "name": "model-comparison",
            "status": "queued",
            "started_at": timestamp(),
            "models": [],
            "selections": [],
            "launched_by": "twistboard",
        }
        write_json(output / "config.json", steering)
        write_json(output / "comparison.json", record)
        logs = contained(root, "comparisons/logs")
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{output.name}.log"
        process = launch_job(
            ["-m", "twistformer.cli.compare", "--config", str(card), "--output", str(output)], output, log, record, "comparison"
        )
        _jobs[str(root)] = process
        threading.Thread(target=_finish_job, args=(process, output, log), daemon=True).start()
    return {"id": output.relative_to(root).as_posix(), "status": "queued"}
