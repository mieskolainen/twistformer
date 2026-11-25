# Dashboard checkpoint jobs and process records
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from .experiments import timestamp
from .outputs import write_json

def checkpoint_path(root, value):
    """
    Resolve an experiment checkpoint inside the dashboard root

    Args:
        root:  Resolved output directory
        value:  Browser checkpoint path
    Returns:
        path: Valid checkpoint file
    """
    from .server import contained

    if not isinstance(value, str):
        raise ValueError("Invalid checkpoint path")
    path = contained(root, value)
    if (
        path.suffix != ".pt"
        or not path.is_file()
        or path.parent.name != "checkpoints"
        or not (path.parent.parent / "experiment.json").is_file()
    ):
        raise ValueError("Select a checkpoint from an experiment")
    return path

def launch_job(argv, output, log, record, kind, **options):
    """
    Launch a host job and retain startup failures in its record

    Args:
        argv:  Python module arguments
        output:  Job directory
        log:  Host log path
        record:  Initial job metadata
        kind:  Evaluation or comparison record stem
        options:  Additional subprocess options
    Returns:
        process: Running Python subprocess
    """
    try:
        with log.open("w") as stream:
            return subprocess.Popen([sys.executable, *argv], stdout=stream, stderr=subprocess.STDOUT, cwd=Path.cwd(), **options)
    except OSError as error:
        record.update(status="failed", error=str(error), finished_at=timestamp())
        write_json(output / f"{kind}.json", record)
        raise

def finish_job(process, output, log, *, kind, lock=None):
    """
    Retain process logs and mark jobs that exit before completion

    Args:
        process:  Host subprocess
        output:  Job directory
        log:  Host log path
        kind:  Evaluation or comparison record stem
        lock:  Optional lock around output updates
    """
    code = process.wait()
    destination = output / ("evaluate.log" if kind == "evaluation" else "compare.log")
    with (lock if lock is not None else nullcontext()):
        try:
            record = json.loads((output / f"{kind}.json").read_text())
            if kind == "evaluation":
                shutil.copyfile(log, destination)
            if record.get("status") not in {"completed", "failed", "interrupted"}:
                with log.open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 2000))
                    message = stream.read().decode(errors="replace")
                record.update(
                    status="failed", error=f"{kind.title()} process exited with code {code}\n{message}", finished_at=timestamp()
                )
                write_json(output / f"{kind}.json", record)
            if kind == "comparison":
                shutil.copyfile(log, destination)
        except FileNotFoundError:
            if kind != "evaluation":
                raise
