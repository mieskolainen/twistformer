# Checkpoint evaluation jobs on the dashboard host
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import signal
from .jobs import checkpoint_path, launch_job, finish_job, subprocess, threading
from .experiments import timestamp, unique_directory
from .outputs import write_json

_lock = threading.Lock()
_jobs = {}
_finish_job = partial(finish_job, kind="evaluation", lock=_lock)


@contextmanager
def run_removal(root, run, *, force):
    """
    Hold the evaluation lock until the experiment directory is removed

    Args:
        root:  Dashboard output directory
        run:  Experiment directory selected for removal
        force:  Stop an evaluation launched by this dashboard for the selected run
    """
    with _lock:
        job = _jobs.get(str(root))
        if job is not None and job[0] == run and job[1].poll() is None:
            if not force:
                raise RuntimeError("Stop evaluation before deleting this directory")
            _stop_evaluation(job[1])
        yield


def _stop_evaluation(process):
    """
    Stop the owned evaluation process and its workers

    Args:
        process:  Subprocess started by this dashboard for the selected experiment
    """
    for kill in (False, True):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            elif kill:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("The dashboard evaluation process did not stop")

def start_evaluation(root, payload):
    """
    Evaluate a selected checkpoint on the host test sample

    Args:
        root:  Dashboard output directory
        payload:  Checkpoint path relative to the output directory
    Returns:
        job: Queued evaluation directory and status
    """
    from .server import contained

    root = Path(root).resolve()
    if not isinstance(payload, dict) or set(payload) != {"checkpoint"}:
        raise ValueError("Select a checkpoint to evaluate")
    with _lock:
        checkpoint = checkpoint_path(root, payload["checkpoint"])
        run = checkpoint.parent.parent
        active = _jobs.get(str(root))
        if active is not None and active[1].poll() is None:
            raise RuntimeError("An evaluation is already running on this dashboard host")
        parent = contained(root, (run / "evaluations").relative_to(root))
        logs = contained(root, (run / "logs").relative_to(root))
        logs.mkdir(parents=True, exist_ok=True)
        output = unique_directory(parent, "test-checkpoint")
        output.mkdir(parents=True)
        record = {
            "schema_version": 1,
            "status": "queued",
            "started_at": timestamp(),
            "checkpoint": str(checkpoint),
            "split": "test",
            "launched_by": "twistboard",
        }
        write_json(output / "evaluation.json", record)
        log = logs / f"{output.name}.log"
        process = launch_job(
            ["-u", "-m", "twistformer.cli.evaluate", "--checkpoint", str(checkpoint), "--split", "test", "--output", str(output)],
            output,
            log,
            record,
            "evaluation",
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        _jobs[str(root)] = (run, process)
        threading.Thread(target=_finish_job, args=(process, output, log), daemon=True).start()
    return {"id": output.relative_to(root).as_posix(), "status": "queued"}
