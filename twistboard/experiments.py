# Experiment lifecycle and file provenance
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import traceback
from uuid import uuid4
from .outputs import write_json, training_plots
from .status import INTERRUPTION_MESSAGE, training_activity
from .timing import format_duration, training_seconds

def timestamp():
    """
    Record UTC wall time

    Returns:
        time: ISO timestamp
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")

def unique_directory(root, name):
    """
    Generate an unused experiment or evaluation path

    Args:
        root:  Parent file directory
        name:  Human-readable experiment name
    Returns:
        path: Unique path without creating the directory
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Names must contain only letters, digits, dots, underscores and hyphens")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root).resolve() / f"{name}-{stamp}-{uuid4().hex[:8]}"

class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
        self.progress_line = False

    @property
    def encoding(self):
        """
        Expose terminal encoding for Unicode progress-bar detection

        Returns:
            encoding: Terminal character encoding, or None if unavailable
        """
        return getattr(self.terminal, "encoding", None)

    def write(self, text):
        """
        Copy text to the run log while keeping progress updates in the terminal

        Args:
            text:  Output fragment
        Returns:
            size: Number of input characters
        """
        if "\r" in text or text == "\x1b[A":
            if getattr(self.terminal, "isatty", lambda: False)():
                self.terminal.write(text)
            self.progress_line = True
            return len(text)
        if self.progress_line and text == "\n":
            if getattr(self.terminal, "isatty", lambda: False)():
                self.terminal.write(text)
            self.progress_line = False
            return len(text)
        self.progress_line = False
        self.terminal.write(text)
        self.log.write(re.sub(r"\x1b\[[0-9;]*m", "", text))
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

class TrainingTimer:
    def __init__(self, path, record, previous):
        """
        Start a monotonic timer for one training continuation

        Args:
            path:  Experiment metadata path
            record:  Current experiment and attempt metadata
            previous:  Active seconds from earlier attempts or None
        """
        self.path, self.record, self.previous = path, record, previous
        self.started = time.monotonic()
        self.saved = self.started

    def elapsed(self):
        """
        Return cumulative active seconds without offline intervals

        Returns:
            seconds: Total wall time or None for unknown earlier timing
        """
        return self.previous + time.monotonic() - self.started if self.previous is not None else None

    def inherit(self, seconds):
        """
        Restore checkpoint time for continuation into a new directory

        Args:
            seconds:  Cumulative time stored in the resumed checkpoint
        """
        if seconds is not None:
            self.previous = max(self.previous or 0.0, seconds)
        elif len(self.record["attempts"]) == 1:
            self.previous = None
        self.persist(force=True)

    def persist(self, force=False):
        """
        Save active time periodically and at training boundaries

        Args:
            force:  Write even when less than ten seconds have elapsed
        """
        now = time.monotonic()
        if not force and now - self.saved < 10:
            return
        session = now - self.started
        stamp = timestamp()
        self.record.update(
            elapsed_seconds=self.previous + session if self.previous is not None else None,
            elapsed_updated_at=stamp,
            updated_at=stamp,
        )
        self.record["attempts"][-1].update(elapsed_seconds=session, elapsed_updated_at=stamp)
        write_json(self.path, self.record)
        self.saved = now

@contextmanager
def training_run(root, config, resume=None, init_from=None, *, command=None, model_parameters=None):
    """
    Record training status, logs and plots for every public training path

    Args:
        root:  Experiment directory
        config:  Resolved configuration dictionary
        resume:  Optional checkpoint to resume
        init_from:  Optional checkpoint for weight initialization
        command:  Optional training process argument list
        model_parameters:  Total and trainable parameter counts
    Returns:
        timer: Active training timer with failure and interruption records
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError(f"Run directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".training.lock"
    try:
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError(f"Training lock exists: {lock}. Check the recorded process before removing a stale lock") from None
    record_path = root / "experiment.json"
    try:
        record = (
            json.loads(record_path.read_text())
            if record_path.exists()
            else {"schema_version": 1, "id": root.name, "created_at": timestamp(), "attempts": []}
        )
        previous = training_seconds(record, stopped_at=training_activity(root, record)) if record_path.exists() else 0.0
        attempt = {
            "started_at": timestamp(),
            "resume": str(resume) if resume else None,
            "init_from": str(init_from) if init_from else None,
            "command": shlex.join(command) if command else None,
            "argv": list(command) if command else None,
            "working_directory": str(Path.cwd()),
        }
        record["attempts"].append(attempt)
        if os.environ.get("TWISTBOARD_GROUP"):
            group = json.loads(os.environ["TWISTBOARD_GROUP"])
            if not isinstance(group, dict):
                raise ValueError("TWISTBOARD_GROUP must be a JSON mapping")
            write_json(root / "group.json", group)
        record.update(
            status="training",
            updated_at=timestamp(),
            driver=config["driver"]["name"],
            model=config["model"]["family"],
            task=config["task"]["name"],
            configuration=config,
        )
        if model_parameters is not None:
            record["model_parameters"] = model_parameters
        record.pop("error", None)
        timer = TrainingTimer(record_path, record, previous)
        timer.persist(force=True)
        (root / "logs").mkdir(exist_ok=True)
        with (root / "logs/train.log").open("a", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    yield timer
                except BaseException as error:
                    record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    if isinstance(error, KeyboardInterrupt):
                        print(INTERRUPTION_MESSAGE, flush=True)
                    else:
                        traceback.print_exc()
                    raise
                else:
                    record["status"] = "trained"
                finally:
                    attempt.update(finished_at=timestamp(), status=record["status"])
                    timer.persist(force=True)
                    print(f"Total training time: {format_duration(record['elapsed_seconds'])}", flush=True)
                    try:
                        if record["status"] != "interrupted":
                            training_plots(root)
                    except Exception as error:
                        record["plot_error"] = str(error)
                        print(f"Training plots could not be written: {error}", file=sys.stderr)
                    write_json(record_path, record)
    finally:
        lock.unlink(missing_ok=True)

@contextmanager
def evaluation_run(checkpoint, output, split, input_path=None):
    """
    Record an evaluation against the exact checkpoint file checksum

    Args:
        checkpoint:  Checkpoint selected for evaluation
        output:  Empty evaluation file directory
        split:  Dataset split
        input_path:  Optional driver input override
    Returns:
        stream: Open checkpoint snapshot matching the recorded checksum
    """
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        files = list(output.iterdir())
        reservation = output / "evaluation.json"
        queued = json.loads(reservation.read_text()) if files == [reservation] else {}
        if not (
            isinstance(queued, dict)
            and queued.get("status") == "queued"
            and queued.get("launched_by") == "twistboard"
            and queued.get("checkpoint") == str(checkpoint)
            and queued.get("split") == split
            and input_path is None
        ):
            raise ValueError(f"Evaluation directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = timestamp()
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        record = {
            "schema_version": 1,
            "status": "evaluating",
            "started_at": started_at,
            "checkpoint": str(checkpoint),
            "sha256": digest,
            "split": split,
            "input": input_path,
        }
        write_json(output / "evaluation.json", record)
        with (output / "evaluate.log").open("a", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    yield stream
                except BaseException as error:
                    record.update(
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                    traceback.print_exc()
                    raise
                else:
                    record["status"] = "completed"
                finally:
                    record["finished_at"] = timestamp()
                    record["elapsed_seconds"] = time.monotonic() - started
                    write_json(output / "evaluation.json", record)
                    print(f"Evaluation time: {format_duration(record['elapsed_seconds'])}", flush=True)
