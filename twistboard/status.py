# Training activity and lifecycle records
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import datetime, timezone
import json
from pathlib import Path
import time


STALE_SECONDS = 30 * 60
INTERRUPTION_MESSAGE = "Training interrupted. Completed epoch checkpoints are retained"


def training_activity(path, record):
    """
    Find the latest recorded training activity without evaluation timestamps

    Args:
        path:  Experiment directory
        record:  Saved training lifecycle record
    Returns:
        time: UTC timestamp or None when no training activity is known
    """
    saved = record.get("updated_at")
    try:
        updated = datetime.fromisoformat(saved)
        latest = updated.timestamp() if updated.tzinfo else None
    except (TypeError, ValueError):
        latest = None
    if record.get("status") in {"trained", "failed", "interrupted"} and latest is not None:
        return saved
    directory = path.resolve()
    for relative in ("experiment.json", "logs/train.log", "metrics.csv", "events.jsonl", "checkpoints/index.json"):
        file = path / relative
        try:
            if file.is_file() and file.resolve().is_relative_to(directory):
                modified = file.stat().st_mtime
                latest = max(latest, modified) if latest is not None else modified
        except OSError:
            continue
    return datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest is not None else None


def read_record(path):
    """
    Read an optional JSON file without blocking a live dashboard refresh

    Args:
        path:  Input file
    Returns:
        record: Parsed mapping, or an empty mapping for unavailable records
    """
    try:
        record = json.loads(Path(path).read_text())
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def read_training_record(path, record=None):
    """
    Recover a complete final record left before an atomic replacement

    Args:
        path:  Run directory
        record:  Indexed experiment metadata, if available
    Returns:
        record: Saved record or a newer final record for the same training attempt
    """
    if record is None:
        record = read_record(path / "experiment.json")
    if record.get("status") != "training" or (path / ".training.lock").exists():
        return record
    pending_file = path / "experiment.json.tmp"
    if pending_file.is_symlink():
        return record
    pending = read_record(pending_file)
    if pending.get("status") not in {"trained", "interrupted", "failed"}:
        return record
    # A final record must identify the same run and the same complete attempt history
    for key in ("schema_version", "id", "created_at", "configuration"):
        if not record.get(key) or pending.get(key) != record[key]:
            return record
    attempts, final_attempts = record.get("attempts"), pending.get("attempts")
    if (
        not isinstance(attempts, list)
        or not isinstance(final_attempts, list)
        or not attempts
        or len(attempts) != len(final_attempts)
        or attempts[:-1] != final_attempts[:-1]
        or not isinstance(attempts[-1], dict)
        or not isinstance(final_attempts[-1], dict)
    ):
        return record
    current, final = attempts[-1], final_attempts[-1]
    if (
        not current.get("started_at")
        or any(final.get(key) != value for key, value in current.items() if key not in {"elapsed_seconds", "elapsed_updated_at"})
        or final.get("status") != pending["status"]
    ):
        return record
    try:
        started = datetime.fromisoformat(current["started_at"])
        finished = datetime.fromisoformat(final["finished_at"])
        saved = datetime.fromisoformat(record["updated_at"])
        updated = datetime.fromisoformat(pending["updated_at"])
        if not all(value.tzinfo is not None for value in (started, finished, saved, updated)):
            return record
        if updated > saved and started <= finished <= updated:
            return pending
    except (KeyError, TypeError, ValueError):
        pass
    return record


def training_log_tail(path):
    """
    Read the last training messages for one run

    Args:
        path:  Run directory
    Returns:
        tail: Final log text or an empty string
    """
    file = path / "logs/train.log"
    try:
        if file.is_file() and file.resolve().is_relative_to(path.resolve()):
            with file.open("rb") as stream:
                stream.seek(max(0, file.stat().st_size - 16000))
                return stream.read(16000).decode(errors="replace")
    except OSError:
        pass
    return ""


def training_status(path, record, tail, activity):
    """
    Resolve a recorded training state from its final log or last activity

    Args:
        path:  Run directory
        record:  Experiment metadata for the current attempt
        tail:  Final training log text
        activity:  Latest training timestamp
    Returns:
        status: Displayed training state
    """
    status = record.get("status", "unknown")
    if status != "training":
        return status
    attempts = record.get("attempts") or []
    first_attempt = not attempts or (
        len(attempts) == 1 and isinstance(attempts[0], dict) and not attempts[0].get("resume")
    )
    if first_attempt and not (path / ".training.lock").exists():
        lines = tail.rstrip().splitlines()
        if lines and lines[-1].startswith("Total training time: "):
            lines.pop()
        last_line = lines[-1] if lines else ""
        traceback = last_line == "KeyboardInterrupt" and "Traceback (most recent call last):" in lines
        if last_line == INTERRUPTION_MESSAGE or traceback:
            return "interrupted"
    if activity and time.time() - datetime.fromisoformat(activity).timestamp() >= STALE_SECONDS:
        return "stale"
    return status
