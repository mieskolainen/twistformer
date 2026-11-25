# Active training and evaluation wall time
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import datetime, timezone
import math


def interval_seconds(start, end):
    """
    Measure a recorded UTC interval without negative durations

    Args:
        start:  ISO start timestamp
        end:  ISO end timestamp
    Returns:
        seconds: Nonnegative duration or None for missing timestamps
    """
    try:
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if start.tzinfo is None or end.tzinfo is None:
            return None
        return max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        return None


def duration_seconds(record, active=False, now=None):
    """
    Read a saved duration and extend only an active interval

    Args:
        record:  Job or training attempt metadata
        active:  Extend the latest timing sample to the current time
        now:  Optional ISO timestamp for the current time
    Returns:
        seconds: Elapsed wall time or None when timing is unavailable
    """
    now = now or datetime.now(timezone.utc).isoformat()
    if "elapsed_seconds" in record:
        elapsed = record["elapsed_seconds"]
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            return None
        if active:
            elapsed += interval_seconds(record.get("elapsed_updated_at"), now) or 0.0
        return elapsed
    return interval_seconds(record.get("started_at"), now if active else record.get("finished_at"))


def training_seconds(record, active=False, now=None, stopped_at=None):
    """
    Sum training attempts without counting pauses between continuations

    Args:
        record:  Experiment metadata
        active:  The latest attempt is running
        now:  Optional current ISO timestamp
        stopped_at:  Last known training activity for an unfinished old attempt
    Returns:
        seconds: Total active wall time or None when an interval is unknown
    """
    if "elapsed_seconds" in record:
        return duration_seconds(record, active, now)
    attempts = record.get("attempts") or []
    if not attempts:
        return None
    total = 0.0
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            return None
        latest = index == len(attempts) - 1
        if latest and not active and not attempt.get("finished_at"):
            attempt = {**attempt, "finished_at": stopped_at or record.get("updated_at")}
        elapsed = duration_seconds(attempt, latest and active, now)
        if elapsed is None:
            return None
        total += elapsed
    return total


def format_duration(seconds):
    """
    Format elapsed wall time with hours that can exceed one day

    Args:
        seconds:  Nonnegative duration or None
    Returns:
        text: Hours, minutes and seconds or an unavailable marker
    """
    if seconds is None:
        return "unavailable"
    hours, remainder = divmod(int(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
