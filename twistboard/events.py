# Named training observables and event records
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import Lock


class EventLogger:
    """Append observables from one process that writes output files"""

    def __init__(self, root, attempt_id=None, elapsed=None):
        """
        Bind a logger to one run

        Args:
            root:  Run directory
            attempt_id:  Training attempt identifier
            elapsed:  Optional cumulative time callback
        """
        self.path = Path(root) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempt_id = attempt_id
        self.elapsed = elapsed
        self.lock = Lock()
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as stream:
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")

    def _write(self, kind, values, step, epoch=None):
        """
        Append one complete JSON line

        Args:
            kind:  Event type
            values:  Named JSON values
            step:  Global optimizer step
            epoch:  Optional completed epoch coordinate
        """
        if type(step) is not int or step < 0:
            raise ValueError("Step must be a nonnegative integer")
        if epoch is not None and (not math.isfinite(epoch) or epoch < 0):
            raise ValueError("Epoch must be finite and nonnegative")
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError("Event names must be nonempty strings")
        record = {
            "version": 1, "type": kind, "values": values, "step": step,
            "epoch": epoch, "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": self.elapsed() if self.elapsed else None,
            "attempt_id": self.attempt_id,
        }
        line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    def log_scalars(self, values, step, epoch=None):
        """
        Record arbitrary numeric observables, with null gaps for nonfinite values

        Args:
            values:  Mapping from metric names to numeric values
            step:  Global optimizer step
            epoch:  Optional completed epoch coordinate
        """
        values = {name: float(value) for name, value in values.items()}
        self._write("scalar", {name: value if math.isfinite(value) else None for name, value in values.items()}, step, epoch)

    def log_text(self, name, value, step, epoch=None):
        """
        Record a named text or Boolean value

        Args:
            name:  Event name
            value:  Text or Boolean value
            step:  Global optimizer step
            epoch:  Optional completed epoch coordinate
        """
        if not isinstance(value, (str, bool)):
            raise ValueError("Text events accept strings or Boolean values")
        self._write("text", {name: value}, step, epoch)

    def start_attempt(self, step):
        """
        Mark the restored optimizer step without deleting previous records

        Args:
            step:  Restored optimizer step, or zero for a new training run
        """
        self._write("session", {"resume_step": step}, step)


def active_events(path):
    """
    Read complete records on the active continuation branch

    Args:
        path:  Event file
    Returns:
        events: Iterator excluding observations abandoned by later resumes
    """
    sessions = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream):
            if not line.endswith("\n"):
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "session":
                    step = event.get("values", {}).get("resume_step")
                    if isinstance(step, int) and step >= 0:
                        sessions.append((line_number, step))
            except (ValueError, TypeError, AttributeError):
                continue
    boundaries = []
    ceiling = math.inf
    for line_number, step in reversed(sessions):
        ceiling = min(ceiling, step)
        boundaries.append((line_number, ceiling))
    boundaries.reverse()
    segment = 0
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream):
            while segment < len(boundaries) and line_number >= boundaries[segment][0]:
                segment += 1
            ceiling = boundaries[segment][1] if segment < len(boundaries) else math.inf
            if not line.endswith("\n"):
                continue
            try:
                event = json.loads(line)
                if event.get("type") != "session" and event.get("step", 0) <= ceiling:
                    yield event
            except (ValueError, TypeError, AttributeError):
                continue
