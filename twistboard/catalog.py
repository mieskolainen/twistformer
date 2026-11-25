# File index for training observables and parameter scans
#
# mikael.mieskolainen@cern.ch (2026)

import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import threading
import time
from .events import active_events
from .status import read_record, read_training_record, training_activity, training_log_tail, training_status


def has_run_record(directory):
    """
    Identify an experiment directory by its training record

    Args:
        directory:  Candidate output directory
    Returns:
        managed: True when the directory contains an experiment record
    """
    file = directory / "experiment.json"
    return file.is_file() and not file.is_symlink()


def discover_outputs(root):
    """
    Find experiments and external evaluations in the output tree

    Args:
        root:  Resolved dashboard directory
    Returns:
        run_ids: Experiment paths relative to root
        evaluations: Evaluation record files outside experiment directories
    """
    run_ids, evaluations = set(), []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        path = Path(directory)
        if path == root:
            continue
        if "experiment.json" in files and has_run_record(path):
            run_ids.add(path.relative_to(root).as_posix())
            subdirectories.clear()
        elif "evaluation.json" in files:
            evaluations.append(path / "evaluation.json")
    return run_ids, evaluations


def finite(value):
    """
    Identify finite numeric observations

    Args:
        value:  Candidate observation
    Returns:
        valid: True for finite numbers other than Boolean values
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten(value, prefix=""):
    """
    Flatten configuration leaves into parameter names

    Args:
        value:  Configuration mapping
        prefix:  Parent parameter path
    Returns:
        leaves: Scalar parameter mapping
    """
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif item is None or isinstance(item, (str, int, float, bool)):
            if not isinstance(item, float) or math.isfinite(item):
                result[name] = item
    return result


def signature(path):
    """
    Identify replacement, append and truncation of a local file

    Args:
        path:  File path
    Returns:
        stamp: File identity, byte count and modification time
    """
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def observe(stats, name, value, step=None):
    """
    Update one scalar summary without retaining its history

    Args:
        stats:  Mutable metric summary mapping
        name:  Metric name
        value:  Numeric observation or gap
        step:  Optimizer step
    """
    item = stats.setdefault(name, {"latest": None, "min": None, "max": None, "count": 0, "missing": 0})
    item["latest"] = value if finite(value) else None
    item["step"] = step
    if not finite(value):
        item["missing"] += 1
        return
    item["count"] += 1
    item["min"] = value if item["min"] is None else min(item["min"], value)
    item["max"] = value if item["max"] is None else max(item["max"], value)


def aggregate_trials(rows, metric, reduction, keys, kind):
    """
    Aggregate all filtered trials before any browser sampling

    Args:
        rows:  Filtered run summaries
        metric:  Objective metric
        reduction:  Latest, minimum or maximum
        keys:  Plot parameter names
        kind:  Heatmap or distribution
    Returns:
        groups: Counts, moments and quantiles for each parameter bin
    """
    if kind not in {"heatmap", "distribution"} or not keys:
        return None
    keys = keys[:2] if kind == "heatmap" else keys[:1]
    groups = {}
    domains = [set() for key in keys]
    for row in rows:
        value = row["metrics"].get(metric, {}).get(reduction)
        coordinates = tuple(row["parameters"].get(key) for key in keys)
        if not finite(value) or any(item is None for item in coordinates):
            continue
        for domain, coordinate in zip(domains, coordinates):
            domain.add(coordinate)
            if len(domain) > 40:
                return {"error": "Select parameters with at most 40 distinct values for this plot", "groups": []}
        groups.setdefault(coordinates, []).append(value)
    result = []
    for coordinates, values in groups.items():
        values.sort()
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / max(1, count - 1)
        def quantile(fraction):
            """
            Interpolate a sample quantile between adjacent observations

            Args:
                fraction:  Cumulative sample fraction
            Returns:
                value: Interpolated quantile
            """
            coordinate = fraction * (count - 1)
            lower = math.floor(coordinate)
            upper = math.ceil(coordinate)
            return values[lower] + (values[upper] - values[lower]) * (coordinate - lower)

        result.append({"x": coordinates[0], "y": coordinates[1] if len(coordinates) > 1 else None, "count": count,
                       "mean": mean, "sd": math.sqrt(variance) if count > 1 else None, "minimum": values[0], "maximum": values[-1],
                       "q1": quantile(0.25), "median": quantile(0.5), "q3": quantile(0.75)})
    return {"groups": result, "keys": keys}


class Entry:
    """Retain metadata and scalar summaries for one run"""

    def __init__(self, path, run_id):
        """
        Initialize an empty run index

        Args:
            path:  Run directory
            run_id:  Run path relative to the dashboard root
        """
        self.path = path
        self.run_id = run_id
        self.stamps = {}
        self.offsets = {}
        self.tails = {}
        self.stats = {"metrics.csv": {}, "events.jsonl": {}}
        self.record = {}
        self.status = None
        self.log_tail = ""
        self.headers = []
        self.epochs = 0
        self.latest = {}
        self.texts = []
        self.budget_cache = {}

    def at_step(self, metric, step):
        """
        Read a metric at one exact optimizer budget and cache the result

        Args:
            metric:  Observable name
            step:  Exact optimizer step
        Returns:
            value: Recorded value or None if the metric was not logged at this step
        """
        key = (metric, step, self.stamps.get("events.jsonl"), self.stamps.get("metrics.csv"))
        if key in self.budget_cache:
            return self.budget_cache[key]
        value = None
        if metric in self.stats["events.jsonl"] and not (self.path / "events.jsonl").is_symlink():
            for event in active_events(self.path / "events.jsonl"):
                if event.get("type") == "scalar" and event.get("step") == step and metric in event.get("values", {}):
                    value = event["values"][metric]
        elif metric in self.stats["metrics.csv"] and not (self.path / "metrics.csv").is_symlink():
            with (self.path / "metrics.csv").open() as stream:
                for row in csv.DictReader(stream):
                    try:
                        if float(row["step"]) == step:
                            value = float(row[metric])
                    except (TypeError, ValueError, KeyError):
                        continue
        value = value if finite(value) else None
        if len(self.budget_cache) >= 8:
            self.budget_cache.pop(next(iter(self.budget_cache)))
        self.budget_cache[key] = value
        return value

    def update(self):
        """
        Read changed metadata and complete appended metric records

        Returns:
            changed: True when an indexed file changed
        """
        changed = False
        rebuild_events = False
        for filename in ("experiment.json", "group.json", "metrics.csv", "events.jsonl"):
            path = self.path / filename
            if path.is_symlink():
                continue
            stamp = signature(path)
            previous = self.stamps.get(filename)
            if stamp == previous:
                continue
            changed = True
            self.stamps[filename] = stamp
            if filename.endswith(".json"):
                if filename == "experiment.json":
                    self.record = read_record(path)
                else:
                    self.group = read_record(path)
                continue
            reset = previous is None or stamp is None or stamp[0] != previous[0] or stamp[1] <= previous[1]
            offset = self.offsets.get(filename, 0)
            if not reset and offset:
                with path.open("rb") as stream:
                    stream.seek(max(0, offset - 128))
                    reset = stream.read(min(128, offset)) != self.tails.get(filename)
            if reset:
                self.offsets[filename] = 0
                self.stats[filename] = {}
                if filename == "metrics.csv":
                    self.headers, self.latest, self.epochs = [], {}, 0
                else:
                    self.texts = []
                    self.event_epochs = 0
            if stamp is None:
                continue
            with path.open("rb") as stream:
                stream.seek(self.offsets.get(filename, 0))
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offsets[filename] = stream.tell()
                    try:
                        if filename == "metrics.csv":
                            fields = next(csv.reader([line.decode()]))
                            if not self.headers:
                                self.headers = fields
                                continue
                            row = {key: float(value) for key, value in zip(self.headers, fields)}
                            self.latest = {key: value for key, value in row.items() if finite(value)}
                            self.epochs += 1
                            for name, value in row.items():
                                if name not in {"epoch", "step"}:
                                    observe(self.stats[filename], name, value, row.get("step"))
                        else:
                            event = json.loads(line)
                            if event.get("type") == "scalar":
                                for name, value in event.get("values", {}).items():
                                    observe(self.stats[filename], name, value, event.get("step"))
                                if finite(event.get("epoch")):
                                    self.event_epochs = event["epoch"]
                            elif event.get("type") == "text":
                                self.texts.append(event)
                                self.texts = self.texts[-100:]
                            elif event.get("type") == "session":
                                rebuild_events = True
                    except (ValueError, TypeError, AttributeError, UnicodeError):
                        continue
                offset = self.offsets.get(filename, 0)
                stream.seek(max(0, offset - 128))
                self.tails[filename] = stream.read(min(128, offset))
        if rebuild_events:
            self.stats["events.jsonl"], self.texts, self.event_epochs = {}, [], 0
            for event in active_events(self.path / "events.jsonl"):
                if event.get("type") == "scalar":
                    for name, value in event.get("values", {}).items():
                        observe(self.stats["events.jsonl"], name, value, event.get("step"))
                    if finite(event.get("epoch")):
                        self.event_epochs = event["epoch"]
                elif event.get("type") == "text":
                    self.texts.append(event)
                    self.texts = self.texts[-100:]
        status = self.record.get("status", "unknown")
        if status == "training":
            record = read_training_record(self.path, self.record)
            stamp = signature(self.path / "logs/train.log")
            if stamp != self.stamps.get("logs/train.log"):
                self.log_tail = training_log_tail(self.path)
                self.stamps["logs/train.log"] = stamp
            status = training_status(self.path, record, self.log_tail, training_activity(self.path, record))
        status_changed = status != self.status
        self.status = status
        return changed or status_changed

    def summary(self):
        """
        Build one lightweight run summary

        Returns:
            row: Run identity, parameters and scalar summaries
        """
        record = self.record
        group = getattr(self, "group", {})
        attempts = record.get("attempts") or []
        config = record.get("configuration") or {}
        params = flatten(config)
        params.update(group.get("parameters", {}))
        return {
            **{key: record.get(key) for key in ("created_at", "updated_at", "driver", "model", "task")},
            "id": self.run_id,
            "status": self.status,
            "driver": record.get("driver") or "Unknown driver",
            "group_id": group.get("group_id") or record.get("group_id") or "Ungrouped",
            "scan_id": group.get("scan_id") or record.get("scan_id"),
            "trial_id": group.get("trial_id") or record.get("trial_id"),
            "synthetic": bool(record.get("synthetic")),
            "training_started_at": next((attempt["started_at"] for attempt in reversed(attempts) if attempt.get("started_at")), record.get("created_at")),
            "epochs": self.epochs or getattr(self, "event_epochs", 0),
            "latest": self.latest,
            "elapsed_seconds": record.get("elapsed_seconds"),
            "parameters": params,
            "metrics": self.stats["metrics.csv"] | self.stats["events.jsonl"],
        }


class RunIndex:
    """Index run files without a database or full-history refreshes"""

    def __init__(self, root):
        """
        Bind an index to one output directory

        Args:
            root:  Experiment root
        """
        self.root = Path(root).resolve()
        self.entries = {}
        self.rows = []
        self.revision = 0
        self.checked = 0
        self.discovered = 0
        self.root_stamp = None
        self.lock = threading.RLock()
        self.curve_cache = {}

    def refresh(self, force=False):
        """
        Reconcile run directories and read only changed files

        Args:
            force:  Bypass the two-second refresh interval
        Returns:
            rows: Cached run summaries
        """
        with self.lock:
            now = time.monotonic()
            if not force and now - self.checked < 2:
                return self.rows
            changed = False
            root_stamp = signature(self.root)
            reconcile = force or now - self.discovered >= 10 or root_stamp != self.root_stamp
            if reconcile:
                found, _ = discover_outputs(self.root)
                for name in sorted(found):
                    entry = self.entries.setdefault(name, Entry(self.root / name, name))
                    try:
                        changed = entry.update() or changed
                    except OSError:
                        continue
                for name in self.entries.keys() - found:
                    del self.entries[name]
                    changed = True
                self.discovered = now
                self.root_stamp = root_stamp
            else:
                for entry in self.entries.values():
                    if entry.record.get("status") in {"training", "queued", "evaluating"}:
                        try:
                            changed = entry.update() or changed
                        except OSError:
                            continue
            if changed:
                self.rows = sorted((entry.summary() for entry in self.entries.values()), key=lambda row: row.get("created_at") or "", reverse=True)
                self.revision += 1
            self.checked = now
            return self.rows

    def query(self, options):
        """
        Filter summaries and return a bounded trial page or plot projection

        Args:
            options:  Query values from the dashboard
        Returns:
            result: Counts, available axes, branches and selected trial rows
        """
        with self.lock:
            rows = self.refresh()
            search = options.get("q", "").lower()
            rows = [row for row in rows if all(not options.get(key) or str(row.get(key)) == options[key] for key in ("id", "driver", "group_id", "scan_id", "status")) and (not search or search in " ".join(str(row.get(key) or "") for key in ("id", "driver", "model", "group_id", "scan_id")).lower())]
            names = sorted({name for row in rows for name in row["metrics"]})
            parameters = {}
            for row in rows:
                for key, value in row["parameters"].items():
                    values = parameters.setdefault(key, set())
                    if len(values) < 101:
                        values.add(value)
            axes = {key: sorted(values, key=str) for key, values in parameters.items() if len(values) > 1}
            metric = options.get("metric") or ("val_loss" if "val_loss" in names else next(iter(names), ""))
            mode = options.get("mode", "min")
            if mode not in {"min", "max"}:
                raise ValueError("Metric direction must be min or max")
            reduction = mode if options.get("reduce") == "best" else "latest"
            if options.get("reduce") == "budget":
                budget = int(options.get("budget", 0))
                if budget < 0:
                    raise ValueError("Optimizer budget must be nonnegative")
                reduction = "budget"
                rows = [{**row, "metrics": {**row["metrics"], metric: {**row["metrics"].get(metric, {}), "budget": self.entries[row["id"]].at_step(metric, budget)}}} for row in rows]
            filters = json.loads(options.get("filters", "{}"))
            if not isinstance(filters, dict):
                raise ValueError("Filters must be a mapping")
            for constraint in filters.values():
                if isinstance(constraint, dict):
                    if any(key not in {"min", "max"} or not finite(value) for key, value in constraint.items()):
                        raise ValueError("Range filters require finite minimum and maximum values")
                elif not isinstance(constraint, list) or any(isinstance(value, (dict, list)) for value in constraint):
                    raise ValueError("Category filters require lists of scalar values")
            def accepts(row):
                """
                Apply numeric ranges or categorical membership to one trial

                Args:
                    row:  Trial summary
                Returns:
                    accepted: True when all selected ranges match
                """
                for key, constraint in filters.items():
                    value = row["metrics"].get(metric, {}).get(reduction) if key == "$metric" else row["parameters"].get(key)
                    if isinstance(constraint, dict):
                        if not finite(value) or value < constraint.get("min", -math.inf) or value > constraint.get("max", math.inf):
                            return False
                    elif isinstance(constraint, list) and value not in constraint:
                        return False
                return True
            rows = [row for row in rows if accepts(row)]
            branches = {}
            for row in rows:
                key = (row["driver"], row["group_id"])
                branch = branches.setdefault(key, {"driver": key[0], "group_id": key[1], "count": 0, "statuses": {}})
                branch["count"] += 1
                status = row.get("status") or "unknown"
                branch["statuses"][status] = branch["statuses"].get(status, 0) + 1
            def score(row):
                """
                Rank finite objective values before missing measurements

                Args:
                    row:  Trial summary
                Returns:
                    key: Stable metric order
                """
                value = row["metrics"].get(metric, {}).get(reduction)
                return (not finite(value), value * (-1 if mode == "max" else 1) if finite(value) else 0, row["id"])
            if options.get("sort") == "metric":
                rows = sorted(rows, key=score)
            total = len(rows)
            offset = max(0, int(options.get("offset", 0)))
            limit = min(200, max(1, int(options.get("limit", 50))))
            requested = [key for key in options.get("axes", "").split(",") if key][:10]
            points = None
            if options.get("plot") == "1":
                requested = requested or list(axes)[:5]
                # Deterministic bounded sampling keeps browser transfer independent of scan size
                stride = max(1, math.ceil(total / 10000))
                points = [{"id": row["id"], "status": row["status"], "value": row["metrics"].get(metric, {}).get(reduction), "parameters": {key: row["parameters"].get(key) for key in requested}} for row in rows[::stride]]
            page = [] if options.get("tree") == "1" else rows[offset:offset + limit]
            return {"revision": self.revision, "total": total, "offset": offset, "rows": page, "branches": list(branches.values()), "metrics": names, "parameters": axes, "metric": metric, "points": points, "aggregates": aggregate_trials(rows, metric, reduction, requested, options.get("kind")), "sampled": points is not None and len(points) < total}

    def curves(self, run_ids, metric, axis="step", budget=600):
        """
        Load and reduce histories for a small explicit run selection

        Args:
            run_ids:  At most twelve run identifiers
            metric:  Scalar name
            axis:  Step, epoch or cumulative active time
            budget:  Maximum displayed points per run
        Returns:
            curves: Reduced histories and recent text events
        """
        if len(run_ids) > 12 or axis not in {"step", "epoch", "elapsed_seconds"}:
            raise ValueError("Select at most twelve runs and a valid horizontal axis")
        budget = max(20, min(2000, int(budget)))
        with self.lock:
            self.refresh()
            result = []
            for name in run_ids:
                if name not in self.entries:
                    continue
                entry = self.entries[name]
                key = (name, metric, axis, budget, signature(entry.path / "events.jsonl"), signature(entry.path / "metrics.csv"))
                cached = self.curve_cache.get(key)
                if cached is not None:
                    result.append(cached)
                    continue
                points = []
                events = entry.path / "events.jsonl"
                if events.is_file() and not events.is_symlink():
                    for event in active_events(events):
                        x = event.get(axis)
                        if event.get("type") == "scalar" and metric in event.get("values", {}) and finite(x):
                            y = event["values"][metric]
                            points.append([x, y if finite(y) else None])
                if metric not in entry.stats["events.jsonl"]:
                    path = entry.path / "metrics.csv"
                    if path.is_file() and not path.is_symlink():
                        with path.open() as stream:
                            for row in csv.DictReader(stream):
                                try:
                                    x = float(row[axis]) + (1 if axis == "epoch" else 0)
                                    y = float(row[metric])
                                    if finite(x):
                                        points.append([x, y if finite(y) else None])
                                except (ValueError, KeyError, TypeError):
                                    continue
                item = {"id": name, "points": downsample(points, budget), "count": len(points), "texts": entry.texts}
                if len(self.curve_cache) >= 64:
                    self.curve_cache.pop(next(iter(self.curve_cache)))
                self.curve_cache[key] = item
                result.append(item)
            return result


def downsample(points, budget):
    """
    Preserve extrema and missing-value markers within contiguous bins

    Args:
        points:  Ordered coordinate pairs
        budget:  Maximum output size
    Returns:
        reduced: Bounded plot coordinates
    """
    if len(points) <= budget:
        return points
    width = math.ceil(len(points) / max(1, (budget - 2) // 3))
    selected = {0, len(points) - 1}
    for start in range(0, len(points), width):
        indices = range(start, min(start + width, len(points)))
        valid = [i for i in indices if finite(points[i][1])]
        if valid:
            selected.add(min(valid, key=lambda i: points[i][1]))
            selected.add(max(valid, key=lambda i: points[i][1]))
        missing = next((i for i in indices if points[i][1] is None), None)
        if missing is not None:
            selected.add(missing)
    return [points[i] for i in sorted(selected)]


@lru_cache(maxsize=4)
def get_index(root):
    """
    Reuse one index across HTTP requests for an output directory

    Args:
        root:  Resolved root path
    Returns:
        index: Shared file index
    """
    return RunIndex(root)
