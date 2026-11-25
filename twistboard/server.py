# Browser access and removal of experiment files
#
# mikael.mieskolainen@cern.ch (2026)

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import errno
import json
import mimetypes
import itertools
import os
import re
from pathlib import Path
import shutil
import stat
from urllib.parse import unquote, urlsplit, parse_qs
import webbrowser
from .outputs import read_history
from .timing import duration_seconds, training_seconds
from .catalog import discover_outputs, get_index, has_run_record
from .status import read_record, read_training_record, training_activity, training_log_tail, training_status


def deletion_block_reason(path, record, evaluations=()):
    """
    Check active work before enabling directory removal

    Args:
        path:  Run or evaluation directory
        record:  Record with the current status
        evaluations:  Evaluations associated with the run
    Returns:
        reason: Blocking message or None when no active work is recorded
    """
    if (path / ".training.lock").exists() or record.get("status") in {"training", "stale", "evaluating", "queued"}:
        return "Stop training or evaluation before deleting this directory"
    if any(item.get("status") in {"queued", "evaluating"} for item in evaluations) or any(
        read_record(metadata).get("status") in {"queued", "evaluating"} for metadata in path.rglob("evaluation.json")
    ):
        return "Stop evaluation before deleting this directory"
    return None

def contained(root, relative):
    """
    Resolve a file path inside the configured experiment directory

    Args:
        root:  Dashboard output directory
        relative:  Relative file path supplied by the browser
    Returns:
        path: Resolved path, rejecting hidden files and directory escape
    """
    relative = Path(relative)
    if relative.is_absolute() or any(part.startswith(".") for part in relative.parts):
        raise ValueError("Invalid file path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("File lies outside the experiment root")
    return path

def file_list(root, directory):
    """
    List downloadable files contained within the dashboard root

    Args:
        root:  Resolved dashboard root
        directory:  Run or evaluation directory
    Returns:
        files: File paths, sizes and modification times
    """
    allowed = {".png", ".pdf", ".json", ".jsonl", ".yml", ".csv", ".log", ".pt", ".npz"}
    files = []
    for folder, subdirectories, names in os.walk(directory):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        relative = Path(folder).relative_to(root)
        try:
            folder_path = contained(root, relative)
        except (OSError, ValueError):
            subdirectories.clear()
            continue
        for name in names:
            if name.startswith("."):
                continue
            try:
                file = folder_path / name
                info = file.lstat()
                if stat.S_ISLNK(info.st_mode):
                    file = contained(root, relative / name)
                    info = file.stat()
                if file.suffix in allowed and stat.S_ISREG(info.st_mode):
                    files.append({"path": (relative / name).as_posix(), "bytes": info.st_size, "modified": info.st_mtime})
            except (OSError, ValueError):
                continue
    return sorted(files, key=lambda item: Path(item["path"]))


def training_distribution(root, relative, feature):
    """
    Read one variable from a saved training histogram block

    Args:
        root:  Dashboard output directory
        relative:  Experiment path relative to the dashboard root
        feature:  Variable index in the histogram manifest
    Returns:
        result:  Bin edges and class weights for all categories
    """
    import numpy as np
    from zipfile import BadZipFile

    run = contained(root, relative)
    if run == root or run != root / Path(relative) or not has_run_record(run):
        raise FileNotFoundError(relative)
    manifest = contained(root, Path(relative) / "plots/training_distributions.json")
    archive = contained(root, Path(relative) / "plots/training_distributions.npz")
    if not manifest.is_relative_to(run) or not archive.is_relative_to(run):
        raise ValueError("Histogram files must be inside their experiment")
    record = read_record(manifest)
    block_size = record.get("feature_block")
    if (record.get("schema_version") != 1 or type(block_size) is not int or block_size < 1
            or not 0 <= feature < len(record.get("features", []))):
        raise ValueError("Unknown training variable")
    block, index = divmod(feature, block_size)
    try:
        with np.load(archive, allow_pickle=False) as data:
            return {"edges": data[f"edges_{block}"][index].tolist(), "weights": data[f"weights_{block}"][index].tolist()}
    except (BadZipFile, KeyError, IndexError) as error:
        raise ValueError("Invalid training histogram file") from error


def category_views(root, directory, files):
    """
    Read category results through paths contained in their evaluation directory

    Args:
        root:  Dashboard root
        directory:  Evaluation directory
        files:  Safe downloadable evaluation files
    Returns:
        views: Bin metrics and comparison figures for the dashboard selector
    """
    manifest_path = contained(root, (directory / "categories.json").relative_to(root))
    manifest = read_record(manifest_path)
    views = []
    for group in manifest.get("categories", []):
        if not isinstance(group, dict):
            continue
        entries = list(group.get("bins", []))
        name = group.get("name", "")
        entries.append(
            {"id": f"{name}/compare", "directory": f"categories/{name}", "label": f"{name} - Compare bins", "status": "comparison"}
        )
        for entry in entries:
            try:
                if not isinstance(entry, dict):
                    continue
                relative = Path(entry["directory"])
                if relative.is_absolute() or relative.parts[:1] != ("categories",):
                    continue
                target = contained(root, (directory / relative).relative_to(root))
                if not target.is_relative_to(directory):
                    continue
                prefix = str(target.relative_to(root))
                comparison = entry.get("status") == "comparison"
                selected = [item for item in files if item["path"].startswith(prefix + ("/plots/" if comparison else "/"))]
                if comparison and not selected:
                    continue
                metrics_path = contained(root, (target / "metrics.json").relative_to(root))
                views.append(
                    {
                        **entry,
                        "directory": prefix,
                        "definition": group.get("definition", {}),
                        "missing_events": group.get("missing_events", 0),
                        "out_of_range_events": group.get("out_of_range_events", 0),
                        "metrics": {} if comparison else read_record(metrics_path),
                        "files": selected,
                    }
                )
            except (KeyError, OSError, ValueError, TypeError):
                continue
    return views

def evaluation_owner(root, checkpoint, run_ids):
    """
    Match a saved checkpoint to its run, including records from another host

    Args:
        root:  Resolved dashboard directory
        checkpoint:  Saved checkpoint path
        run_ids:  Experiment paths relative to root
    Returns:
        run_id: Run path relative to the dashboard root
    """
    owner = checkpoint.parent.parent
    if owner.is_absolute() and owner.is_relative_to(root):
        name = owner.relative_to(root).as_posix()
        return name if name in run_ids else None
    parts = owner.parts[1:] if owner.anchor else owner.parts
    boundary = parts.index(root.name) if root.name in parts else len(parts)
    for start in range(len(parts)):
        if start > boundary and parts[start - 1] != root.name:
            continue
        name = "/".join(parts[start:])
        if name in run_ids:
            return name
    return None

def run_detail(root, name, *, outputs=None):
    """
    Read configuration, histories, evaluations and file links for one run

    Args:
        root:  Dashboard output directory
        name:  Run path relative to the dashboard root
        outputs:  Discovered runs and external evaluation records
    Returns:
        run: JSON-compatible dashboard record without loading model weights
    """
    root = Path(root).resolve()
    relative = Path(name)
    path = contained(root, relative)
    if path == root or path != root / relative or any(
        has_run_record(root / parent) for parent in relative.parents if parent.parts
    ):
        raise ValueError("Select one experiment directory")
    if not has_run_record(path):
        raise FileNotFoundError(name)
    name = relative.as_posix()
    record = read_training_record(path)
    history = read_history(path)
    files = file_list(root, path)
    evaluations = []
    run_ids, external = discover_outputs(root) if outputs is None else outputs
    markers = itertools.chain(path.rglob("evaluation.json"), external)
    for metadata in sorted(markers):
        try:
            metadata = contained(root, metadata.relative_to(root))
            item = read_record(metadata)
            if not item:
                continue
            checkpoint = Path(item.get("checkpoint") or "")
            owner = evaluation_owner(root, checkpoint, run_ids) if checkpoint.parent.name == "checkpoints" else None
            if owner is None and metadata.is_relative_to(path):
                owner = name
            if owner != name:
                continue
            directory = metadata.parent
            relative_directory = directory.relative_to(root).as_posix()
            if directory.is_relative_to(path):
                evaluation_files = [file for file in files if file["path"].startswith(relative_directory + "/")]
            else:
                evaluation_files = file_list(root, directory)
            metrics_path = contained(root, (directory / "metrics.json").relative_to(root))
            item.update(
                elapsed_seconds=duration_seconds(item, active=item.get("status") == "evaluating"),
                directory=relative_directory,
                path=str(directory),
                delete_blocked=deletion_block_reason(directory, item),
                metrics=read_record(metrics_path),
                category_views=category_views(root, directory, evaluation_files),
                files=evaluation_files,
            )
            log_path = contained(root, (path / "logs" / f"{directory.name}.log").relative_to(root))
            if not log_path.is_file():
                log_path = contained(root, (directory / "evaluate.log").relative_to(root))
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 6000))
                    item["log"] = stream.read().decode(errors="replace")
            evaluations.append(item)
        except (OSError, ValueError, TypeError):
            continue
    evaluations.sort(key=lambda item: item.get("started_at", ""))
    files = list(
        {
            item["path"]: item
            for item in files + [file for evaluation in evaluations for file in evaluation["files"]]
        }.values()
    )
    log = path / "logs/train.log"
    if "model_parameters" not in record and log.is_file() and log.resolve().is_relative_to(path):
        # Earlier runs saved parameter counts in the training log header
        with log.open("rb") as stream:
            count = rb"(\d{1,3}(?:,\d{3}){0,5})"
            match = re.search(rb"Model parameters: " + count + rb" total \| " + count + rb" trainable", stream.read(8192))
            if match:
                total, trainable = (int(value.replace(b",", b"")) for value in match.groups())
                record["model_parameters"] = {"total": total, "trainable": trainable}
    tail = training_log_tail(path)
    last_training_activity = training_activity(path, record)
    record["status"] = training_status(path, record, tail, last_training_activity)
    status = record["status"]
    if status == "trained" and evaluations:
        status = (
            "completed"
            if evaluations[-1].get("status") == "completed"
            else "evaluation_" + evaluations[-1].get("status", "unknown")
        )
    return {
        **record,
        "id": name,
        "directory": str(path),
        "command_directory": (path.relative_to(Path.cwd()).as_posix() if path.is_relative_to(Path.cwd()) else str(path)),
        "status": status,
        "last_training_activity": last_training_activity,
        "training_elapsed_seconds": training_seconds(
            record, active=record["status"] == "training", stopped_at=last_training_activity
        ),
        "delete_blocked": deletion_block_reason(path, record, evaluations),
        "history": history,
        "checkpoints": read_record(path / "checkpoints/index.json"),
        "evaluations": evaluations,
        "files": files,
        "log": tail,
    }

def list_runs(root):
    """
    Discover managed experiments under one root

    Args:
        root:  Directory containing experiment directories
    Returns:
        runs: Newest-first experiment summaries
    """
    root = Path(root).resolve()
    outputs = discover_outputs(root)
    run_ids, _ = outputs
    runs = []
    for name in sorted(run_ids):
        try:
            detail = run_detail(root, name, outputs=outputs)
            runs.append(
                {key: detail.get(key) for key in ("id", "status", "created_at", "updated_at", "driver", "model", "task")}
                | {
                    "training_started_at": next(
                        (
                            attempt["started_at"]
                            for attempt in reversed(detail.get("attempts") or [])
                            if isinstance(attempt, dict) and attempt.get("started_at")
                        ),
                        detail.get("created_at"),
                    ),
                    "epochs": len(detail["history"]),
                    "latest": detail["history"][-1] if detail["history"] else {},
                }
            )
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda row: row.get("created_at") or "", reverse=True)

def delete_directory(root, relative, kind, *, force=False):
    """
    Remove one run or evaluation directory after checking its status

    Args:
        root:  Resolved dashboard root
        relative:  Directory relative to the dashboard root
        kind:  Run or evaluation
        force:  Allow removal of a run with active work recorded
    Returns:
        None: The directory and its contents are removed
    """
    parts = Path(relative).parts
    if not parts:
        raise ValueError("Select one experiment directory")
    path = contained(root, relative)
    if path == root:
        raise ValueError("Cannot delete the experiment root")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Cannot delete through a symbolic link")
    if kind == "run":
        from .evaluations import run_removal

        reason = run_detail(root, relative)["delete_blocked"]
        if reason and not force:
            raise RuntimeError(reason)
        with run_removal(root, path, force=force):
            shutil.rmtree(path)
        return
    if force:
        raise ValueError("Force deletion requires an experiment directory")
    file = path / "evaluation.json"
    if not file.is_file() or file.is_symlink():
        raise FileNotFoundError("Evaluation directory not found")
    if any(path.rglob("experiment.json")):
        raise ValueError("The evaluation directory contains an experiment")
    reason = deletion_block_reason(path, read_record(file))
    if reason:
        raise RuntimeError(reason)
    shutil.rmtree(path)

def list_comparisons(root):
    """
    List comparison jobs stored under the dashboard root

    Args:
        root: Dashboard output directory
    Returns:
        records: Comparison names, status and creation times
    """
    root = Path(root).resolve()
    records = []
    for path in root.rglob("comparison.json"):
        if not (path.parent / "config.json").is_file():
            continue
        try:
            path = contained(root, path.relative_to(root))
            record = read_record(path)
            if record.get("schema_version") == 1 and "name" in record:
                records.append(
                    {key: record.get(key) for key in ("name", "status", "started_at", "finished_at")}
                    | {"id": path.parent.relative_to(root).as_posix()}
                )
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda record: record.get("started_at") or "", reverse=True)

def comparison_detail(root, relative, selection=None):
    """
    Read one comparison and an optional contained event selection

    Args:
        root: Dashboard output directory
        relative: Comparison directory relative to root
        selection: Optional saved selection ID
    Returns:
        record: Comparison metadata and selected metric/curve arrays
    """
    root = Path(root).resolve()
    directory = contained(root, relative)
    record = read_record(contained(directory, "comparison.json"))
    if record.get("schema_version") != 1 or "name" not in record:
        raise ValueError("Comparison not found")
    record["id"] = relative
    log = contained(root, f"comparisons/logs/{directory.name}.log")
    if not log.is_file():
        log = contained(directory, "compare.log")
    if log.is_file():
        with log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 6000))
            record["log"] = stream.read().decode(errors="replace")
    if selection is not None:
        entry = next((row for row in record["selections"] if row["id"] == selection), None)
        if entry is None:
            raise ValueError("Comparison selection not found")
        path = contained(directory, entry["path"])
        record["selection"] = read_record(path)
        record["files"] = file_list(Path(root).resolve(), path.parent / "plots")
    return record

class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        return

    def handle(self):
        """
        Close requests quietly when the browser disconnects

        Returns:
            None: The connection is served or closed
        """
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        """
        Serve fixed dashboard assets, JSON records or contained output files

        Returns:
            None: An HTTP response is written to the browser
        """
        route = unquote(urlsplit(self.path).path)
        try:
            if route == "/api/explorer":
                options = {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}
                return self.send_json(get_index(self.root).query(options))
            if route == "/api/curves":
                options = parse_qs(urlsplit(self.path).query)
                return self.send_json(get_index(self.root).curves(
                    options.get("run", []), options.get("metric", [""])[0],
                    options.get("axis", ["step"])[0], int(options.get("budget", ["600"])[0]),
                ))
            if route == "/api/runs":
                return self.send_json(list_runs(self.root))
            if route == "/api/comparisons":
                return self.send_json(list_comparisons(self.root))
            if route.startswith("/api/comparisons/"):
                selection = parse_qs(urlsplit(self.path).query).get("selection", [None])[0]
                return self.send_json(comparison_detail(self.root, route[len("/api/comparisons/") :], selection))
            if route.startswith("/api/runs/"):
                return self.send_json(run_detail(self.root, route[len("/api/runs/") :]))
            if route.startswith("/api/training-distributions/"):
                feature = parse_qs(urlsplit(self.path).query).get("feature", ["-1"])[0]
                return self.send_json(training_distribution(self.root, route[len("/api/training-distributions/") :], int(feature)))
            if route in ("/", "/dashboard.js", "/comparisons.js", "/explorer.js", "/style.css"):
                name = route.lstrip("/") or "index.html"
                return self.send_file(Path(__file__).parent / "static" / name)
            if route.startswith("/files/"):
                path = contained(self.root, route[len("/files/") :])
                if path.suffix not in {".png", ".pdf", ".json", ".jsonl", ".yml", ".csv", ".log", ".pt", ".npz"}:
                    raise ValueError("Unsupported file")
                return self.send_file(path)
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (OSError, ValueError):
            self.send_error(404)

    def do_POST(self):
        """
        Start a checkpoint evaluation or model comparison on the host

        Returns:
            None: New job ID or a validation error is sent to the browser
        """
        origin = self.headers.get("Origin")
        route = urlsplit(self.path).path
        action = {"/api/comparisons": "compare", "/api/evaluations": "evaluate"}.get(route)
        if action is None:
            return self.send_json({"error": "Unknown job route"}, 404)
        if self.headers.get("X-Twistboard-Action") != action or (
            origin is not None
            and (urlsplit(origin).scheme not in {"http", "https"} or urlsplit(origin).netloc != self.headers.get("Host"))
        ):
            return self.send_json({"error": "Job launch requires a dashboard request"}, 403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32768:
                raise ValueError("Invalid job request size")
            from .comparisons import start_comparison
            from .evaluations import start_evaluation

            launch = start_comparison if action == "compare" else start_evaluation
            result = launch(self.root, json.loads(self.rfile.read(size)))
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (ValueError, TypeError) as error:
            return self.send_json({"error": str(error)}, 400)
        except RuntimeError as error:
            return self.send_json({"error": str(error)}, 409)
        except OSError as error:
            return self.send_json({"error": str(error)}, 500)
        return self.send_json(result, 202)

    def do_DELETE(self):
        """
        Delete a selected run or evaluation from the dashboard

        Returns:
            None: An HTTP response reports removal or the error
        """
        origin = self.headers.get("Origin")
        action = self.headers.get("X-Twistboard-Action")
        if action not in {"delete", "force-delete"} or (
            origin is not None
            and (urlsplit(origin).scheme not in {"http", "https"} or urlsplit(origin).netloc != self.headers.get("Host"))
        ):
            return self.send_json({"error": "Deletion requires a dashboard request"}, 403)
        route = unquote(urlsplit(self.path).path)
        for prefix, kind in [("/api/runs/", "run"), ("/api/evaluations/", "evaluation")]:
            if route.startswith(prefix):
                relative = route[len(prefix) :]
                break
        else:
            return self.send_json({"error": "Unknown deletion route"}, 404)
        try:
            delete_directory(self.root, relative, kind, force=action == "force-delete")
            get_index(self.root).refresh(force=True)
        except FileNotFoundError:
            return self.send_json({"error": "Directory not found. Refresh the dashboard."}, 404)
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)
        except RuntimeError as error:
            return self.send_json({"error": str(error)}, 409)
        except OSError:
            return self.send_json({"error": "Cannot remove the directory. Check file permissions and refresh."}, 500)
        return self.send_json({"deleted": relative})

    def send_json(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        with path.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if path.suffix in {".pt", ".npz"}:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile)

def make_server(root, host="127.0.0.1", port=8765):
    """
    Construct the local experiment dashboard server

    Args:
        root:  Experiment output directory
        host:  Listening address
        port:  Listening port, or zero for a free port
    Returns:
        server: Threaded HTTP server
    """
    root = Path(root).resolve()
    return ThreadingHTTPServer((host, port), partial(Handler, root=root))

def serve_dashboard(root, host="127.0.0.1", port=8765, open_browser=False, force=False):
    """
    Serve the dashboard until interrupted

    Args:
        root:  Experiment output directory
        host:  Listening address
        port:  Listening port
        open_browser:  Open the local URL in the default browser
        force:  Stop a previous same-user dashboard on the requested port
    Returns:
        None: The server shuts down on keyboard interruption
    """
    try:
        server = make_server(root, host, port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        if not force:
            raise RuntimeError(
                f"Address {host}:{port} is already in use. "
                "Use --force to stop the previous Twistboard instance, "
                "or --port 0 to use a free port"
            ) from None
        from .processes import stop_dashboard

        if not stop_dashboard(host, port):
            raise RuntimeError(
                f"Address {host}:{port} is already in use. "
                "--force found no same-user Twistboard listener to stop. "
                "Use --port 0 to use a free port"
            ) from None
        server = make_server(root, host, port)
    with server:
        url = f"http://{host}:{server.server_port}"
        print(f"Twistboard: {url}  |  experiments: {Path(root).resolve()}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
