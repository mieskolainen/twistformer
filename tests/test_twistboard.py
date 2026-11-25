# Experiment lifecycle and dashboard validation
#
# mikael.mieskolainen@cern.ch (2026)

import json
import os
import shlex
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pytest
from twistformer import build_experiment, parse_config, Trainer
from twistboard.catalog import get_index
from twistboard.status import STALE_SECONDS
from twistboard.experiments import evaluation_run, training_run, unique_directory
from twistboard.server import delete_directory, list_runs, make_server, run_detail, training_activity


def test_training_activity_uses_live_files(tmp_path):
    """
    Track live training writes while retaining recorded terminal timestamps

    Args:
        tmp_path:  Temporary experiment directory
    """
    from datetime import datetime, timezone

    start = "2026-10-13T08:00:00+00:00"
    record = {"status": "training", "updated_at": start}
    assert training_activity(tmp_path, record) == start
    for name, hour in [("metrics.csv", 9), ("logs/train.log", 10), ("evaluations/test/metrics.json", 11)]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("progress")
        stamp = datetime(2026, 10, 13, hour, tzinfo=timezone.utc).timestamp()
        os.utime(file, (stamp, stamp))
    assert training_activity(tmp_path, record) == "2026-10-13T10:00:00+00:00"
    record.update(status="interrupted", updated_at="2026-10-13T10:05:00+00:00")
    assert training_activity(tmp_path, record) == record["updated_at"]
    assert training_activity(tmp_path / "missing", {}) is None


@pytest.mark.parametrize("attempts", [None, [], [{"started_at": "2026-10-13T08:00:00+00:00"}]])
def test_sidebar_training_date(tmp_path, attempts):
    """
    Show the latest training start with a creation-date fallback

    Args:
        tmp_path:  Temporary run directory
        attempts:  Saved training attempts
    """
    run = tmp_path / "demo"
    run.mkdir()
    created = "2026-10-12T08:00:00+00:00"
    (run / "experiment.json").write_text(json.dumps({
        "id": "demo", "status": "trained", "created_at": created,
        "attempts": [{"started_at": created}, *attempts] if attempts else attempts,
    }))
    expected = attempts[-1]["started_at"] if attempts else created
    assert list_runs(tmp_path)[0]["training_started_at"] == expected


def test_training_commands_preserve_attempts(tmp_path, monkeypatch):
    """
    Preserve launch arguments and working directories across interrupted resumes

    Args:
        tmp_path:  Temporary run directory
        monkeypatch:  Isolated plotting replacement
    """
    monkeypatch.setattr("twistboard.experiments.training_plots", lambda root: None)
    config = {
        "driver": {"name": "synthetic"},
        "model": {"family": "transformer"},
        "task": {"name": "classification"},
    }
    root = tmp_path / "run"
    command = [
        "python", "-m", "twistformer.cli.train", "--config", "cards/a 'b.yml",
        "--set", "driver.label=$sample",
    ]
    with pytest.raises(KeyboardInterrupt):
        with training_run(root, config, command=command):
            saved = json.loads((root / "experiment.json").read_text())
            assert saved["attempts"][0]["argv"] == command
            raise KeyboardInterrupt
    resume = ["python", "-m", "twistformer.cli.train", "--resume", "run/checkpoints/last.pt"]
    with training_run(root, config, resume=resume[-1], command=resume):
        pass
    attempts = run_detail(tmp_path, "run")["attempts"]
    assert [attempt["argv"] for attempt in attempts] == [command, resume]
    assert [shlex.split(attempt["command"]) for attempt in attempts] == [command, resume]
    assert attempts[0]["status"] == "interrupted"
    assert attempts[1]["status"] == "trained"
    assert all(
        attempt["working_directory"] == str(Path.cwd()) for attempt in attempts
    )


@pytest.fixture
def deletion_dashboard(tmp_path):
    """
    Serve temporary experiment directories for deletion checks

    Args:
        tmp_path:  Temporary test directory
    Returns:
        dashboard: File root and local server URL
    """
    root = tmp_path / "runs"
    for name in ["demo", "other"]:
        run = root / name
        run.mkdir(parents=True)
        (run / "experiment.json").write_text('{"status":"trained"}')
    with make_server(root, port=0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield root, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join(timeout=5)


def delete_request(url, route, headers=None):
    """
    Send a directory deletion request

    Args:
        url:  Dashboard URL
        route:  API path
        headers:  Optional request headers
    Returns:
        response: Open HTTP response
    """
    return urlopen(Request(
        url + route, method="DELETE",
        headers={"X-Twistboard-Action": "delete"} if headers is None else headers,
    ))


def test_delete_run_removes_only_selected_folder(deletion_dashboard, tmp_path):
    """
    Remove the run contents without following file links

    Args:
        deletion_dashboard:  Temporary dashboard
        tmp_path:  Temporary test directory
    """
    root, url = deletion_dashboard
    run = root / "demo"
    (run / "checkpoints").mkdir()
    (run / "checkpoints/best.pt").write_bytes(b"weights")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (run / "linked").symlink_to(outside, target_is_directory=True)
    with delete_request(url, "/api/runs/demo") as response:
        assert json.load(response) == {"deleted": "demo"}
    assert not run.exists()
    assert (outside / "keep.txt").read_text() == "keep"
    assert (root / "other/experiment.json").is_file()
    with urlopen(url + "/api/runs") as response:
        assert [item["id"] for item in json.load(response)] == ["other"]


def test_nested_experiments_use_relative_paths(deletion_dashboard):
    """
    Discover grouped runs and use their paths for curves, files and evaluations

    Args:
        deletion_dashboard:  Temporary dashboard
    """
    root, url = deletion_dashboard
    name = "scan group/preset/demo"
    sibling = "other group/demo"
    for directory in [name, sibling, ".hidden/demo"]:
        run = root / directory
        run.mkdir(parents=True)
        (run / "experiment.json").write_text('{"status":"trained"}')
    (root / "alias").symlink_to(root / "scan group", target_is_directory=True)
    (root / name / "metrics.csv").write_text("epoch,step,val_loss\n0,1,2\n")
    for directory, checkpoint in [
        (name + "/evaluations/test", root / name / "checkpoints/best.pt"),
        (name + "/evaluations/moved", root / "old-run/checkpoints/best.pt"),
        ("copied/test", Path("/previous/runs/archive/runs") / name / "checkpoints/best.pt"),
    ]:
        output = root / directory
        output.mkdir(parents=True)
        (output / "evaluation.json").write_text(json.dumps({"checkpoint": str(checkpoint), "status": "completed"}))

    for route in ["/api/runs", "/api/explorer"]:
        with urlopen(url + route) as response:
            result = json.load(response)
            rows = result["rows"] if route == "/api/explorer" else result
            assert {row["id"] for row in rows} == {"demo", "other", name, sibling}
    for run_id in [name, sibling, "demo"]:
        with urlopen(url + "/api/runs/" + quote(run_id, safe="")) as response:
            detail = json.load(response)
            assert detail["id"] == run_id
            expected = {name + "/evaluations/test", name + "/evaluations/moved", "copied/test"} if run_id == name else set()
            assert {item["directory"] for item in detail["evaluations"]} == expected
    with urlopen(url + "/api/curves?metric=val_loss&run=" + quote(name, safe="")) as response:
        assert json.load(response)[0]["points"] == [[1, 2]]
    with urlopen(url + "/files/" + quote(name, safe="/") + "/metrics.csv") as response:
        assert response.read() == b"epoch,step,val_loss\n0,1,2\n"
    with delete_request(url, "/api/runs/" + quote(name, safe="")) as response:
        assert json.load(response)["deleted"] == name
    assert (root / name).parent.is_dir()
    assert (root / sibling).is_dir()
    with urlopen(url + "/api/explorer") as response:
        assert {row["id"] for row in json.load(response)["rows"]} == {"demo", "other", sibling}
    with urlopen(url + "/api/runs/demo") as response:
        assert not json.load(response)["evaluations"]


def test_delete_selected_evaluation(deletion_dashboard):
    """
    Remove one evaluation and retain its checkpoint run and other evaluations

    Args:
        deletion_dashboard:  Temporary dashboard
    """
    root, url = deletion_dashboard
    for directory in ["demo/evaluations/test one", "demo/evaluations/test-two", "external"]:
        output = root / directory
        output.mkdir(parents=True)
        (output / "evaluation.json").write_text(json.dumps({
            "status": "completed", "checkpoint": str(root / "demo/checkpoints/best.pt")
        }))
        (output / "predictions.npz").write_bytes(b"predictions")
    for directory in ["demo/evaluations/test one", "external"]:
        with delete_request(url, "/api/evaluations/" + quote(directory, safe="")) as response:
            assert json.load(response)["deleted"] == directory
        assert not (root / directory).exists()
    assert (root / "demo/experiment.json").is_file()
    assert (root / "demo/evaluations/test-two/predictions.npz").is_file()
    with urlopen(url + "/api/runs/demo") as response:
        assert len(json.load(response)["evaluations"]) == 1


@pytest.mark.parametrize("relative", ["", ".", "..", "../other", "/", "demo/..", "demo/checkpoints", ".hidden", "alias"])
def test_delete_rejects_invalid_run_paths(deletion_dashboard, relative):
    """
    Reject root, non-experiment, hidden and symbolic run paths

    Args:
        deletion_dashboard:  Temporary dashboard
        relative:  Rejected directory path
    """
    root, url = deletion_dashboard
    (root / "alias").symlink_to(root / "demo", target_is_directory=True)
    for action in ("delete", "force-delete"):
        with pytest.raises(HTTPError) as error:
            delete_request(url, "/api/runs/" + quote(relative, safe=""), {"X-Twistboard-Action": action})
        assert error.value.code == 400
    assert (root / "demo/experiment.json").is_file()


def test_delete_evaluation_rejects_escape_and_run_folders(deletion_dashboard, tmp_path):
    """
    Reject evaluation paths outside the root or containing experiment records

    Args:
        deletion_dashboard:  Temporary dashboard
        tmp_path:  Temporary test directory
    """
    root, url = deletion_dashboard
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evaluation.json").write_text("{}")
    (root / "demo/evaluation.json").write_text("{}")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    for relative in ["", "../outside", str(outside), "alias", "demo"]:
        with pytest.raises(HTTPError) as error:
            delete_request(url, "/api/evaluations/" + quote(relative, safe=""))
        assert error.value.code == 400
    assert (outside / "evaluation.json").is_file()
    assert (root / "demo/experiment.json").is_file()


@pytest.mark.parametrize("active", ["training", "training-without-lock", "lock", "internal-evaluation", "external-evaluation"])
def test_active_work_requires_force_to_delete_run(deletion_dashboard, active):
    """
    Require explicit force deletion while work is active

    Args:
        deletion_dashboard:  Temporary dashboard
        active:  Active operation or training lock
    """
    root, url = deletion_dashboard
    routes = ["/api/runs/demo"]
    if active in {"training", "training-without-lock"}:
        (root / "demo/experiment.json").write_text('{"status":"training"}')
        if active == "training":
            (root / "demo/.training.lock").write_text("1234")
    elif active == "lock":
        (root / "demo/.training.lock").write_text("1234")
    else:
        relative = "demo/evaluations/test" if active == "internal-evaluation" else "external"
        output = root / relative
        output.mkdir(parents=True)
        (output / "evaluation.json").write_text(json.dumps({
            "status": "evaluating", "checkpoint": str(root / "demo/checkpoints/best.pt")
        }))
        routes.append("/api/evaluations/" + quote(relative, safe=""))
    for route in routes:
        with pytest.raises(HTTPError) as error:
            delete_request(url, route)
        assert error.value.code == 409
        assert "Stop" in json.load(error.value)["error"]
    assert (root / "demo/experiment.json").is_file()
    with urlopen(url + "/api/runs/demo") as response:
        detail = json.load(response)
    assert detail["delete_blocked"]
    if detail["evaluations"]:
        assert detail["evaluations"][0]["delete_blocked"]
        with pytest.raises(HTTPError) as error:
            delete_request(url, routes[1], {"X-Twistboard-Action": "force-delete"})
        assert error.value.code == 400
    with delete_request(url, routes[0], {"X-Twistboard-Action": "force-delete", "Origin": url}) as response:
        assert json.load(response) == {"deleted": "demo"}
    assert not (root / "demo").exists()
    assert (root / "other/experiment.json").is_file()
    if active == "external-evaluation":
        assert (root / "external/evaluation.json").is_file()


def test_stopped_training_status_and_deletion(deletion_dashboard):
    """
    Show a recorded interruption and permit deletion after lock removal

    Args:
        deletion_dashboard:  Temporary dashboard
    """
    root, url = deletion_dashboard
    marker = root / "demo/experiment.json"
    marker.write_text('{"status":"interrupted"}')
    with urlopen(url + "/api/runs/demo") as response:
        detail = json.load(response)
    assert detail["status"] == "interrupted"
    assert detail["delete_blocked"] is None
    assert next(item for item in list_runs(root) if item["id"] == "demo")["status"] == "interrupted"
    assert json.loads(marker.read_text())["status"] == "interrupted"
    with delete_request(url, "/api/runs/demo") as response:
        assert response.status == 200
    assert not marker.parent.exists()


@pytest.fixture
def pending_training_record(tmp_path):
    """
    Store a final training record before its atomic replacement

    Args:
        tmp_path:  Temporary experiment root
    Returns:
        records: Run directory, active record and pending interruption
    """
    run = tmp_path / "demo"
    run.mkdir()
    current = {
        "schema_version": 1,
        "id": "demo",
        "created_at": "2026-10-12T12:00:00+00:00",
        "updated_at": "2026-10-12T12:00:00+00:00",
        "configuration": {"driver": {"name": "synthetic"}},
        "status": "training",
        "attempts": [{"started_at": "2026-10-12T12:00:00+00:00", "resume": None}],
    }
    pending = json.loads(json.dumps(current))
    pending.update(
        status="interrupted", updated_at="2026-10-12T12:30:00+00:00",
        error="KeyboardInterrupt: ",
    )
    pending["attempts"][-1].update(
        status="interrupted", finished_at="2026-10-12T12:29:59+00:00"
    )
    return run, current, pending


@pytest.mark.parametrize("status", ["interrupted", "failed", "trained"])
@pytest.mark.parametrize("resumed", [False, True])
def test_recover_final_training_record(pending_training_record, status, resumed):
    """
    Recover complete final metadata without a final traceback or epoch history

    Args:
        pending_training_record:  Current and pending metadata
        status:  Final training status
        resumed:  Include a completed earlier attempt
    Returns:
        None: Detail, listing and deletion use the same recovered status
    """
    run, current, pending = pending_training_record
    pending["status"] = pending["attempts"][-1]["status"] = status
    if resumed:
        previous = {"started_at": "2026-10-12T11:00:00+00:00", "status": "interrupted"}
        for record in (current, pending):
            record["attempts"][-1]["resume"] = "checkpoints/last.pt"
            record["attempts"].insert(0, previous)
    marker = run / "experiment.json"
    marker.write_text(json.dumps(current))
    (run / "experiment.json.tmp").write_text(json.dumps(pending))
    detail = run_detail(run.parent, run.name)
    assert detail["status"] == status
    assert get_index(run.parent).refresh()[0]["status"] == status
    assert detail["updated_at"] == pending["updated_at"]
    assert detail["attempts"] == pending["attempts"]
    assert detail["delete_blocked"] is None
    assert list_runs(run.parent)[0]["status"] == status
    assert json.loads(marker.read_text()) == current
    delete_directory(run.parent, run.name, "run")
    assert not run.exists()


@pytest.mark.parametrize("case", [
    "lock", "partial", "stale", "identity", "configuration", "resumed",
    "different-start", "unfinished", "invalid-time", "naive-time",
    "finish-after-update", "status-mismatch", "malformed-attempts",
])
def test_ignore_unconfirmed_final_record(pending_training_record, case):
    """
    Retain active training when pending metadata cannot confirm this attempt ended

    Args:
        pending_training_record:  Current and pending metadata
        case:  Incomplete, stale or unrelated evidence
    Returns:
        None: Training remains active and deletion remains blocked
    """
    run, current, pending = pending_training_record
    if case == "lock":
        (run / ".training.lock").write_text("1234")
    elif case == "stale":
        pending["updated_at"] = current["updated_at"]
    elif case == "identity":
        pending["id"] = "other"
    elif case == "configuration":
        pending["configuration"] = {"driver": {"name": "toptag"}}
    elif case == "resumed":
        current["attempts"].append({"started_at": "2026-10-12T13:00:00+00:00"})
    elif case == "different-start":
        pending["attempts"][-1]["started_at"] = "2026-10-12T11:00:00+00:00"
    elif case == "unfinished":
        pending["attempts"][-1].pop("finished_at")
    elif case == "invalid-time":
        pending["updated_at"] = "unknown"
    elif case == "naive-time":
        pending["updated_at"] = "2026-10-12T12:30:00"
    elif case == "finish-after-update":
        pending["attempts"][-1]["finished_at"] = "2026-10-12T13:00:00+00:00"
    elif case == "status-mismatch":
        pending["attempts"][-1]["status"] = "training"
    elif case == "malformed-attempts":
        pending["attempts"] = [None]
    (run / "experiment.json").write_text(json.dumps(current))
    (run / "experiment.json.tmp").write_text(
        '{"status": "interrupted",' if case == "partial" else json.dumps(pending)
    )
    detail = run_detail(run.parent, run.name)
    assert detail["status"] == "training"
    assert detail["delete_blocked"]
    with pytest.raises(RuntimeError, match="Stop training"):
        delete_directory(run.parent, run.name, "run")


def test_interrupt_during_final_record_replacement(tmp_path, monkeypatch):
    """
    Recover a second Ctrl+C between the final JSON write and atomic replacement

    Args:
        tmp_path:  Temporary experiment root
        monkeypatch:  Atomic replacement interruption
    Returns:
        None: A complete pending record identifies the interrupted attempt
    """
    from twistboard import outputs

    replace = outputs.os.replace
    root = tmp_path / "demo"

    def interrupt_final_save(source, target):
        """
        Interrupt only the terminal experiment replacement

        Args:
            source:  Complete temporary JSON file
            target:  Destination record
        Returns:
            None: Initial writes succeed and the final write is interrupted
        """
        if target.name == "experiment.json" and json.loads(source.read_text())["status"] == "interrupted":
            raise KeyboardInterrupt
        replace(source, target)

    monkeypatch.setattr(outputs.os, "replace", interrupt_final_save)
    config = {
        "driver": {"name": "synthetic"},
        "model": {"family": "pelican"},
        "task": {"name": "classification"},
    }
    with pytest.raises(KeyboardInterrupt):
        with training_run(root, config):
            raise KeyboardInterrupt
    assert json.loads((root / "experiment.json").read_text())["status"] == "training"
    assert not (root / ".training.lock").exists()
    assert run_detail(tmp_path, "demo")["status"] == "interrupted"


@pytest.mark.parametrize("status", ["training", "trained", "failed", "interrupted"])
def test_training_status_does_not_require_lock(tmp_path, status):
    """
    Preserve recorded states with or without a training lock

    Args:
        tmp_path:  Temporary experiment root
        status:  Saved training status
    """
    run = tmp_path / "demo"
    run.mkdir()
    (run / "experiment.json").write_text(json.dumps({"status": status}))
    lock = run / ".training.lock"
    lock.write_text("1234")
    detail = run_detail(tmp_path, "demo")
    assert detail["status"] == status
    assert detail["delete_blocked"]
    lock.unlink()
    detail = run_detail(tmp_path, "demo")
    assert detail["status"] == status
    assert bool(detail["delete_blocked"]) == (status == "training")


@pytest.mark.parametrize("locked", [False, True])
def test_stale_training_recovers_on_new_activity(tmp_path, monkeypatch, locked):
    """
    Age a cached training state and restore it on a new training event

    Args:
        tmp_path:  Temporary experiment root
        monkeypatch:  Training clock override
        locked:  Keep a training lock in the run directory
    """
    run = tmp_path / "demo"
    run.mkdir()
    record = {"status": "training", "elapsed_seconds": 18}
    marker = run / "experiment.json"
    marker.write_text(json.dumps(record))
    if locked:
        (run / ".training.lock").write_text("1234")
    now = marker.stat().st_mtime
    monkeypatch.setattr("twistboard.status.time.time", lambda: now)
    index = get_index(tmp_path)
    assert index.refresh()[0]["status"] == "training"
    now += STALE_SECONDS + 1
    assert index.refresh(force=True)[0]["status"] == "stale"
    detail = run_detail(tmp_path, "demo")
    assert detail["status"] == "stale"
    assert detail["training_elapsed_seconds"] == 18
    assert detail["delete_blocked"]
    assert json.loads(marker.read_text()) == record
    events = run / "events.jsonl"
    events.write_text('{"type":"scalar","step":1,"values":{"loss":0.2}}\n')
    os.utime(events, (now, now))
    assert index.refresh(force=True)[0]["status"] == "training"
    assert run_detail(tmp_path, "demo")["status"] == "training"


def test_parameter_counts_from_saved_log(tmp_path):
    """
    Recover the exact parameter counts when only the startup log has them

    Args:
        tmp_path:  Temporary experiment root
    """
    run = tmp_path / "demo"
    (run / "logs").mkdir(parents=True)
    marker = run / "experiment.json"
    marker.write_text('{"status":"trained"}')
    (run / "logs/train.log").write_text("Model parameters: 314,689 total | 42 trainable\n" + "x" * 20000)
    assert run_detail(tmp_path, "demo")["model_parameters"] == {"total": 314689, "trainable": 42}
    assert "model_parameters" not in json.loads(marker.read_text())


def test_training_before_first_epoch_without_synced_lock(tmp_path):
    """
    Keep training active before the first epoch when hidden files are not synced

    Args:
        tmp_path:  Temporary experiment root
    """
    root = tmp_path / "source" / "demo"
    mirror = tmp_path / "mirror"
    config = {
        "driver": {"name": "synthetic"},
        "model": {"family": "transformer"},
        "task": {"name": "classification"},
    }
    with training_run(root, config):
        shutil.copytree(root, mirror / "demo", ignore=shutil.ignore_patterns(".*"))
        assert not (mirror / "demo/.training.lock").exists()
        for directory in (root.parent, mirror):
            detail = run_detail(directory, "demo")
            assert detail["status"] == "training"
            assert detail["history"] == []
            assert detail["delete_blocked"]
            summary = list_runs(directory)[0]
            assert summary["status"] == "training"
            assert summary["epochs"] == 0
            with pytest.raises(RuntimeError, match="Stop training"):
                delete_directory(directory, "demo", "run")


@pytest.mark.parametrize("case", ["interrupted", "message", "timed-message", "lock", "resumed", "progress", "mention", "trained"])
def test_training_traceback_status(deletion_dashboard, case):
    """
    Recover a final interruption without overriding active or resumed training

    Args:
        deletion_dashboard:  Temporary dashboard
        case:  Training evidence in the saved record and log
    """
    root, url = deletion_dashboard
    run = root / "demo"
    record = {"status": "training", "attempts": [{"resume": None}]}
    tail = 'Traceback (most recent call last):\n  File "train.py", line 1\nKeyboardInterrupt\n'
    if case in {"message", "timed-message"}:
        tail = "Training interrupted. Completed epoch checkpoints are retained\n"
        if case == "timed-message":
            tail += "Total training time: 01:20:35\n"
    elif case == "lock":
        (run / ".training.lock").write_text("1234")
    elif case == "resumed":
        record["attempts"].append({"resume": "checkpoints/last.pt"})
    elif case == "progress":
        tail += "Loading train dataset...\n"
    elif case == "mention":
        tail = "KeyboardInterrupt\n"
    elif case == "trained":
        record["status"] = "trained"
    marker = run / "experiment.json"
    marker.write_text(json.dumps(record))
    (run / "logs").mkdir()
    (run / "logs/train.log").write_text(tail)
    expected = "interrupted" if case in {"interrupted", "message", "timed-message"} else record["status"]
    with urlopen(url + "/api/runs/demo") as response:
        detail = json.load(response)
    assert detail["status"] == expected
    assert next(item for item in get_index(root).refresh() if item["id"] == "demo")["status"] == expected
    assert next(item for item in list_runs(root) if item["id"] == "demo")["status"] == expected
    assert json.loads(marker.read_text()) == record
    if expected == "training":
        assert detail["delete_blocked"]
        with pytest.raises(HTTPError) as error:
            delete_request(url, "/api/runs/demo")
        assert error.value.code == 409
    else:
        assert detail["delete_blocked"] is None
        with delete_request(url, "/api/runs/demo") as response:
            assert response.status == 200


@pytest.mark.parametrize("interrupted", [False, True])
def test_training_status_saved_before_plot_interrupt(tmp_path, monkeypatch, interrupted):
    """
    Save the final training status before a second interrupt stops plot generation

    Args:
        tmp_path:  Temporary experiment root
        monkeypatch:  Plot function replacement
        interrupted:  Stop training with a first keyboard interrupt
    """
    root = tmp_path / "run"
    expected = "interrupted" if interrupted else "trained"

    def stop_plots(path):
        """
        Interrupt plotting after checking the saved training status

        Args:
            path:  Run directory
        """
        record = json.loads((path / "experiment.json").read_text())
        assert record["status"] == expected
        assert record["attempts"][-1]["status"] == expected
        assert record["attempts"][-1]["finished_at"]
        raise KeyboardInterrupt

    monkeypatch.setattr("twistboard.experiments.training_plots", stop_plots)
    config = {"driver": {"name": "synthetic"}, "model": {"family": "transformer"}, "task": {"name": "classification"}}
    with pytest.raises(KeyboardInterrupt):
        with training_run(root, config):
            if interrupted:
                raise KeyboardInterrupt
    assert json.loads((root / "experiment.json").read_text())["status"] == expected
    assert not (root / ".training.lock").exists()


def test_delete_requires_dashboard_request(deletion_dashboard):
    """
    Reject requests without a dashboard header or from another origin

    Args:
        deletion_dashboard:  Temporary dashboard
    """
    root, url = deletion_dashboard
    for headers in (
        {},
        {"X-Twistboard-Action": "delete", "Origin": "https://other.example"},
        {"X-Twistboard-Action": "force-delete", "Origin": "https://other.example"},
    ):
        with pytest.raises(HTTPError) as error:
            delete_request(url, "/api/runs/demo", headers)
        assert error.value.code == 403
    assert (root / "demo/experiment.json").is_file()
    with delete_request(url, "/api/runs/demo", {"X-Twistboard-Action": "delete", "Origin": url}) as response:
        assert response.status == 200


def test_delete_reports_missing_and_permission_errors(deletion_dashboard, monkeypatch):
    """
    Report missing directories and file permission errors

    Args:
        deletion_dashboard:  Temporary dashboard
        monkeypatch:  Test replacement for directory removal
    """
    root, url = deletion_dashboard
    with pytest.raises(HTTPError) as error:
        delete_request(url, "/api/runs/missing")
    assert error.value.code == 404

    def denied(path):
        """
        Simulate a directory permission error

        Args:
            path:  Directory selected for removal
        """
        raise PermissionError(path)

    monkeypatch.setattr("twistboard.server.shutil.rmtree", denied)
    with pytest.raises(HTTPError) as error:
        delete_request(url, "/api/runs/demo")
    assert error.value.code == 500
    assert "permissions" in json.load(error.value)["error"]
    assert (root / "demo/experiment.json").is_file()


def test_lifecycle_checkpoints_plots_and_failure(tmp_path):
    config = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 8,
                "val_size": 4,
                "max_tokens": 3,
            },
            "model": {"width": 8, "heads": 2, "depth": 1},
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "device": "cpu",
                "save_every": 1,
            },
        }
    )
    root = unique_directory(tmp_path, "test")
    experiment = build_experiment(config)
    next(experiment.model.parameters()).requires_grad_(False)
    counts = {
        "total": sum(p.numel() for p in experiment.model.parameters()),
        "trainable": sum(p.numel() for p in experiment.model.parameters() if p.requires_grad),
    }
    checkpoint = Trainer(root).fit(experiment)
    record = json.loads((root / "experiment.json").read_text())
    assert record["status"] == "trained"
    assert record["model_parameters"] == counts
    assert run_detail(tmp_path, root.name)["model_parameters"] == counts
    assert not (root / ".training.lock").exists()
    assert (root / "plots/loss.png").is_file()
    assert (root / "plots/loss.pdf").is_file()
    assert "epoch 1/1" in (root / "logs/train.log").read_text()
    index = json.loads((root / "checkpoints/index.json").read_text())
    assert {item["file"] for item in index["checkpoints"]} == {
        "best.pt",
        "last.pt",
        "epoch_00000.pt",
    }
    output = unique_directory(root / "evaluations", "test-best")
    with evaluation_run(checkpoint, output, "test"):
        pass
    evaluated = json.loads((output / "evaluation.json").read_text())
    assert evaluated["status"] == "completed" and len(evaluated["sha256"]) == 64
    assert list_runs(tmp_path)[0]["status"] == "completed"
    with pytest.raises(ValueError, match="not empty"):
        with evaluation_run(checkpoint, output, "test"):
            pass
    broken = build_experiment(config)
    broken.driver.dataset = lambda split: (_ for _ in ()).throw(
        ValueError("missing data")
    )
    failure = tmp_path / "failure"
    with pytest.raises(ValueError, match="missing data"):
        Trainer(failure).fit(broken)
    assert json.loads((failure / "experiment.json").read_text())["status"] == "failed"
    assert not (failure / ".training.lock").exists()


def test_dashboard_http_and_file_boundaries(tmp_path):
    root = tmp_path / "runs"
    run = root / "demo"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "status": "trained",
                "created_at": "2026-10-08",
                "driver": "synthetic",
            }
        )
    )
    (run / "metrics.csv").write_text("epoch,train_loss,val_loss,lr\n0,1,2,0.01\n1,")
    external = tmp_path / "private.json"
    external.write_text('{"private":true}')
    (run / "escape.json").symlink_to(external)
    with make_server(root, port=0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(url) as response:
                assert b"twistboard" in response.read()
            with urlopen(url + "/api/runs") as response:
                assert json.load(response)[0]["epochs"] == 1
            with urlopen(url + "/api/runs/demo") as response:
                detail = json.load(response)
                assert len(detail["history"]) == 1
                assert not any(
                    item["path"].endswith("escape.json") for item in detail["files"]
                )
            for path in [
                "/files/demo/escape.json",
                "/files/%2e%2e/private.json",
                "/files/demo/.training.lock",
            ]:
                with pytest.raises(HTTPError) as error:
                    urlopen(url + path)
                assert error.value.code == 404
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_evaluations_follow_checkpoint_and_expose_all_files(tmp_path):
    """
    Associate external output directories with the checkpoint run and its figures
    """
    from twistboard.server import run_detail

    root = tmp_path / "runs"
    run = root / "tau3-source"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps({"status": "trained"}))
    other = root / "tau3-example" / "evaluations" / "test-best"
    (other / "plots").mkdir(parents=True)
    record = {
        "checkpoint": "/eos/previous-mount/runs/tau3-source/checkpoints/best.pt",
        "status": "completed",
        "split": "test",
        "started_at": "2026-10-12",
    }
    (other / "evaluation.json").write_text(json.dumps(record))
    metrics = {
        "accuracy": 0.9,
        "classes": {"0pi0": {"auc": None}},
        "confusion_matrix": [[1, 0], [0, 2]],
    }
    (other / "metrics.json").write_text(json.dumps(metrics))
    for name in ["roc.png", "roc.pdf", "confusion.png", "scores.pdf"]:
        (other / "plots" / name).write_bytes(b"figure")
    (other / "predictions.npz").write_bytes(b"predictions")
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"private")
    (other / "plots" / "escape.pdf").symlink_to(outside)
    detail = run_detail(root.resolve(), run.name)
    assert detail["status"] == "completed"
    assert len(detail["evaluations"]) == 1
    evaluation = detail["evaluations"][0]
    assert evaluation["metrics"] == metrics
    assert evaluation["directory"] == "tau3-example/evaluations/test-best"
    paths = {item["path"] for item in evaluation["files"]}
    assert len([path for path in paths if "/plots/" in path]) == 4
    assert not any("escape" in path for path in paths)
    assert paths <= {item["path"] for item in detail["files"]}
    other_run = root / "unrelated"
    other_run.mkdir()
    (other_run / "experiment.json").write_text("{}")
    assert not run_detail(root.resolve(), other_run.name)["evaluations"]


def test_partial_and_failed_evaluation_records_remain_visible(tmp_path):
    """
    Keep live and failed evaluations visible before metrics are written
    """
    from twistboard.server import run_detail

    root = tmp_path / "runs"
    run = root / "demo"
    output = run / "evaluations" / "test"
    output.mkdir(parents=True)
    (run / "experiment.json").write_text('{"status":"trained"}')
    record = {"split": "test", "status": "failed", "error": "Missing input"}
    (output / "evaluation.json").write_text(json.dumps(record))
    (output / "metrics.json").write_text('{"accuracy":')
    evaluation = run_detail(root.resolve(), run.name)["evaluations"][0]
    assert evaluation["status"] == "failed"
    assert evaluation["error"] == "Missing input"
    assert evaluation["metrics"] == {}
