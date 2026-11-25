# Host evaluation launches and SSH tunnel requests
#
# mikael.mieskolainen@cern.ch (2026)

import io
import json
import sys
from types import SimpleNamespace

import pytest

from twistboard import evaluations
from twistboard.experiments import evaluation_run
from twistboard.server import Handler, delete_directory, run_detail


@pytest.fixture
def checkpoint(tmp_path):
    """
    Create a contained checkpoint with an experiment record

    Args:
        tmp_path:  Temporary output directory
    Returns:
        path: Checkpoint with spaces in its experiment name
    """
    run = tmp_path / "run with spaces"
    (run / "checkpoints").mkdir(parents=True)
    (run / "experiment.json").write_text('{"status": "trained"}')
    path = run / "checkpoints/best.pt"
    path.write_bytes(b"checkpoint snapshot")
    return path


@pytest.fixture
def host(monkeypatch):
    """
    Capture subprocess arguments and retain the queued job until inspected

    Args:
        monkeypatch:  Isolated process and watcher replacements
    Returns:
        calls: Host process launches
    """
    calls = []

    def launch(argv, **options):
        """
        Record an argument-vector launch without invoking a shell
        """
        calls.append((argv, options))
        options["stdout"].write("Host evaluation log\n")
        return SimpleNamespace(poll=lambda: None, wait=lambda: 1)

    monkeypatch.setattr(evaluations.subprocess, "Popen", launch)
    monkeypatch.setattr(
        evaluations.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None),
    )
    return calls


def test_host_launch_status_log_and_startup_failure(tmp_path, checkpoint, host):
    """
    Preserve queued work, reject duplicates and expose early process failures

    Args:
        tmp_path:  Dashboard root
        checkpoint:  Selected experiment checkpoint
        host:  Captured host launches
    """
    payload = {"checkpoint": checkpoint.relative_to(tmp_path).as_posix()}
    job = evaluations.start_evaluation(tmp_path, payload)
    output = tmp_path / job["id"]
    argv, options = host[0]
    assert argv == [
        sys.executable,
        "-u",
        "-m",
        "twistformer.cli.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--output",
        str(output),
    ]
    assert options["start_new_session"] is True
    assert "shell" not in options
    detail = run_detail(tmp_path, "run with spaces")
    assert detail["evaluations"][0]["status"] == "queued"
    assert detail["evaluations"][0]["log"] == "Host evaluation log\n"
    assert detail["delete_blocked"]
    with pytest.raises(RuntimeError, match="already running"):
        evaluations.start_evaluation(tmp_path, payload)
    with pytest.raises(RuntimeError, match="Stop"):
        delete_directory(tmp_path, job["id"], "evaluation")
    log = checkpoint.parent.parent / "logs" / f"{output.name}.log"
    _, process = evaluations._jobs[str(tmp_path.resolve())]
    evaluations._finish_job(process, output, log)
    record = json.loads((output / "evaluation.json").read_text())
    assert record["status"] == "failed"
    assert "code 1" in record["error"]
    assert (output / "evaluate.log").read_text() == "Host evaluation log\n"


def test_force_delete_stops_only_the_selected_runs_evaluation(tmp_path, checkpoint, monkeypatch):
    """
    Stop the dashboard child before removing its experiment

    Args:
        tmp_path:  Dashboard root
        checkpoint:  Checkpoint for the live job
        monkeypatch:  Launch a long-lived test child in place of model evaluation
    """
    from twistboard.jobs import launch_job

    def launch(argv, *args, **options):
        """Use the normal process launch and a waiting Python child"""
        return launch_job(["-c", "import time\ntime.sleep(60)"], *args, **options)

    monkeypatch.setattr(evaluations, "launch_job", launch)
    evaluations.start_evaluation(tmp_path, {"checkpoint": checkpoint.relative_to(tmp_path).as_posix()})
    run, process = evaluations._jobs[str(tmp_path)]
    other = tmp_path / "other"
    other.mkdir()
    (other / "experiment.json").write_text('{"status":"training"}')
    try:
        delete_directory(tmp_path, other.name, "run", force=True)
        assert not other.exists() and process.poll() is None
        delete_directory(tmp_path, run.name, "run", force=True)
        assert process.returncode is not None and not run.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_reserved_evaluation_uses_existing_lifecycle(tmp_path, checkpoint, host):
    """
    Consume only matching queued reservations and protect completed files

    Args:
        tmp_path:  Dashboard root
        checkpoint:  Selected experiment checkpoint
        host:  Captured host launches
    """
    job = evaluations.start_evaluation(
        tmp_path, {"checkpoint": checkpoint.relative_to(tmp_path).as_posix()}
    )
    output = tmp_path / job["id"]
    with pytest.raises(ValueError, match="not empty"):
        with evaluation_run(checkpoint, output, "val"):
            pass
    with evaluation_run(checkpoint, output, "test") as stream:
        assert stream.read() == b"checkpoint snapshot"
    assert json.loads((output / "evaluation.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="not empty"):
        with evaluation_run(checkpoint, output, "test"):
            pass


def test_host_spawn_failure_is_visible(tmp_path, checkpoint, monkeypatch):
    """
    Mark an unstarted evaluation as failed and allow file removal

    Args:
        tmp_path:  Dashboard root
        checkpoint:  Selected experiment checkpoint
        monkeypatch:  Process launch failure
    """

    def fail(*args, **kwargs):
        """
        Simulate an unavailable host process executable
        """
        raise OSError("Host process unavailable")

    monkeypatch.setattr(evaluations.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="Host process unavailable"):
        evaluations.start_evaluation(
            tmp_path, {"checkpoint": checkpoint.relative_to(tmp_path).as_posix()}
        )
    item = run_detail(tmp_path, "run with spaces")["evaluations"][0]
    assert item["status"] == "failed"
    assert item["error"] == "Host process unavailable"
    assert item["delete_blocked"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"checkpoint": 3},
        {"checkpoint": "../escape.pt"},
        {"checkpoint": "/tmp/escape.pt"},
        {"checkpoint": "run with spaces/experiment.json"},
    ],
)
def test_invalid_checkpoint_rejected(tmp_path, checkpoint, payload):
    """
    Reject invalid files before any process launch

    Args:
        tmp_path:  Dashboard root
        checkpoint:  Contained checkpoint fixture
        payload:  Invalid browser request
    """
    with pytest.raises(ValueError):
        evaluations.start_evaluation(tmp_path, payload)


@pytest.mark.parametrize(
    "origin, status", [("http://127.0.0.1:18765", 202), ("https://other.example", 403)]
)
def test_evaluation_post_through_forwarded_port(tmp_path, monkeypatch, origin, status):
    """
    Accept the forwarded browser origin and reject unrelated origins

    Args:
        tmp_path:  Dashboard root
        monkeypatch:  Captured job launcher
        origin:  Browser origin
        status:  Expected HTTP response
    """
    calls, responses = [], []
    payload = {"checkpoint": "demo/checkpoints/best.pt"}
    body = json.dumps(payload).encode()

    def launch(root, value):
        """
        Record the host request independently of the browser port
        """
        calls.append((root, value))
        return {"id": "demo/evaluations/test", "status": "queued"}

    monkeypatch.setattr(evaluations, "start_evaluation", launch)
    handler = SimpleNamespace(
        root=tmp_path,
        path="/api/evaluations",
        headers={
            "X-Twistboard-Action": "evaluate",
            "Origin": origin,
            "Host": "127.0.0.1:18765",
            "Content-Length": str(len(body)),
        },
        rfile=io.BytesIO(body),
        send_json=lambda value, code: responses.append((value, code)),
    )
    Handler.do_POST(handler)
    assert responses[-1][1] == status
    assert calls == ([(tmp_path, payload)] if status == 202 else [])
