# Active wall time across training continuations and evaluations
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from types import SimpleNamespace

import pytest
from tqdm import tqdm

from twistboard import experiments
from twistboard.experiments import Tee, evaluation_run, training_run
from twistboard.server import run_detail
from twistboard.timing import duration_seconds, format_duration, training_seconds


@pytest.fixture
def clock(monkeypatch):
    """
    Control active time and wall time without real waits

    Args:
        monkeypatch:  Isolated lifecycle clock replacements
    Returns:
        clock: Mutable elapsed seconds and configuration
    """
    clock = SimpleNamespace(seconds=0.0)
    origin = datetime(2026, 10, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(experiments, "time", SimpleNamespace(monotonic=lambda: clock.seconds))
    monkeypatch.setattr(experiments, "timestamp", lambda: (origin + timedelta(seconds=clock.seconds)).isoformat())
    monkeypatch.setattr(experiments, "training_plots", lambda root: None)
    clock.config = {
        "driver": {"name": "synthetic"},
        "model": {"family": "transformer"},
        "task": {"name": "classification"},
    }
    return clock


def test_training_continuation_excludes_offline_time(tmp_path, clock, capsys):
    """
    Retain interrupted work without counting the pause before continuation

    Args:
        tmp_path:  Experiment root
        clock:  Controlled session clock
        capsys:  Captured terminal output
    """
    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        with training_run(root, clock.config) as timer:
            clock.seconds = 12
            timer.persist(force=True)
            raise KeyboardInterrupt
    clock.seconds += 1000
    with training_run(root, clock.config, resume="last.pt") as timer:
        timer.inherit(10)
        clock.seconds += 8
        assert timer.elapsed() == 20
    record = json.loads((root / "experiment.json").read_text())
    assert record["elapsed_seconds"] == 20
    assert [attempt["elapsed_seconds"] for attempt in record["attempts"]] == [12, 8]
    assert run_detail(tmp_path, "run")["training_elapsed_seconds"] == 20
    assert "Total training time: 00:00:20" in capsys.readouterr().out


def test_checkpoint_fork_and_weight_initialization_time(tmp_path, clock):
    """
    Inherit checkpoint time only for continuation, not weight initialization

    Args:
        tmp_path:  Experiment root
        clock:  Controlled session clock
    """
    with training_run(tmp_path / "fork", clock.config, resume="source.pt") as timer:
        clock.seconds += 3
        timer.inherit(100)
        clock.seconds += 2
        assert timer.elapsed() == 105
    with training_run(tmp_path / "fresh", clock.config, init_from="source.pt") as timer:
        clock.seconds += 7
        assert timer.elapsed() == 7
    assert run_detail(tmp_path, "fork")["training_elapsed_seconds"] == 105
    with training_run(tmp_path / "legacy", clock.config, resume="old.pt") as timer:
        timer.inherit(None)
        assert timer.elapsed() is None


def test_legacy_attempt_timing_and_live_snapshot():
    """
    Sum legacy intervals and extend only the current timing snapshot
    """
    record = {"attempts": [
        {"started_at": "2026-10-15T10:00:00+00:00", "finished_at": "2026-10-15T10:10:00+00:00"},
        {"started_at": "2026-10-15T12:00:00+00:00", "finished_at": "2026-10-15T12:20:00+00:00"},
    ]}
    assert training_seconds(record) == 1800
    record["attempts"][-1].pop("finished_at")
    now = "2026-10-15T12:25:00+00:00"
    assert training_seconds(record, active=True, now=now) == 2100
    assert training_seconds(record, stopped_at="2026-10-15T12:20:00+00:00") == 1800
    snapshot = {"elapsed_seconds": 1800, "elapsed_updated_at": "2026-10-15T12:20:00+00:00"}
    assert training_seconds(snapshot, active=True, now=now) == 2100
    assert training_seconds(snapshot, now=now) == 1800
    assert duration_seconds({"elapsed_seconds": None}) is None
    assert format_duration(90061) == "25:01:01"


@pytest.mark.parametrize("interrupted", [False, True])
def test_evaluation_duration_is_saved_on_exit(tmp_path, clock, capsys, interrupted):
    """
    Freeze evaluation duration on completion or interruption

    Args:
        tmp_path:  Evaluation root
        clock:  Controlled session clock
        capsys:  Captured terminal output
        interrupted:  Interrupt inference before completion
    """
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evaluation"
    try:
        with evaluation_run(checkpoint, output, "test"):
            clock.seconds += 65
            if interrupted:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        assert interrupted
    record = json.loads((output / "evaluation.json").read_text())
    assert record["elapsed_seconds"] == 65
    assert record["status"] == ("interrupted" if interrupted else "completed")
    assert duration_seconds(record, now="2026-12-01T00:00:00+00:00") == 65
    assert "Evaluation time: 00:01:05" in capsys.readouterr().out


@pytest.mark.parametrize("interactive", [False, True])
def test_text_log_suppresses_tqdm_but_retains_summaries(interactive):
    """
    Keep progress bars in terminals and plain summaries in text logs

    Args:
        interactive:  The terminal stream has a TTY
    """
    terminal, log = StringIO(), StringIO()
    terminal.isatty = lambda: interactive
    stream = Tee(terminal, log)
    with tqdm(total=2, file=stream, mininterval=0, desc="inference") as progress:
        progress.update(1)
        progress.update(1)
    stream.write("\033[35mepoch 1/1 total=00:01:05\033[0m\n")
    stream.write("RuntimeError: failed\n")
    assert log.getvalue() == "epoch 1/1 total=00:01:05\nRuntimeError: failed\n"
    assert ("inference" in terminal.getvalue()) == interactive
    assert "RuntimeError: failed" in terminal.getvalue()
