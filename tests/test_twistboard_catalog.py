# Scalar event and scan index validation
#
# mikael.mieskolainen@cern.ch (2026)

import json
from pathlib import Path
import pytest
from twistboard.catalog import RunIndex, downsample
from twistboard.events import EventLogger
from twistboard.outputs import write_json, read_history
from twistboard.scans import synthetic_scans


def make_run(root, name="trial", driver="tau3", **metadata):
    """
    Create a minimal managed experiment

    Args:
        root:  Test output directory
        name:  Run identifier
        driver:  Physics driver
        metadata:  Extra run metadata
    Returns:
        path: Run directory
    """
    path = root / name
    path.mkdir()
    write_json(path / "experiment.json", {"driver": driver, "status": "trained", "configuration": {"training": {"lr": 0.001}}, **metadata})
    return path


def test_incremental_metrics_and_partial_lines(tmp_path):
    """
    Retain complete events and discover metrics added during training

    Args:
        tmp_path:  Test output directory
    """
    path = make_run(tmp_path)
    logger = EventLogger(path, "attempt-1", lambda: 12.5)
    logger.log_scalars({"loss": 2}, 1, epoch=1)
    index = RunIndex(tmp_path)
    assert index.query({})["rows"][0]["metrics"]["loss"]["count"] == 1
    offset = index.entries["trial"].offsets["events.jsonl"]
    logger.log_scalars({"loss": 1, "physics/response": float("nan")}, 2, epoch=2)
    logger.log_text("stable", False, 2)
    with (path / "events.jsonl").open("a") as stream:
        stream.write('{"type":"scalar","step":3,"values":{"new":4}')
    index.refresh(force=True)
    row = index.query({})["rows"][0]
    assert row["metrics"]["loss"] == {"latest": 1.0, "min": 1.0, "max": 2.0, "count": 2, "missing": 0, "step": 2}
    assert row["metrics"]["physics/response"]["missing"] == 1
    assert "new" not in row["metrics"]
    assert index.entries["trial"].offsets["events.jsonl"] > offset
    with (path / "events.jsonl").open("a") as stream:
        stream.write('}\n')
    index.refresh(force=True)
    assert "new" in index.query({})["metrics"]
    curve = index.curves(["trial"], "loss", "elapsed_seconds")[0]
    assert curve["points"] == [[12.5, 2], [12.5, 1]]
    assert curve["texts"][0]["values"] == {"stable": False}


def test_index_skips_unchanged_files_and_recovers_replacement(tmp_path, monkeypatch):
    """
    Avoid reopening unchanged event files and reset replaced summaries

    Args:
        tmp_path:  Test output directory
        monkeypatch:  Temporary file-open interception
    """
    path = make_run(tmp_path)
    EventLogger(path).log_scalars({"loss": 4}, 1)
    index = RunIndex(tmp_path)
    index.refresh()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Unchanged file reopened"))
        index.refresh(force=True)
    (path / "events.jsonl").write_text('')
    EventLogger(path).log_scalars({"replacement": 7}, 1)
    index.refresh(force=True)
    assert set(index.query({})["rows"][0]["metrics"]) == {"replacement"}


def test_group_filters_ranking_and_pages(tmp_path):
    """
    Filter driver branches and rank missing metrics after finite trials

    Args:
        tmp_path:  Test output directory
    """
    for i in range(12):
        path = make_run(tmp_path, str(i), driver="tau3" if i < 10 else "top")
        write_json(path / "group.json", {"group_id": "scan", "parameters": {"width": i}})
        EventLogger(path).log_scalars({"loss": i if i != 4 else float("nan")}, 1)
    index = RunIndex(tmp_path)
    result = index.query({"driver": "tau3", "metric": "loss", "mode": "max", "sort": "metric", "offset": "2", "limit": "3", "plot": "1", "axes": "width"})
    assert result["total"] == 10
    assert [row["id"] for row in result["rows"]] == ["7", "6", "5"]
    assert result["branches"][0]["count"] == 10
    result = index.query({"filters": json.dumps({"width": {"min": 3, "max": 6}}), "metric": "loss"})
    assert result["total"] == 4
    result = index.query({"filters": json.dumps({"$metric": {"min": 3, "max": 6}}), "metric": "loss"})
    assert result["total"] == 3


def test_legacy_csv_metrics_and_curves(tmp_path):
    """
    Expose arbitrary legacy CSV metrics beside standard training observables

    Args:
        tmp_path:  Test output directory
    """
    path = make_run(tmp_path)
    (path / "metrics.csv").write_text("epoch,step,custom,loss\n0,10,0.4,2\n1,20,nan,1\n")
    index = RunIndex(tmp_path)
    assert index.query({})["metrics"] == ["custom", "loss"]
    assert index.curves(["trial"], "custom", "epoch")[0]["points"] == [[1, 0.4], [2, None]]
    assert index.curves(["trial"], "custom", "elapsed_seconds")[0]["points"] == []


def test_bounded_downsampling_preserves_spike_and_gap():
    """Preserve isolated extrema and a missing-value gap in reduced curves"""
    points = [[i, 0] for i in range(10000)]
    points[1234][1] = 200
    points[5000][1] = None
    reduced = downsample(points, 100)
    assert len(reduced) <= 100
    assert [1234, 200] in reduced
    assert [5000, None] in reduced
    assert reduced[0] == points[0] and reduced[-1] == points[-1]


def test_synthetic_scan_and_dynamic_history(tmp_path):
    """
    Create reproducible scan fixtures without training model weights

    Args:
        tmp_path:  Test output directory
    """
    manifest = synthetic_scans(tmp_path, trials=3, epochs=8)
    record = json.loads(manifest.read_text())
    index = RunIndex(tmp_path)
    assert index.query({})["total"] == 3
    path = tmp_path / record["trials"][1]
    history = read_history(path)
    assert len(history) == 8
    assert history[-1]["epoch"] == 7
    assert "validation/late_metric" in index.query({})["metrics"]
    assert all(row["synthetic"] for row in index.query({})["rows"])


def test_event_validation(tmp_path):
    """
    Reject invalid steps and text types before writing events

    Args:
        tmp_path:  Test output directory
    """
    logger = EventLogger(tmp_path)
    with pytest.raises(ValueError):
        logger.log_scalars({"loss": 1}, -1)
    with pytest.raises(ValueError):
        logger.log_text("text", object(), 1)
    assert not (tmp_path / "events.jsonl").exists()


def test_resume_keeps_only_checkpoint_ancestry(tmp_path):
    """
    Exclude abandoned optimizer steps while retaining the raw attempt log

    Args:
        tmp_path:  Test output directory
    """
    path = make_run(tmp_path)
    logger = EventLogger(path, "first")
    logger.start_attempt(0)
    logger.log_scalars({"loss": 5}, 1)
    logger.log_scalars({"loss": 0.01}, 2)
    index = RunIndex(tmp_path)
    assert index.query({})["rows"][0]["metrics"]["loss"]["min"] == 0.01
    logger = EventLogger(path, "resumed")
    logger.start_attempt(1)
    logger.log_scalars({"loss": 3}, 2)
    index.refresh(force=True)
    assert index.query({})["rows"][0]["metrics"]["loss"]["min"] == 3
    assert index.curves(["trial"], "loss")[0]["points"] == [[1, 5], [2, 3]]
    assert '0.01' in (path / "events.jsonl").read_text()


def test_scan_dry_run_records_validated_configurations(tmp_path):
    """
    Expand all parameter and seed combinations without launching training

    Args:
        tmp_path:  Test scan root
    """
    from twistboard.scans import run_scan

    card = Path(__file__).resolve().parents[1] / "configs/scans/synthetic.yml"
    manifest = run_scan(card, tmp_path)
    record = json.loads(manifest.read_text())
    assert len(record["trials"]) == 8
    assert len({trial["directory"] for trial in record["trials"]}) == 8
    assert {trial["parameters"]["training.seed"] for trial in record["trials"]} == {11, 12}
    assert all(trial["configuration"]["training"]["epochs"] == 4 for trial in record["trials"])
    assert not list(tmp_path.glob("*/experiment.json"))


def test_missing_and_symlinked_files_do_not_leak(tmp_path):
    """
    Keep linked files outside the run index

    Args:
        tmp_path:  Test output directory
    """
    root = tmp_path / "runs"
    root.mkdir()
    path = make_run(root)
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"type":"scalar","step":1,"values":{"private":42}}\n')
    (path / "events.jsonl").symlink_to(outside)
    index = RunIndex(root)
    assert index.query({})["metrics"] == []
    assert index.curves(["trial"], "private")[0]["points"] == []


def test_aggregate_uses_all_trials_and_reports_seed_spread(tmp_path):
    """
    Aggregate all selected seeds with exact counts and sample deviation

    Args:
        tmp_path:  Test output directory
    """
    for i, value in enumerate([1, 3, 5]):
        path = make_run(tmp_path, str(i))
        write_json(path / "group.json", {"parameters": {"width": 32, "seed": i}})
        EventLogger(path).log_scalars({"loss": value}, 1)
    result = RunIndex(tmp_path).query({"metric": "loss", "plot": "1", "kind": "distribution", "axes": "width"})
    group = result["aggregates"]["groups"][0]
    assert group["count"] == 3
    assert group["mean"] == 3 and group["sd"] == 2
    assert group["minimum"] == 1 and group["maximum"] == 5
    assert group["median"] == 3


def test_fixed_budget_requires_an_exact_logged_step(tmp_path):
    """
    Compare equal optimizer budgets without interpolating absent metrics

    Args:
        tmp_path:  Test output directory
    """
    path = make_run(tmp_path)
    logger = EventLogger(path)
    logger.log_scalars({"loss": 4}, 10)
    logger.log_scalars({"loss": 2}, 20)
    index = RunIndex(tmp_path)
    result = index.query({"metric": "loss", "reduce": "budget", "budget": "10"})
    assert result["rows"][0]["metrics"]["loss"]["budget"] == 4
    result = index.query({"metric": "loss", "reduce": "budget", "budget": "15"})
    assert result["rows"][0]["metrics"]["loss"]["budget"] is None
    assert index.query({})["rows"][0]["metrics"]["loss"]["latest"] == 2


def test_new_attempt_recovers_after_an_incomplete_final_line(tmp_path):
    """
    Retain complete observations when a previous process stopped during a write

    Args:
        tmp_path:  Test output directory
    """
    path = make_run(tmp_path)
    EventLogger(path).log_scalars({"loss": 5}, 1)
    with (path / "events.jsonl").open("a") as stream:
        stream.write('{"type":"scalar"')
    logger = EventLogger(path, "resumed")
    logger.start_attempt(1)
    logger.log_scalars({"loss": 3}, 2)
    index = RunIndex(tmp_path)
    assert index.curves(["trial"], "loss")[0]["points"] == [[1, 5], [2, 3]]
