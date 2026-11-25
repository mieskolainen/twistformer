# Batch timing boundaries and training integration
#
# mikael.mieskolainen@cern.ch (2026)

import csv
from contextlib import closing
from types import SimpleNamespace
import pytest
import torch
from twistformer import Trainer, build_experiment, parse_config
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.training.timing import BatchTiming


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_timing_separates_wait_compute_and_metadata(tmp_path, monkeypatch, device):
    """
    Measure loader stalls and completed steps with bounded synchronization

    Args:
        tmp_path:  Timing output directory
        monkeypatch:  Clock and CUDA completion replacements
        device:  Execution path to check
    """
    clock, syncs = [0.0], []
    monkeypatch.setattr("twistformer.training.timing.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda value: syncs.append(value))
    batch = collate_samples([Sample(torch.ones(n, 8)) for n in (2, 4)])

    class Loader:
        def __len__(self):
            return 3

        def __iter__(self):
            for _ in range(3):
                clock[0] += 0.02
                yield batch

    def persist():
        """
        Simulate a metadata write delay
        """
        clock[0] += 0.03

    timing = BatchTiming(2, torch.device(device), tmp_path / "timing.csv")
    with closing(timing.batches(Loader(), epoch=4)) as batches:
        for actual in batches:
            assert actual is batch
            clock[0] += 0.04
            timing.persist(SimpleNamespace(persist=persist))
    rows = list(csv.DictReader(timing.path.open()))
    assert len(rows) == 2 and len(syncs) == (3 if device == "cuda" else 0)
    for row in rows:
        assert row["epoch"] == "5" and row["events"] == "2"
        assert row["padded_tokens"] == "4" and row["valid_tokens"] == "6"
        assert float(row["padding_fraction"]) == 0.25
        assert float(row["data_wait_ms"]) == pytest.approx(20)
        assert float(row["step_ms"]) == pytest.approx(70)
        assert float(row["metadata_ms"]) == pytest.approx(30)


def test_disabled_timing_has_no_device_wait_or_output(tmp_path, monkeypatch):
    """
    Preserve asynchronous execution when diagnostic timing is disabled
    """
    def forbidden(*args):
        raise AssertionError("Disabled timing must not synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    timing = BatchTiming(0, torch.device("cuda"), tmp_path / "timing.csv")
    inputs = [object(), object()]
    assert list(timing.batches(inputs, epoch=0)) == inputs
    timing.save()
    assert not timing.path.exists()


def test_timing_flushes_last_step_at_training_stop(tmp_path, monkeypatch):
    """
    Retain the final measured batch when the optimizer budget stops an epoch
    """
    monkeypatch.setattr("twistboard.experiments.training_plots", lambda root: None)
    cfg = parse_config({
        "driver": {"name": "synthetic", "train_size": 6, "val_size": 2, "max_tokens": 4},
        "model": {"width": 8, "heads": 2, "depth": 1},
        "training": {"epochs": 1, "max_steps": 1, "batch_size": 2, "device": "cpu", "profile_batches": 5},
    })
    Trainer(tmp_path / "run").fit(build_experiment(cfg))
    rows = list(csv.DictReader((tmp_path / "run/batch_timing_rank0.csv").open()))
    assert len(rows) == 1 and rows[0]["batch"] == rows[0]["epoch"] == "1"
    assert float(rows[0]["step_ms"]) > 0


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_timing_budget(count):
    """
    Reject invalid diagnostic batch counts before training
    """
    with pytest.raises(ValueError, match="profile_batches"):
        parse_config({"driver": {"name": "synthetic"}, "training": {"profile_batches": count}})
