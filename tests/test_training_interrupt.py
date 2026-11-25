# Training interruption and worker cleanup
#
# mikael.mieskolainen@cern.ch (2026)

import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from twistformer import Trainer, build_experiment, parse_config
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.distributed import DistributedRuntime


def children(pid):
    """
    Collect descendants of one Linux test process

    Args:
        pid: Parent process identifier
    Returns:
        pids: Descendant process identifiers
    """
    try:
        direct = [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []
    return direct + [child for parent in direct for child in children(parent)]


@pytest.mark.skipif(sys.platform != "linux", reason="Process group checks use Linux")
@pytest.mark.parametrize("workflow,workers", [(False, 0), (False, 2), (True, 2)])
def test_ctrl_c_closes_training_and_workers(tmp_path, workflow, workers):
    """
    Send a real terminal-group interrupt after a completed training checkpoint

    Args:
        tmp_path: Isolated run directory
        workflow: Launch through the Twistboard workflow when true
        workers: Persistent reader count
    Returns:
        None: Interrupted metadata, valid checkpoints and closed workers are verified
    """
    base = [sys.executable, "-u", "-m"]
    if workflow:
        command = base + ["twistboard", "run", "--root", str(tmp_path), "--name", "interrupt", "--no-evaluate"]
    else:
        command = base + ["twistformer.cli.train", "--output", str(tmp_path / "run")]
    command += ["--config", "configs/synthetic/classification.yml", "--device", "cpu", "--workers", str(workers)]
    for key, value in {
        "driver.train_size": 64, "driver.val_size": 16, "driver.max_tokens": 4,
        "model.width": 8, "model.heads": 2, "model.depth": 1,
        "training.epochs": 10000, "training.batch_size": 8,
    }.items():
        command += ["--set", f"{key}={value}"]
    log = tmp_path / "terminal.log"
    descendants = []
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        try:
            deadline = time.monotonic() + 40
            checkpoint = None
            while time.monotonic() < deadline:
                checkpoint = next(tmp_path.glob("*/checkpoints/last.pt"), None)
                if checkpoint is not None or process.poll() is not None:
                    break
                time.sleep(0.05)
            assert checkpoint is not None, log.read_text()
            descendants = children(process.pid)
            if workers:
                assert len(descendants) >= workers
            os.killpg(process.pid, signal.SIGINT)
            assert process.wait(timeout=20) == 130, log.read_text()
            remaining = [pid for pid in descendants if Path(f"/proc/{pid}").exists()]
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            for pid in descendants:
                if Path(f"/proc/{pid}").exists():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    root = checkpoint.parent.parent
    record = json.loads((root / "experiment.json").read_text())
    assert record["status"] == "interrupted"
    assert record["attempts"][-1]["status"] == "interrupted"
    assert record["attempts"][-1]["finished_at"]
    assert record["elapsed_seconds"] > 0
    assert not (root / ".training.lock").exists()
    assert not (root / "plots").exists()
    state = read_checkpoint(checkpoint)
    assert state["epoch"] >= 0 and state["step"] > 0
    assert not remaining
    output = log.read_text()
    assert "Training interrupted" in output
    assert "Traceback" not in output
    assert "Exception ignored" not in output


def test_interrupt_during_checkpoint_can_resume(tmp_path, monkeypatch):
    """
    Interrupt an epoch write and resume with matching metrics and optimizer state

    Args:
        tmp_path: Isolated run directory
        monkeypatch: Checkpoint interruption injection
    Returns:
        None: The saved epoch remains resumable
    """
    from twistformer.training import engine

    config = parse_config({
        "driver": {"name": "synthetic", "train_size": 8, "val_size": 4, "max_tokens": 4},
        "model": {"width": 8, "heads": 2, "depth": 1},
        "training": {"epochs": 2, "batch_size": 4, "device": "cpu"},
    })
    save = engine.save_checkpoint
    handler = signal.getsignal(signal.SIGINT)

    def interrupt_save(*args, **kwargs):
        """
        Deliver Ctrl+C while an epoch checkpoint is being written

        Args:
            args: Checkpoint positional arguments
            kwargs: Checkpoint state arguments
        Returns:
            None: The checkpoint write completes before interruption
        """
        signal.raise_signal(signal.SIGINT)
        save(*args, **kwargs)

    monkeypatch.setattr(engine, "save_checkpoint", interrupt_save)
    monkeypatch.setattr("twistboard.experiments.training_plots", lambda root: None)
    with pytest.raises(KeyboardInterrupt):
        Trainer(tmp_path).fit(build_experiment(config))
    assert signal.getsignal(signal.SIGINT) == handler
    checkpoint = tmp_path / "checkpoints/last.pt"
    state = read_checkpoint(checkpoint)
    with (tmp_path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert int(rows[-1]["epoch"]) == state["epoch"] == 0
    assert json.loads((tmp_path / "experiment.json").read_text())["status"] == "interrupted"
    monkeypatch.setattr(engine, "save_checkpoint", save)
    Trainer(tmp_path, resume=checkpoint).fit(build_experiment(config))
    assert read_checkpoint(checkpoint)["epoch"] == 1
    assert json.loads((tmp_path / "experiment.json").read_text())["status"] == "trained"


def interrupted_rank(rank, root):
    """
    Interrupt rank zero during a shared file operation

    Args:
        rank: Distributed worker rank
        root: Rendezvous and result directory
    Returns:
        None: Every rank records its interrupt and closes its process group
    """
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        runtime = DistributedRuntime(torch.device("cpu"), rank=rank, world_size=2, enabled=True)
        try:
            runtime.on_primary(lambda: signal.raise_signal(signal.SIGINT))
        except KeyboardInterrupt:
            (root / f"interrupted-{rank}").touch()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_primary_interrupt_reaches_all_ranks(tmp_path):
    """
    Check that rank-zero interruption releases waiting CPU ranks

    Args:
        tmp_path: Rendezvous and result directory
    Returns:
        None: Both ranks exit without a blocked collective
    """
    context = mp.spawn(interrupted_rank, args=(tmp_path,), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 30
        while not context.join(timeout=0.2):
            assert time.monotonic() < deadline, "Distributed interrupt did not close all ranks"
        assert all((tmp_path / f"interrupted-{rank}").exists() for rank in range(2))
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
