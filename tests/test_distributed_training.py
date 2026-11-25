# Distributed weighted likelihood and checkpoint tests
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
from datetime import timedelta
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from twistformer import Trainer, build_experiment, load_predictor, parse_config
from twistformer.api.config import TrainingConfig
from twistformer.data.batching import collate_samples
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.distributed import (
    DistributedRuntime,
    PaddedTrainingDataset,
    WeightedDistributedSampler,
    distributed_session,
)
from twistformer.training.engine import evaluate_validation, make_loader


def configuration(task="classification", **training):
    """
    Define a small event sample with unequal rank lengths

    Args:
        task:  Classification or Gaussian regression
        training:  Training overrides
    Returns:
        config: Validated experiment configuration
    """
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 9,
                "val_size": 3,
                "test_size": 3,
                "max_tokens": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
            "task": {"name": task},
            "training": {
                "epochs": 2,
                "device": "cpu",
                "optimizer": "sgd",
                "scheduler": "cosine",
                "grad_clip": 0.0,
                "ema": True,
                "batch_size": 2,
                "accumulation": 4,
                "eval_batch_size": 2,
                "save_every": 1,
                **training,
            },
        }
    )


def weighted_experiment(cfg):
    """
    Assign unequal event weights including zero-weight observations

    Args:
        cfg:  Experiment configuration
    Returns:
        experiment: Weighted synthetic experiment
    """
    experiment = build_experiment(cfg)
    original = experiment.driver.dataset

    def dataset(split):
        """
        Generate deterministic weighted events

        Args:
            split:  Dataset split
        Returns:
            samples: Events with unequal likelihood weights
        """
        samples = [
            replace(sample, weight=float(index % 4))
            for index, sample in enumerate(original(split))
        ]
        if split == "val" and cfg.task.name == "classification":
            # Each validation rank sees one class and only the full sample defines AUC
            samples = [replace(sample, target=torch.tensor(index % 2)) for index, sample in enumerate(samples)]
        return samples

    experiment.driver.dataset = dataset
    return experiment


def compile_cpu_graphs(model, cfg):
    """
    Exercise AOTAutograd and the DDP compiler on CPU

    Args:
        model:  DDP model with event records
        cfg:  Training configuration
    Returns:
        model: Compiled forward and backward graphs
    """
    return (
        torch.compile(model, backend="aot_eager", dynamic=True)
        if cfg.compile
        else model
    )


def distributed_worker(
    rank, rendezvous, root, task, static_graph, stochastic, compiled
):
    """
    Train and resume two real Gloo ranks

    Args:
        rank:  Process rank
        rendezvous:  File-store path
        root:  File directory
        task:  Supervised task
        static_graph:  Enable static DDP reduction
        stochastic:  Enable dropout for rank RNG validation
        compiled:  Exercise compiled forward and backward graphs
    Returns:
        None: Shared training files and rank checks are written
    """
    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        cfg = configuration(
            task, strategy="ddp", ddp_static_graph=static_graph, compile=compiled
        )
        if compiled:
            from twistformer.training import engine

            engine.compile_model = compile_cpu_graphs
        if stochastic:
            cfg.model.dropout = 0.2
            cfg.training.token_dropout = 0.2
        experiment = weighted_experiment(cfg)
        Trainer(Path(root) / "full").fit(experiment)
        for parameter in experiment.model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        Trainer(
            Path(root) / "resumed",
            resume=Path(root) / "full/checkpoints/epoch_00000.pt",
        ).fit(weighted_experiment(cfg))
        # Trainer must preserve a process group owned by its caller
        assert dist.is_initialized()
    finally:
        dist.destroy_process_group()


def test_explicit_strategy_and_sampler(monkeypatch):
    """
    Check opt-in DDP and exact physical event coverage

    Args:
        monkeypatch:  Environment override fixture
    Returns:
        None: Strategy and sampler invariants are asserted
    """
    assert TrainingConfig().strategy == "single"
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="Multiple ranks"):
        with distributed_session(TrainingConfig(device="cpu")):
            pass
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(ValueError, match="torchrun"):
        with distributed_session(TrainingConfig(strategy="ddp", device="cpu")):
            pass
    with pytest.raises(ValueError, match="strategy"):
        configuration(strategy="automatic")
    with pytest.raises(ValueError, match="bucket"):
        configuration(ddp_bucket_cap_mb=0)
    samples = weighted_experiment(configuration()).driver.dataset("train")
    seen, weight = [], 0.0
    for rank in range(2):
        sampler = WeightedDistributedSampler(samples, num_replicas=2, rank=rank, seed=7)
        sampler.set_epoch(3)
        for index, padding in sampler:
            if not padding:
                seen.append(index)
        loader = DataLoader(
            PaddedTrainingDataset(samples),
            sampler=sampler,
            batch_size=2,
            collate_fn=collate_samples,
        )
        weight += sum(float(batch.sample_weights.sum()) for batch in loader)
    assert sorted(seen) == list(range(len(samples)))
    assert weight == sum(sample.weight for sample in samples)
    # Validation can have fewer events than ranks and never repeats events
    shards = [
        list(
            make_loader(
                samples[:1],
                configuration().training,
                runtime=DistributedRuntime(torch.device("cpu"), rank, 2, True),
            )
        )
        for rank in range(2)
    ]
    assert len(shards[0]) == 1 and not shards[1]


@pytest.mark.parametrize(
    "task,static_graph,stochastic,compiled",
    [
        ("classification", False, False, False),
        ("gaussian_regression", True, False, False),
        ("classification", False, True, False),
        ("classification", False, False, True),
    ],
)
def test_distributed_weighted_training_and_resume(
    tmp_path, task, static_graph, stochastic, compiled
):
    """
    Compare distributed likelihood updates and portable epoch resume

    Args:
        tmp_path:  Temporary output directory
        task:  Supervised task
        static_graph:  Static DDP reduction switch
        stochastic:  Rank-local dropout switch
        compiled:  Compiled graph switch
    Returns:
        None: Global metrics, model weights and RNG restoration are checked
    """
    mp.spawn(
        distributed_worker,
        args=(
            str(tmp_path / "store"),
            str(tmp_path),
            task,
            static_graph,
            stochastic,
            compiled,
        ),
        nprocs=2,
    )
    full = read_checkpoint(tmp_path / "full/checkpoints/last.pt")
    resumed = read_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert full["world_size"] == 2 and len(full["rank_rng"]) == 2
    assert full["step"] == resumed["step"] == 2
    for key, tensor in full["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][key], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(
            full["ema"]["model"][key], resumed["ema"]["model"][key]
        )
    assert not torch.equal(full["rank_rng"][0]["torch"], full["rank_rng"][1]["torch"])
    assert not (tmp_path / "full/.training.lock").exists()
    metadata = json.loads((tmp_path / "full/manifest.json").read_text())
    assert metadata["world_size"] == 2 and metadata["effective_batch_size"] == 16
    with (tmp_path / "full/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    # A distributed checkpoint remains ordinary single-device inference input
    predictor = load_predictor(tmp_path / "full/checkpoints/last.pt", "cpu")
    from twistformer.data.preprocessing import PreparedDataset

    samples = weighted_experiment(configuration(task)).driver.dataset("val")
    loader = make_loader(
        PreparedDataset(samples, predictor.preprocessor), configuration(task).training
    )
    metrics = evaluate_validation(
        predictor.experiment.model,
        predictor.experiment.task,
        loader,
        torch.device("cpu"),
    )
    for key, value in metrics.items():
        assert float(rows[-1][key]) == pytest.approx(value, rel=2e-6, abs=2e-7)
    if not stochastic:
        cfg = configuration(task, batch_size=9, accumulation=1)
        Trainer(tmp_path / "single").fit(weighted_experiment(cfg))
        single = read_checkpoint(tmp_path / "single/checkpoints/last.pt")
        for key, tensor in full["model"].items():
            torch.testing.assert_close(
                tensor, single["model"][key], atol=2e-6, rtol=2e-5
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torchrun_cli(tmp_path, device):
    """
    Launch the public CLI with two ranks and spawned reader workers

    Args:
        tmp_path:  Temporary output directory
        device:  CPU Gloo or two-device CUDA NCCL
    Returns:
        None: Launch, partial windows and primary-only files are checked
    """
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "twistformer.cli.train",
        "--config",
        "configs/synthetic/regression.yml",
        "--strategy",
        "ddp",
        "--device",
        device,
        "--workers",
        "1",
        "--output",
        str(tmp_path / "run"),
    ]
    overrides = {
        "driver.train_size": 9,
        "driver.val_size": 3,
        "driver.max_tokens": 4,
        "model.width": 8,
        "model.heads": 2,
        "model.depth": 1,
        "training.epochs": 1,
        "training.batch_size": 2,
        "training.accumulation": 2,
        "training.amp": "true",
        "training.amp_dtype": "fp16" if device == "cuda" else "bf16",
        "training.fused_optimizer": "true" if device == "cuda" else "false",
    }
    for key, value in overrides.items():
        command.extend(["--set", f"{key}={value}"])
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Best checkpoint:") == 1
    state = read_checkpoint(tmp_path / "run/checkpoints/last.pt")
    assert state["world_size"] == 2 and state["step"] == 2
    record = json.loads((tmp_path / "run/experiment.json").read_text())
    assert record["status"] == "trained" and len(record["attempts"]) == 1
