# Distributed flow objectives and deterministic resume
#
# mikael.mieskolainen@cern.ch (2026)

from datetime import timedelta
from pathlib import Path
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from twistformer import parse_config, build_experiment, Trainer, load_predictor
from twistformer.training.checkpoint import read_checkpoint


def worker(rank, directory, flow_type, task):
    """
    Train and resume a flow across two CPU ranks

    Args:
        rank:  Process rank
        directory:  Shared temporary directory
        flow_type:  Continuous or discrete model
        task:  Matching or likelihood objective
    Returns:
        None: Portable model and RNG checkpoints are written
    """
    torch.set_num_threads(1)
    root = Path(directory)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", init_method=f"file://{root / 'store'}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        config = parse_config({"driver": {"name": "flow", "train_size": 9, "val_size": 3, "test_size": 3},
                               "task": {"name": task},
                               "flow": {"type": flow_type, "width": 8, "depth": 1, "layers": 1,
                                        "steps": 2, "trace": "exact"},
                               "training": {"strategy": "ddp", "device": "cpu", "epochs": 2, "batch_size": 2,
                                            "accumulation": 2, "ema": True, "save_every": 1, "scheduler": "none"}})
        experiment = build_experiment(config)
        Trainer(root / "full").fit(experiment)
        for parameter in experiment.model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        Trainer(root / "resumed", resume=root / "full/checkpoints/epoch_00000.pt").fit(build_experiment(config))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("flow_type,task", [("continuous", "flow_matching"), ("continuous", "flow_nll"), ("discrete", "flow_nll")])
def test_distributed_flow_resume(tmp_path, flow_type, task):
    """
    Verify rank reduction, EMA and RNG restoration for native flows

    Args:
        tmp_path:  Local rendezvous and run files
        flow_type:  Flow architecture
        task:  Training objective
    """
    mp.spawn(worker, args=(str(tmp_path), flow_type, task), nprocs=2)
    full = read_checkpoint(tmp_path / "full/checkpoints/last.pt")
    resumed = read_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert full["world_size"] == 2 and len(full["rank_rng"]) == 2
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], atol=0, rtol=0)
        torch.testing.assert_close(full["ema"]["model"][key], resumed["ema"]["model"][key], atol=0, rtol=0)
    predictor = load_predictor(tmp_path / "full/checkpoints/last.pt", "cpu")
    assert predictor.sample(events=3).shape == (3, 1, 2)
