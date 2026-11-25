# Training and checkpoint persistence
#
# mikael.mieskolainen@cern.ch (2026)

"""Versioned tensor/state-dict checkpoints. no pickled module objects"""

from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import numpy as np
import torch

VERSION = 2

@contextmanager
def finish_checkpoint_writes():
    """
    Defer Ctrl+C until epoch metrics and checkpoints agree

    Returns:
        context: A second interrupt forces immediate cancellation
    """
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGINT) != signal.default_int_handler:
        yield
        return
    interrupted = False

    def defer(signum, frame):
        """
        Record the first interrupt and raise on the second

        Args:
            signum: Signal number
            frame: Interrupted Python frame
        Returns:
            None: The pending interrupt is recorded
        """
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True

    previous = signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if interrupted:
        raise KeyboardInterrupt

def rng_state(device=None):
    """
    Capture all training random generators

    Args:
        device:  Optional rank-local CUDA device
    Returns:
        state: Tensor and primitive RNG state dictionary
    """
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device is None and torch.cuda.is_available() else [],
        "cuda_local": torch.cuda.get_rng_state(device) if device is not None and device.type == "cuda" else None,
    }

def restore_rng(state):
    """
    Restore random generators at an epoch boundary

    Args:
        state:  Checkpoint RNG dictionary
    Returns:
        None: Global generators are restored
    """
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if torch.cuda.is_available() and state.get("cuda_local") is not None:
        torch.cuda.set_rng_state(state["cuda_local"])

def code_version():
    """
    Identify the current Git revision and dirty state
    Returns:
        version: Revision string, or unknown outside Git
    """
    try:
        version = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip()
        return version + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

def save_checkpoint(
    path, experiment, preprocessing, *, epoch, step, best, stale, optimizer, scheduler, scaler, ema=None, rank_rng=None,
    training_elapsed_seconds=None, feature_importance_json=None,
):
    """
    Atomically save a portable training state dictionary

    Args:
        path:  Destination checkpoint file
        experiment:  Configured model, task and driver
        preprocessing:  Fitted driver preprocessor
        epoch:  Completed zero-based epoch
        step:  Optimizer update counter
        best:  Best validation metric
        stale:  Epochs since the last improvement
        optimizer:  Training optimizer
        scheduler:  Optional learning-rate scheduler
        scaler:  AMP gradient scaler
        ema:  Optional exponential parameter average
        rank_rng:  Optional ordered RNG states from every training rank
        training_elapsed_seconds:  Cumulative active training wall time
        feature_importance_json:  Optional ranked tree input gains as JSON text
    Returns:
        None: The complete checkpoint replaces the destination
    """
    path = Path(path)
    config = experiment.config.to_dict()
    if config["driver"]["name"] in {"pi0gamma", "gammapi0"}:
        config["driver"].setdefault("input_mode", experiment.driver.options.input_mode)
    state = {
        "schema_version": VERSION,
        "config": config,
        "input_spec": asdict(experiment.driver.input_spec),
        "task_spec": asdict(experiment.driver.task_spec),
        "model": experiment.model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "inference_weights": "ema" if ema is not None and experiment.config.training.ema_eval else "model",
        "preprocessing": preprocessing.state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "training_elapsed_seconds": training_elapsed_seconds,
        "best": None if best is None else float(best),
        "stale": stale,
        "rng": rank_rng[0] if rank_rng is not None else rng_state(),
        "rank_rng": rank_rng,
        "world_size": len(rank_rng) if rank_rng is not None else 1,
        "torch_version": str(torch.__version__),
        "code_version": code_version(),
    }
    temporary = path.with_suffix(".tmp")
    if feature_importance_json is not None:
        state["feature_importance_json"] = feature_importance_json
    if optimizer is None:
        state["inference_only"] = True
    torch.save(state, temporary)
    os.replace(temporary, path)

def read_checkpoint(path):
    """
    Load a supported checkpoint without pickled model objects

    Args:
        path:  Checkpoint file
    Returns:
        state: Version-checked state dictionary
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != VERSION or "preprocessing" not in state:
        raise ValueError("Unsupported/legacy checkpoint. Use its original Git revision. automatic reinterpretation is unsafe.")
    driver = state.get("config", {}).get("driver", {})
    if driver.get("name") in {"pi0gamma", "gammapi0"}:
        # Older checkpoints with implicit selection used linked cluster members
        driver.setdefault("input_mode", "leading_cluster")
    if state.get("config", {}).get("model", {}).get("family") == "pelican" and any(
        key.startswith("backbone.blocks.") and ".edge." in key for key in state["model"]
    ):
        raise ValueError(
            "This checkpoint uses the former graph model, not the PELICAN pair encoder. "
            "Use its original code revision for inference or start a new PELICAN run"
        )
    return state

def restore_model(experiment, state, *, inference=False):
    """
    Restore strict model weights after checking driver schemas

    Args:
        experiment:  Reconstructed experiment
        state:  Loaded checkpoint dictionary
        inference:  Select the weights used for checkpoint validation
    Returns:
        None: Model parameters and buffers are restored
    """
    if (
        asdict(experiment.driver.input_spec) != state["input_spec"]
        or asdict(experiment.driver.task_spec) != state["task_spec"]
    ):
        raise ValueError("Checkpoint input/task schema does not match the driver")
    weights = state["model"]
    if inference and state.get("inference_weights", "model") == "ema":
        if state.get("ema") is None:
            raise ValueError("Checkpoint selects EMA inference but has no EMA state")
        weights = state["ema"]["model"]
    # Retain the saved binary classifier without changing its logits
    spec = experiment.driver.task_spec
    head = getattr(experiment.model, "head", None)
    saved_weight = weights.get("head.linear.weight")
    if (
        spec.name == "classification"
        and len(spec.outputs) == 2
        and isinstance(getattr(head, "linear", None), torch.nn.Linear)
        and saved_weight is not None
        and saved_weight.shape == (2, head.linear.in_features)
        and head.linear.out_features == 1
    ):
        previous = head.linear
        head.linear = torch.nn.Linear(
            previous.in_features, 2, bias=previous.bias is not None, device=previous.weight.device, dtype=previous.weight.dtype
        ).train(previous.training)
    experiment.model.load_state_dict(weights, strict=True)
