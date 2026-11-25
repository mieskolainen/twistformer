# Twistformer
#
# mikael.mieskolainen@cern.ch (2026)

"""Public API for training and inference. Driver I/O is imported lazily"""

import os
import shutil
from multiprocessing import current_process
import torch

__version__ = "0.0.2"

def _system_summary() -> str:
    """
    Collect the host resources and visible CUDA devices

    Returns:
        summary:  system information for the startup banner
    """
    gib = 1024 ** 3
    ram = "unavailable"
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            ram = f"{pages * page_size / gib:.1f} GiB total"
    except (AttributeError, OSError, ValueError):
        pass

    try:
        disk = shutil.disk_usage(".")
        disk_space = f"{disk.free / gib:.1f} GiB / {disk.total / gib:.1f} GiB"
    except OSError:
        disk_space = "unavailable"
    
    devices = []
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for index in range(device_count):
        device = f"cuda:{index}"
        try:
            device += f" ({torch.cuda.get_device_name(index)})"
        except (RuntimeError, AssertionError, OSError):
            pass
        devices.append(device)

    return (
        f"PyTorch: {torch.__version__} | "
        f"Logical CPUs: {os.cpu_count() or 'unavailable'} | "
        f"RAM: {ram} | "
        f"Disk: {disk_space}\n"
        f"CUDA devices: {', '.join(devices) or 'none'}"
    )

# CPU reader processes import the package without initializing CUDA
if current_process().name == "MainProcess":
    print(
        "\n"
        f"TWISTFORMER {__version__}\n"
        "github.com/mieskolainen/twistformer (MIT License, 2026)\n"
        f"{_system_summary()}"
        "\n"
        ,
        flush=True,
    )

from .api.config import Config, load_config, parse_config
from .api.contracts import InputSpec, TaskSpec, Sample, Batch, Prediction
from .api.experiment import build_experiment
from .api.predictor import load_predictor
from .api.export import export_onnx
from .training.engine import Trainer
from .api.reweighting import ClassifierReweighter, FoldReweighter, WeightedDataset, fit_reweighter

__all__ = [
    "Config",
    "load_config",
    "parse_config",
    "InputSpec",
    "TaskSpec",
    "Sample",
    "Batch",
    "Prediction",
    "build_experiment",
    "load_predictor",
    "export_onnx",
    "Trainer",
    "ClassifierReweighter",
    "FoldReweighter",
    "WeightedDataset",
    "fit_reweighter",
]
