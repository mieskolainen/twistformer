# Distributed event sampling and gradient synchronization
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import inspect
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler
from twistformer.data.loading import fetch_samples

@dataclass
class DistributedRuntime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def primary(self):
        """
        Identify the rank that writes output files

        Returns:
            primary: True on rank zero
        """
        return self.rank == 0

    def sum(self, values):
        """
        Sum weighted event statistics across ranks

        Args:
            values:  Sequence of local scalar statistics
        Returns:
            totals: Global scalar sums
        """
        if not self.enabled:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor.tolist()

    def sum_tensor(self, values):
        """
        Sum device statistics across ranks without a host transfer

        Args:
            values:  Statistics tensor on the runtime device
        Returns:
            totals: Detached global sums without changing the input
        """
        totals = values.detach().clone()
        if self.enabled:
            dist.all_reduce(totals)
        return totals

    def on_primary(self, function):
        """
        Execute shared file operations and propagate failures

        Args:
            function:  Rank-zero callable returning a serializable result
        Returns:
            result: Rank-zero result on every rank
        """
        if not self.enabled:
            return function()
        payload = [None]
        error = None
        if self.primary:
            try:
                payload[0] = (True, function(), False)
            except (Exception, KeyboardInterrupt) as caught:
                error = caught
                payload[0] = (False, f"{type(caught).__name__}: {caught}", isinstance(caught, KeyboardInterrupt))
        try:
            dist.broadcast_object_list(payload, src=0)
        except Exception:
            if isinstance(error, KeyboardInterrupt):
                raise error from None
            raise
        success, result, interrupted = payload[0]
        if interrupted:
            raise KeyboardInterrupt
        if not success:
            if error is not None:
                raise error
            raise RuntimeError(f"Rank zero failed: {result}")
        return result

    def sync_buffers(self, model):
        """
        Select rank-zero running statistics before unwrapped validation

        Args:
            model:  Replicated model or EMA shadow
        Returns:
            None: Buffers agree across ranks
        """
        if self.enabled:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

    def gather_on_primary(self, value):
        """
        Gather rank-local data in rank order

        Args:
            value:  Local serializable value
        Returns:
            values:  Ordered values on rank zero, otherwise None
        """
        if not self.enabled:
            return [value]
        values = [None] * self.world_size if self.primary else None
        dist.gather_object(value, values, dst=0)
        return values

    gather_rng = gather_on_primary

    def wrap(self, model, cfg):
        """
        Wrap the event model with bucket-view gradient reduction

        Args:
            model:  Device-local event model
            cfg:  Training configuration
        Returns:
            model: DDP wrapper or original single-device model
        """
        if not self.enabled:
            return model
        options = {
            "device_ids": [self.device.index] if self.device.type == "cuda" else None,
            "gradient_as_bucket_view": True,
            "bucket_cap_mb": cfg.ddp_bucket_cap_mb,
            "find_unused_parameters": cfg.ddp_find_unused_parameters and not cfg.ddp_static_graph,
            "static_graph": cfg.ddp_static_graph,
        }
        # PyTorch 2.14 batches gradient copies into communication buckets
        if "batched_grad_copy" in inspect.signature(DistributedDataParallel).parameters:
            options["batched_grad_copy"] = True
        return DistributedDataParallel(model, **options)

@contextmanager
def distributed_session(cfg):
    """
    Bind torchrun ranks to devices and own the process-group lifetime

    Args:
        cfg:  Training configuration with explicit single or ddp strategy
    Returns:
        runtime: Device, rank and collective operations
    """
    from .engine import device_for

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized = dist.is_available() and dist.is_initialized()
    if cfg.strategy == "single":
        if world_size > 1 or (initialized and dist.get_world_size() > 1):
            raise ValueError("Multiple ranks require --strategy ddp or training.strategy=ddp")
        yield DistributedRuntime(device_for(cfg.device))
        return
    if not dist.is_available():
        raise RuntimeError("This Torch build has no distributed support")
    if not initialized and not all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise ValueError("DDP requires torchrun, for example torchrun --standalone --nproc-per-node=2")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = device_for(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError("Each local DDP rank requires a visible CUDA device")
        if device.index is not None:
            raise ValueError("DDP uses --device cuda and LOCAL_RANK. Select GPUs with CUDA_VISIBLE_DEVICES")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("DDP supports cuda or cpu devices")
    backend = "nccl" if device.type == "cuda" else "gloo"
    if initialized and dist.get_backend() != backend:
        raise ValueError(f"DDP on {device.type} requires the {backend} backend")
    if not initialized:
        options = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30), **options)
    try:
        yield DistributedRuntime(device, dist.get_rank(), dist.get_world_size(), True)
    finally:
        if not initialized:
            dist.destroy_process_group()

class PaddedTrainingDataset(Dataset):
    def __init__(self, dataset):
        """
        Retain equal rank lengths without adding event weight

        Args:
            dataset:  Prepared training samples
        Returns:
            None: Wrapped dataset is stored
        """
        self.dataset = dataset

    def __len__(self):
        """
        Return the physical event count

        Returns:
            count: Number of training events
        """
        return len(self.dataset)

    def __getitem__(self, key):
        """
        Assign zero weight to sampler padding

        Args:
            key:  Event index and padding flag
        Returns:
            sample: Prepared event with physical or zero weight
        """
        index, padding = key
        sample = self.dataset[index]
        return replace(sample, weight=0.0) if padding else sample

    def __getitems__(self, keys):
        """
        Fetch rank-local batches and remove the weight of sampler padding

        Args:
            keys:  Event indices paired with padding flags
        Returns:
            samples: Ordered events with zero weight for padding
        """
        samples = fetch_samples(self.dataset, [index for index, _ in keys])
        return [replace(sample, weight=0.0) if padding else sample for sample, (_, padding) in zip(samples, keys, strict=True)]

class WeightedDistributedSampler(DistributedSampler):
    def __iter__(self):
        """
        Flag repeated tail events needed for synchronized backward calls

        Returns:
            indices: Event indices paired with padding flags
        """
        for position, index in enumerate(super().__iter__()):
            padding = self.rank + position * self.num_replicas >= len(self.dataset)
            yield index, padding
