# Deterministic event loading and persistent workers
#
# mikael.mieskolainen@cern.ch (2026)

import signal
import torch
from torch.utils.data import Dataset, get_worker_info
from twistformer.randomness import seed_all, seed_python_numpy, torch_generator

def fetch_samples(dataset, indices):
    """
    Fetch an ordered event batch through the standard dataset interface

    Args:
        dataset:  Indexed event dataset
        indices:  Event indices, including possible repeats
    Returns:
        samples: Events in the requested order
    """
    fetch = getattr(dataset, "__getitems__", None)
    return fetch(indices) if fetch else [dataset[index] for index in indices]

def seed_worker(worker_id):
    """
    Seed NumPy and Python from the PyTorch worker seed

    Args:
        worker_id:  Worker identifier supplied by DataLoader
    """
    if get_worker_info() is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed_python_numpy(torch.initial_seed())

def close_loader(loader):
    """
    Close persistent event workers before leaving the training lifecycle

    Args:
        loader: Training or validation DataLoader
    Returns:
        None: Worker processes and loader queues are closed
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        # PyTorch exposes worker shutdown through its multiprocessing iterator
        iterator._shutdown_workers()
        loader._iterator = None

class EpochDataset(Dataset):
    """
    Reseed persistent workers when the parent advances the epoch
    """

    def __init__(self, dataset, seed):
        """
        Share the epoch counter with reader workers

        Args:
            dataset:  Prepared event dataset
            seed:  Rank-specific base seed
        """
        self.dataset, self.seed = dataset, seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.worker_epoch = None

    def __len__(self):
        """
        Return the event count
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Fetch one event with the current worker seed
        """
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """
        Fetch a batch after applying the epoch seed in each worker

        Args:
            indices:  Ordered event indices
        Returns:
            samples: Prepared events with deterministic augmentation
        """
        epoch = int(self.epoch)
        worker = get_worker_info()
        if worker is not None and self.worker_epoch != epoch:
            generator = torch_generator(self.seed + epoch)
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
            seed_all(seed + worker.id, cpu_only=True)
            self.worker_epoch = epoch
        return fetch_samples(self.dataset, indices)

def set_loader_epoch(loader, epoch):
    """
    Reset event order and worker seeds without replacing the loader

    Args:
        loader:  Shared training or evaluation DataLoader
        epoch:  Zero-based epoch index
    """
    seed = loader.epoch_seed + epoch
    loader.generator.manual_seed(seed)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    elif getattr(loader.sampler, "generator", None) is not None:
        loader.sampler.generator.manual_seed(seed)
    if isinstance(loader.dataset, EpochDataset):
        loader.dataset.epoch.fill_(epoch)
