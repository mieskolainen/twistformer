# Training augmentation stages
#
# mikael.mieskolainen@cern.ch (2026)

import random
import numpy as np
import torch
from tqdm import tqdm
from twistformer.randomness import seed_all
from .preprocessing import PreparedDataset


def prepare_training_dataset(dataset, preprocessor, augment, config, *, augment_dataset=None, progress=False):
    """
    Select cached, online or disabled driver augmentation

    Args:
        dataset:  Raw training events in stable order
        preprocessor:  Normalization fitted on unaugmented training events
        augment:  Driver callable from one raw sample to an augmented sample
        config:  Training settings with augmentation mode and seed
        augment_dataset:  Optional driver transform for the full training split
        progress:  Show cache construction on the primary rank
    Returns:
        prepared: Training dataset with the selected augmentation stage
    """
    if config.augmentation == "online":
        return PreparedDataset(dataset, preprocessor, augment)
    if config.augmentation == "none":
        return PreparedDataset(dataset, preprocessor)
    if config.augmentation != "dataset":
        raise ValueError("training.augmentation must be dataset, online or none")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        # Fixed draws across ranks and resume, independent of training RNG streams
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            seed_all(config.seed, cpu_only=True)
            if augment_dataset is not None:
                return PreparedDataset(augment_dataset(dataset), preprocessor)
            samples = [
                augment(dataset[index])
                for index in tqdm(
                    range(len(dataset)), desc="Caching training augmentation", unit="event",
                    dynamic_ncols=True, disable=not progress,
                )
            ]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return PreparedDataset(samples, preprocessor)
