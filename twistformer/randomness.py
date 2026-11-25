# Random seeds for event selection and model training
#
# mikael.mieskolainen@cern.ch (2026)

import random

import numpy as np
import torch


def seed_python_numpy(seed):
    """
    Seed Python and the NumPy global generator

    Args:
        seed:  Run or worker seed
    """
    seed %= 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_all(seed, *, cpu_only=False):
    """
    Seed Python, NumPy and Torch global generators

    Args:
        seed:  Run or worker seed
        cpu_only:  Set only the Torch CPU generator for workers and cached augmentation
    """
    seed_python_numpy(seed)
    if cpu_only:
        torch.random.default_generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def numpy_generator(seed):
    """
    Create an independent NumPy generator

    Args:
        seed:  Seed or existing NumPy generator
    Returns:
        generator: NumPy generator for local event draws
    """
    return np.random.default_rng(seed)


def torch_generator(seed, *, device="cpu"):
    """
    Create an independent Torch generator

    Args:
        seed:  Seed for local event draws
        device:  Torch generator device
    Returns:
        generator: Seeded Torch generator
    """
    return torch.Generator(device=device).manual_seed(seed)
