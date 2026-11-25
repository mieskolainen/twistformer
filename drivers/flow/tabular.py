# Tabular density benchmarks
#
# mikael.mieskolainen@cern.ch (2026)
# Adapted from Supraflow benchmark readers (2025)

from pathlib import Path
import numpy as np
from twistformer.randomness import numpy_generator


def split_tail(data):
    """
    Select fixed validation and test tails without normalization

    Args:
        data:  Ordered observation matrix
    Returns:
        result: Training, validation and test matrices
    """
    test_count = max(1, int(0.1 * len(data)))
    train = data[:-test_count]
    validation_count = max(1, int(0.1 * len(train)))
    if len(train) <= validation_count:
        raise ValueError("Benchmark has too few events for three splits")
    return train[:-validation_count], train[-validation_count:], data[-test_count:]


def load_benchmark(name, root, split, seed=101):
    """
    Read raw benchmark splits with training-only feature selection

    Args:
        name:  Benchmark name
        root:  Parent data directory
        split:  Training, validation or test
        seed:  Reproducible POWER noise and ordering seed
    Returns:
        output: Raw floating observation matrix
    """
    root = Path(root)
    if name == "BSDS300":
        import h5py

        with h5py.File(root / "BSDS300/BSDS300.hdf5", "r") as file:
            return np.asarray(file[{"train": "train", "val": "validation", "test": "test"}[split]], dtype=np.float32)
    if name in {"POWER", "MINIBOONE"}:
        data = np.load(root / name.lower() / "data.npy", allow_pickle=False)
        if name == "POWER":
            generator = numpy_generator(seed)
            data = data[generator.permutation(len(data))]
            data = data[:, [index for index in range(data.shape[1]) if index not in {1, 3}]].copy()
            if data.shape[1] != 6:
                raise ValueError("POWER requires six retained measurement columns")
            data = data + generator.random(data.shape) * np.array([0.001, 0.01, 1, 1, 1, 0])
        parts = split_tail(data)
    elif name == "GAS":
        import pandas as pd

        data = pd.read_pickle(root / "gas/ethylene_CO.pickle")
        data = data.drop(columns=["Meth", "Eth", "Time"])
        parts = split_tail(data.to_numpy())
        indices = list(range(data.shape[1]))
        while len(indices) > 1:
            correlation = np.corrcoef(parts[0][:, indices], rowvar=False)
            repeated = np.flatnonzero((correlation > 0.98).sum(1) > 1)
            if not len(repeated):
                break
            indices = [index for position, index in enumerate(indices) if position != repeated[0]]
        parts = tuple(values[:, indices] for values in parts)
    elif name == "HEPMASS":
        import pandas as pd

        train = pd.read_csv(root / "hepmass/1000_train.csv").to_numpy()
        test = pd.read_csv(root / "hepmass/1000_test.csv").to_numpy()
        train = train[train[:, 0] == 1, 1:]
        test = test[test[:, 0] == 1, 1:]
        if test.shape[1] == train.shape[1] + 1:
            test = test[:, :-1]
        if train.shape[1] != test.shape[1]:
            raise ValueError("HEPMASS train and test feature counts differ")
        validation_count = max(1, int(0.1 * len(train)))
        train, validation = train[:-validation_count], train[-validation_count:]
        keep = [index for index, column in enumerate(train.T) if np.unique(column, return_counts=True)[1].max() <= 5]
        if not keep:
            raise ValueError("HEPMASS feature selection retained no continuous coordinates")
        parts = train[:, keep], validation[:, keep], test[:, keep]
    else:
        raise ValueError(f"Unknown tabular benchmark {name}")
    return np.asarray(parts[{"train": 0, "val": 1, "test": 2}[split]], dtype=np.float32)
