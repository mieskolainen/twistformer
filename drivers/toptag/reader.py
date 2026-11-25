# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from time import monotonic

import numpy as np
import torch
from torch.utils.data import Dataset
from twistformer.randomness import numpy_generator
from twistformer.api.contracts import ClassificationRows, Sample
from twistformer.logging import print
from .features import constituent_features
from .graphs import BASES, jet_graph_kernel
from .graphs.lund_graphs import NATIVE_NAMES, native_top_rows
from twistformer.geometry.native_lund import cpu_workers

def read_columns(path, limit, seed, start=0):
    """
    Read numeric top-tag columns and source indices directly from HDF5

    Args:
        path:  HDF5 file with a pandas table or fixed numeric frame at /table
        limit:  Maximum uniformly sampled events, or -1 for all events
        seed:  Event sampling seed
        start: First eligible physical row before sampling
    Returns:
        data: Column mapping and original event indices
    """
    import h5py

    try:
        # Register compression filters with the HDF5 library used by h5py
        import hdf5plugin  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "hdf5plugin":
            raise
        raise ImportError(
            "Top-tag HDF5 compression support requires hdf5plugin. Install it with: python -m pip install hdf5plugin"
        ) from error

    print(f"Opening top-tag HDF5 file {path}")
    with h5py.File(path, "r") as store:
        group = store["table"]
        table = group.get("table")
        count = table.shape[0] if table is not None else group["axis1"].shape[0]
        if start < 0 or (start and start >= count):
            raise ValueError(f"Invalid start row {start} for {count} available jet rows")
        eligible = count - start
        rows = slice(start, count)
        size = eligible
        if 0 < limit < eligible:
            rows = start + np.sort(numpy_generator(seed).choice(eligible, limit, replace=False))
            size = limit
        print(f"Reading {size:,} of {count:,} jets from {path}")
        columns = {}
        if table is not None:
            values = _read_rows(table, rows)
            indices = values["index"]
            for field in table.dtype.names:
                if field == "index":
                    continue
                names = _column_names(table.attrs[field + "_kind"]) if field + "_kind" in table.attrs else [field]
                block = values[field]
                if block.ndim == 1:
                    block = block[:, None]
                if block.ndim != 2 or len(names) != block.shape[1]:
                    raise ValueError(f"Column metadata does not match HDF5 field {field!r}")
                for index, name in enumerate(names):
                    columns[name] = block[:, index]
        else:
            indices = _read_rows(group["axis1"], rows)
            for name, dataset in group.items():
                if not name.startswith("block") or not name.endswith("_items"):
                    continue
                names = [_label(value) for value in dataset[:]]
                block = _read_rows(group[name.replace("_items", "_values")], rows)
                if block.ndim != 2 or len(names) != block.shape[1]:
                    raise ValueError(f"Column metadata does not match HDF5 block {name!r}")
                for index, column in enumerate(names):
                    columns[column] = block[:, index]
    return columns, indices

def _label(value):
    """
    Decode an HDF5 column label

    Args:
        value:  UTF-8 bytes or string label
    Returns:
        label: Column name
    """
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

def _column_names(value):
    """
    Decode pandas column-name metadata without importing serialized classes

    Args:
        value:  Pickled string list or native string array from an HDF5 attribute
    Returns:
        names: Validated column labels
    """
    import io
    import pickle

    class ColumnUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            """
            Reject executable objects in column metadata

            Args:
                module:  Serialized module reference
                name:  Serialized object reference
            Returns:
                None: Object reconstruction is rejected
            """
            raise ValueError("HDF5 column metadata must contain only string labels")

    if isinstance(value, (bytes, str, np.void)):
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        try:
            value = ColumnUnpickler(io.BytesIO(payload), encoding="utf-8").load()
        except (pickle.UnpicklingError, EOFError, UnicodeError, ValueError) as error:
            raise ValueError("Invalid pandas HDF5 column-name metadata") from error
    if not isinstance(value, (list, tuple, np.ndarray)) or not all(isinstance(name, (str, bytes)) for name in value):
        raise ValueError("HDF5 column metadata must be a sequence of string labels")
    return [_label(name) for name in value]

def _read_rows(dataset, rows):
    """
    Read selected events in bounded contiguous blocks

    Args:
        dataset:  Numeric HDF5 dataset with event axis first
        rows:  Contiguous slice or sorted unique event indices
    Returns:
        values: Requested rows in source order
    """
    if isinstance(rows, slice):
        return dataset[rows]
    if rows is None:
        return dataset[:]
    output = np.empty((len(rows), *dataset.shape[1:]), dtype=dataset.dtype)
    if not len(rows):
        return output
    row_bytes = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:], dtype=np.int64))
    step = max(1, (8 * 1024**2) // max(1, row_bytes))
    groups = rows // step
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(rows)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = int(groups[left] * step)
        block = dataset[start : min(start + step, dataset.shape[0])]
        output[left:right] = block[rows[left:right] - start]
    return output

class Jets(Dataset):
    def __init__(self, path, options, start=0):
        columns, indices = read_columns(path, options.max_events, options.seed, start)
        print(f"Packing constituent four-momenta for {len(indices):,} jets from {path}")
        components = [
            np.stack([columns[f"{name}_{i}"] for i in range(200)], axis=-1).astype(np.float32) for name in ("E", "PX", "PY", "PZ")
        ]
        energies = components[0]
        particle_mask = np.isfinite(energies) & (energies > 0)
        valid = particle_mask.any(axis=1)
        weights = (
            np.asarray(columns[options.weight_column], dtype=float) if options.weight_column is not None else np.ones(len(indices))
        )
        if weights.shape != (len(indices),) or not np.isfinite(weights).all():
            raise ValueError("Event weights must be finite scalars")
        self.weights = weights[valid]
        self.ids = [f"{path}:{index}" for index in indices[valid]]
        self.p4 = torch.from_numpy(np.stack(components, axis=-1)[particle_mask])
        self.labels = torch.from_numpy(np.asarray(columns["is_signal_new"], dtype=np.int64)[valid])
        self.offsets = np.concatenate([[0], np.cumsum(particle_mask[valid].sum(axis=1))])
        self.max_particles = options.max_particles
        self.feature_set = options.feature_set
        print(f"Loaded {len(self):,} nonempty jets from {path}")

    def __len__(self):
        return len(self.labels)

    def graph_inputs(self, start, stop, batch_size):
        """
        Gather leading constituents and full-jet observables for graph batches

        Args:
            start:       First jet in source order
            stop:        Exclusive upper jet index
            batch_size:  Pad the final block to a multiple of this size
        Returns:
            selected:     Zero-padded leading constituent four-momenta
            counts:       Number of retained constituents per jet
            observables:  Detector quantities before truncation
        """
        offsets = torch.from_numpy(self.offsets[start:stop + 1] - self.offsets[start])
        lengths = offsets.diff()
        width = min(self.max_particles, 200)
        padded = ((len(lengths) + batch_size - 1) // batch_size) * batch_size
        selected = self.p4.new_zeros(padded, width, 4)
        counts = torch.zeros(padded, dtype=torch.long)
        counts[:len(lengths)] = lengths.clamp_max(width)
        block = self.p4[self.offsets[start]:self.offsets[stop]]
        totals = torch.segment_reduce(block.double(), "sum", offsets=offsets)
        pt2 = block[:, 1].square() + block[:, 2].square()
        descending = torch.ones_like(pt2)
        descending[:-1] = pt2[:-1] > pt2[1:]
        descending[offsets[1:] - 1] = 1
        ordered = torch.segment_reduce(descending, "min", offsets=offsets).bool()
        # Zenodo stores jets in transverse momentum order
        if ordered.any():
            rows = ordered.nonzero().flatten()
            ranks = torch.arange(width)
            valid = ranks < lengths[rows, None]
            source = offsets[rows, None] + torch.where(valid, ranks, 0)
            selected[rows] = block[source].masked_fill_(~valid[..., None], 0)
        # Use the scalar reader's sort for ties and unsorted input
        for length in lengths[~ordered].unique():
            rows = (~ordered & (lengths == length)).nonzero().flatten()
            p = block[offsets[rows, None] + torch.arange(length)]
            order = p[:, :, 1:3].square().sum(-1).argsort(dim=1, descending=True)[:, :width]
            selected[rows, :order.shape[1]] = p.gather(1, order[..., None].expand(-1, -1, 4))
        observables = {
            "jet_pt": totals[:, 1:3].norm(dim=1),
            "jet_energy": totals[:, 0],
            "jet_mass": (totals[:, 0].square() - totals[:, 1:].square().sum(1)).clamp_min(0).sqrt(),
            "n_constituents": lengths,
        }
        return selected, counts, observables

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        p = self.p4[self.offsets[i] : self.offsets[i + 1]].clone()
        total = p.double().sum(0)
        observables = {
            "jet_pt": float(total[1:3].norm()),
            "jet_energy": float(total[0]),
            "jet_mass": float((total[0].square() - total[1:].square().sum()).clamp_min(0).sqrt()),
            "n_constituents": float(len(p)),
        }
        indices = p[:, 1:3].square().sum(-1).argsort(descending=True)[: self.max_particles]
        p = p[indices]
        return Sample(
            constituent_features(p, self.feature_set),
            self.labels[i].clone(),
            p,
            observables=observables,
            weight=float(self.weights[i]),
            id=self.ids[i],
        )


def _lund_block(kernel, p4, counts, device, projection_batch):
    """
    Cluster jets of similar multiplicity and restore their source order

    Args:
        kernel:            Lund kernel for the selected columns
        p4:                Host tensor with transverse-momentum ordered constituents
        counts:            Retained constituent counts on the host
        device:            Target Torch device
        projection_batch:  Jet count per dense feature projection
    Returns:
        rows:  Feature matrix in source order
    """
    order = counts.argsort(stable=True)
    lengths = counts[order]
    momentum = p4[order].to(device)
    device_counts = lengths.to(device)
    # Share merge launches across jets and bound the double-precision pair table
    max_batch, max_pairs = projection_batch, 128 * 200**2
    if device.type == "cuda":
        memory = torch.cuda.get_device_properties(device).total_memory
        max_batch = 4096 if memory >= 32 * 1024**3 else 1024
        max_pairs = min(16 * 1024**2, max(max_pairs, memory // 2048))
    batches = []
    first = 0
    while first < len(lengths):
        size = min(max_batch, len(lengths) - first)
        while size > projection_batch and int(lengths[first + size - 1])**2 * size > max_pairs:
            size = max(projection_batch, (size // (2 * projection_batch)) * projection_batch)
        stop = first + size
        width = max(1, int(lengths[stop - 1]))
        batches.append(kernel(momentum[first:stop], device_counts[first:stop], width, projection_batch))
        first = stop
    rows = torch.cat(batches).cpu()
    return rows[order.argsort()]


def _native_lund_batches(resources, features, device, compile_graphs, threads, batch_size):
    """
    Combine compiled CPU Lund features with the selected Torch prong observables

    Args:
        resources:       Lifetime of the split and its workers
        features:        Output names in model order
        device:          Device for prong contractions
        compile_graphs:  Compile the dense prong contractions
        threads:         Number of native CPU workers
        batch_size:      Jets per Torch prong projection
    Returns:
        rows:  Feature kernel on a padded host batch in source order
    """
    prefix = tuple(name for name in features if name in BASES["prong_graphs"])
    lund = tuple(name for name in features if name in NATIVE_NAMES)
    native = resources.enter_context(native_top_rows(lund, threads)) if lund else None
    # The Lund prefix uses the 80 leading constituents and shares the Torch reference kernel
    prongs = jet_graph_kernel("lund_graphs", str(device), compile_graphs, prefix) if prefix else None
    executor = (resources.enter_context(ThreadPoolExecutor(max_workers=1))
                if native and prongs is not None and device.type == "cuda" else None)
    layout = prefix + lund
    indices = torch.tensor([layout.index(name) for name in features]) if tuple(features) != layout else None

    def rows(p4, counts):
        """Extract one block and place its columns in checkpoint order"""
        future = executor.submit(native, p4) if executor else None
        parts = []
        if prongs is not None:
            leading = p4[:, :80].contiguous().to(device)
            parts.append(prongs(leading, counts.to(device), leading.shape[1], batch_size).cpu())
        if native:
            parts.append(future.result() if future else native(p4))
        joined = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        return joined if indices is None else joined.index_select(1, indices)

    return rows


class JetGraphs(Dataset):
    @torch.no_grad()
    def __init__(self, jets, basis, device="cpu", compile_graphs=False, graph_features=None,
                 lund_backend="native", lund_threads=0):
        """
        Store one graph row per jet for repeated histogram passes

        Args:
            jets:  Top-tag sample with selected constituent four-momenta
            basis:  Jet graph basis in BASES
            device:  Device for graph contractions
            compile_graphs:  Compile the tensor kernel with TorchInductor
            graph_features:  Ordered feature names, or the complete basis if empty
            lund_backend:    Native CPU or Torch exclusive clustering
            lund_threads:    Number of native CPU workers, or zero for the job allocation
        """
        features = graph_features or BASES[basis]
        count = len(jets)
        mode = "compiled" if compile_graphs else "eager"
        native = basis in ("rest_hypergraphs", "lorentz_hypergraphs") or (basis == "lund_graphs" and lund_backend == "native")
        backend = f"{mode} Torch on {device}"
        if native:
            lund_threads = lund_threads or cpu_workers()
            stages = []
            if any(name in NATIVE_NAMES for name in features):
                family = "decay hypergraphs" if basis in ("rest_hypergraphs", "lorentz_hypergraphs") else "Lund"
                stages.append(f"compiled C++ {family} on {lund_threads} CPU threads")
            if any(name in BASES["prong_graphs"] for name in features):
                stages.append(f"{backend} for prongs")
            backend = " + ".join(stages)
        print(f"Computing {len(features)} {basis} features for {count:,} jets with {backend}")
        started = monotonic()
        next_report = started + 15
        self.ids, self.weights, self.labels = jets.ids, jets.weights, jets.labels
        self.rows = torch.empty((len(jets), 1, len(features)), dtype=torch.float32)
        self.observables = {}
        if not count:
            return
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        # Pad the last batch so validation and prediction reuse the training kernel
        batch_size = 128
        block_size = 128 * batch_size if basis == "lund_graphs" and device.type == "cuda" else 16 * batch_size
        with ExitStack() as resources:
            if native:
                kernel = _native_lund_batches(resources, features, device, compile_graphs, lund_threads, batch_size)
            else:
                kernel = jet_graph_kernel(basis, str(device), compile_graphs, tuple(features))
            last_report, last_count = started, 0
            for start in range(0, count, block_size):
                done = min(start + block_size, count)
                p4, counts, observables = jets.graph_inputs(start, done, batch_size if compile_graphs else 1)
                if start == 0:
                    self.observables = {name: values.new_empty(count) for name, values in observables.items()}
                for name, values in observables.items():
                    self.observables[name][start:done] = values
                size = done - start
                if native:
                    rows = kernel(p4, counts)
                elif basis == "lund_graphs":
                    rows = _lund_block(kernel, p4, counts, device, batch_size)
                else:
                    p4, counts = p4.to(device), counts.to(device)
                    rows = torch.cat([kernel(p, n) for p, n in zip(p4.split(batch_size), counts.split(batch_size), strict=True)])
                self.rows[start:done, 0].copy_(rows[:size])
                now = monotonic()
                if now >= next_report and done < count:
                    rate = (done - last_count) / (now - last_report)
                    last_count, last_report = done, now
                    print(f"{basis} {done:,}/{count:,} jets, {rate:.1f} jets/s, estimated {(count - done) / rate / 60:.1f} min remaining")
                    next_report = now + 15
        print(f"Computed {basis} for {count:,} jets in {monotonic() - started:.1f} s")

    def __len__(self):
        """Return the number of selected jets"""
        return len(self.labels)

    def classification_arrays(self):
        """
        Expose one descriptor row per jet in the same order as the sample reader

        Returns:
            arrays:  CPU feature rows, class labels and physical event weights
        """
        return self.rows[:, 0], self.labels, torch.as_tensor(self.weights)

    def classification_batch(self, indices, observable="", *, observables=()):
        """Gather physical jet rows and selected observables in sampler order"""
        features, labels, weights = self.classification_arrays()
        columns = {name: self.observables[name][indices] for name in observables}
        balance_values = columns.get(observable)
        if observable and balance_values is None:
            balance_values = self.observables[observable][indices]
        return ClassificationRows(features[indices], labels[indices], weights[indices],
                                  balance_values=balance_values, observables=columns)

    def __getitem__(self, index):
        """
        Return the graph row and full jet observables

        Args:
            index:  Event index
        Returns:
            sample:  Jet descriptors, label, weight and source identity
        """
        return Sample(self.rows[index], self.labels[index], weight=float(self.weights[index]), id=self.ids[index],
                      observables={name: float(values[index]) for name, values in self.observables.items()})
