# Parallel ROOT entry ranges and ordered ECAL shower batches
#
# mikael.mieskolainen@cern.ch (2026)

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
import pickle

import numpy as np
import torch

from .graphs.descriptors import BASES
from .reader import descriptor_batches, graph_selection, iter_clusters, open_root, selection_cutflows


def selected_showers(chunk):
    """
    Split local hits from a ROOT chunk in event order

    Args:
        chunk:  Retained cell rows and event offsets
    Returns:
        result:  One hit array or tensor per shower
    """
    if torch.is_tensor(chunk.features):
        return chunk.features.split(np.diff(chunk.offsets).tolist())
    return np.split(chunk.features, chunk.offsets[1:-1])


def read_range(path, start, stop, options, geometry, labeled, cpu_components, defer_descriptors):
    """
    Read one entry range and reduce its CPU graph components

    Args:
        path:  ROOT file
        start:  First event
        stop:  Exclusive entry bound
        options:  Selection and CPU input settings
        geometry:  ECAL readout constants
        labeled:  Read MC labels and observables
        cpu_components:  CPU components used with CUDA contractions
        defer_descriptors:  Keep local hits for graph contractions in the parent process
    Returns:
        result:  Selected shower chunks and selection counts
    """
    chunks = []
    cutflows = selection_cutflows(options, labeled)
    for chunk in iter_clusters(path, options, geometry, labeled, entry_start=start, entry_stop=stop,
                               cutflows=cutflows, defer_descriptors=defer_descriptors):
        cpu_rows = None
        if cpu_components:
            keep = graph_selection(chunk.passes_decay_cut, chunk.passes_photon_theta)
            cpu_rows = descriptor_batches(selected_showers(chunk), cpu_components, "cpu",
                                          getattr(geometry, "inner_radius_mm", None), keep=keep)
        chunks.append((chunk, cpu_rows))
    # Keep large CPU descriptor buffers out of the multiprocessing shared memory pool
    payload = pickle.dumps(chunks, protocol=pickle.HIGHEST_PROTOCOL) if options.shower_basis or cpu_components else chunks
    return payload, cutflows


def ordered_results(pool, jobs, arguments, prefetch):
    """
    Submit a bounded number of ROOT reads and consume results in source order

    Args:
        pool:  CPU process pool
        jobs:  File, entry bounds and end-of-file flag for each read
        arguments:  Common arguments for read_range
        prefetch:  Maximum number of pending reads
    Yields:
        result:  File, selected chunks and end-of-file flag
    """
    futures = ((path, last, pool.submit(read_range, path, start, stop, *arguments))
               for path, start, stop, last in jobs)
    pending = deque(islice(futures, prefetch))
    while pending:
        path, last, future = pending.popleft()
        result = future.result()
        next_job = next(futures, None)
        if next_job is not None:
            pending.append(next_job)
        yield path, result, last


def file_entry_limits(paths, options):
    """
    Read the configured entry limit for each ROOT file

    Args:
        paths:  Ordered ROOT files
        options:  ROOT tree and maximum entries per file
    Yields:
        result:  ROOT file and entry limit
    """
    from uproot.models.TTree import Model_TTree_NumEntries

    # Read only fEntries from TTrees when scheduling worker ranges
    for path in paths:
        with open_root(path, custom_classes={"TTree": Model_TTree_NumEntries}) as root:
            tree = root[options.tree]
            entries = tree.member("fEntries")[0] if isinstance(tree, Model_TTree_NumEntries) else tree.num_entries
            size = min(entries, options.max_events)
        yield path, size


def entry_ranges(files, step):
    """
    Queue entry ranges across ROOT file boundaries in source order

    Args:
        files:  Ordered ROOT files and entry limits
        step:  Maximum entries per range
    Yields:
        result:  ROOT file, entry bounds and end-of-file flag
    """
    for path, size in files:
        for start in range(0, max(1, size), step):
            stop = min(start + step, size)
            yield path, start, stop, stop == size


def ordered_chunks(paths, options, geometry, labeled, device, workers, cutflows):
    """
    Process independent CPU reads and emit shower tensors in deterministic order

    Args:
        paths:  ROOT files in steering order
        options:  Detector selection and shower basis
        geometry:  ECAL readout constants
        labeled:  Read MC observables and labels
        device:  Device for batched graph descriptors
        workers:  Maximum number of CPU reader processes
        cutflows:  Parent event and cell counts for all source files
    Yields:
        result:  File, selected chunks and an end-of-file flag
    """
    files = file_entry_limits(paths, options)
    step = options.chunk_size
    if 1 < workers and len(paths) < workers:
        files = list(files)
        ranges = sum(max(1, (size + step - 1) // step) for _, size in files)
        if ranges < workers:
            entries = sum(size for _, size in files)
            step = max(1, entries // workers)
            workers = min(workers, sum(max(1, size) for _, size in files))

    if workers <= 1:
        for path in paths:
            yield path, iter_clusters(path, options, geometry, labeled, device, cutflows=cutflows), True
        return

    components = options.graph_components
    gpu_components = ()
    if torch.device(device).type == "cuda":
        # ROOT workers contract the projected profiles and transport moments
        cpu_bases = ("projected_profiles", "transport_graphs", "shower_frame_projected_profiles",
                     "shower_frame_transport_graphs", "shower_radial_projected_profiles")
        gpu_components = tuple(name for name in components if name not in cpu_bases)
    cpu_components = tuple(name for name in components if name not in gpu_components) if gpu_components else ()
    cpu_widths = [len(BASES[name]) for name in cpu_components]
    gpu_widths = [len(BASES[name]) for name in gpu_components]
    arguments = options, geometry, labeled, cpu_components, bool(gpu_components)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=torch.set_num_threads, initargs=(1,)) as pool:
        # Keep reads queued while the parent consumes the previous ranges
        for path, (result, counts), last in ordered_results(pool, entry_ranges(files, step), arguments, 2 * workers):
            for total, partial in zip(cutflows, counts, strict=True):
                total.merge(partial)
            if isinstance(result, bytes):
                result = pickle.loads(result)
            chunks = []
            for chunk, cpu_rows in result:
                if gpu_components:
                    radius = getattr(geometry, "inner_radius_mm", None)
                    keep = graph_selection(chunk.passes_decay_cut, chunk.passes_photon_theta)
                    gpu_rows = descriptor_batches(selected_showers(chunk), gpu_components, device, radius, keep=keep)
                    if cpu_rows is None:
                        chunk.features = gpu_rows
                    else:
                        columns = dict(zip(cpu_components, cpu_rows.split(cpu_widths, dim=1), strict=True))
                        columns.update(zip(gpu_components, gpu_rows.split(gpu_widths, dim=1), strict=True))
                        chunk.features = torch.cat([columns[name] for name in components], dim=1)
                    chunk.offsets = np.arange(len(chunk) + 1)
                chunks.append(chunk)
            yield path, chunks, last
