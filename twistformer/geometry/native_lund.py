# Compiled CPU clustering and angular flow for batches of constituent four-momenta
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import contextmanager
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sysconfig
from tempfile import TemporaryDirectory
from threading import Lock

import torch


@lru_cache(maxsize=1)
def _library():
    """Build and load the C++17 extractor for this source version and platform"""
    source = Path(__file__).with_name("native")
    files = [source / name for name in ("jet_rows.cc", "clustering_history.cc", "angular_flow.cc")]
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    flags = ["-std=c++17", "-O3", "-pthread", "-fPIC", "-shared"]
    digest = hashlib.sha256(repr((compiler, flags, sysconfig.get_platform(), platform.libc_ver())).encode())
    for file in [*files, *sorted(source.glob("*.hh"))]:
        digest.update(file.read_bytes())
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "twistformer" / "lund"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"jet_rows_{digest.hexdigest()[:24]}.so"
    if not library.is_file():
        # Publish atomically when independent jobs build the same source on a shared filesystem
        with TemporaryDirectory(dir=cache) as build:
            output = Path(build) / library.name
            try:
                subprocess.run([*compiler, *flags, *files, "-o", output], check=True,
                               capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                detail = getattr(error, "stderr", None) or str(error)
                raise RuntimeError(f"Building the Lund extractor requires a C++17 compiler. "
                                   f"Set CXX or use driver.lund_backend=torch.\n{detail}") from error
            output.replace(library)
    lib = ctypes.CDLL(str(library))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.jet_feature_count.argtypes = [integer, ctypes.c_bool]
    lib.jet_feature_count.restype = integer
    lib.jet_workers_create.argtypes = [ctypes.c_uint]
    lib.jet_workers_create.restype = ptr
    lib.jet_workers_destroy.argtypes = [ptr]
    lib.jet_workers_destroy.restype = None
    lib.jet_workers_run.argtypes = [ptr, integer, ptr, ptr, integer, integer, ptr]
    lib.jet_workers_run.restype = integer
    return lib


def cpu_workers():
    """Count logical CPUs available to the job"""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count() or 1
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "")
    return min(count, int(allocation)) if allocation.isdecimal() and int(allocation) > 0 else count


@contextmanager
def native_lund_rows(blocks, mass_hypothesis, threads):
    """
    Keep one CPU worker group for successive batches of jet four-momenta

    Args:
        blocks:           Mask for clustering, angular flow and decay graph blocks
        mass_hypothesis:  Parent and pair masses and widths in GeV, or None
        threads:          Number of CPU workers
    Yields:
        rows:  Callable from a CPU tensor [jets, slots, 4] to float32 feature rows
    """
    lib = _library()
    columns = lib.jet_feature_count(blocks, mass_hypothesis is not None)
    if not columns or not 0 < threads <= 2**31 - 1:
        raise ValueError("Expected valid Lund blocks and a positive CPU worker count")
    mass_fit = (ctypes.c_double * 4)(*mass_hypothesis) if mass_hypothesis is not None else None
    handle = lib.jet_workers_create(threads)
    if not handle:
        raise RuntimeError("Could not allocate Lund CPU workers")

    lock = Lock()

    def rows(momenta):
        """Compute one batch in source order"""
        if (momenta.device.type != "cpu" or momenta.ndim != 3 or momenta.shape[2] != 4
                or not 0 < momenta.shape[1] < 2**30 or len(momenta) >= 2**31):
            raise ValueError("Expected CPU four-momenta with shape [jets, slots, 4]")
        momenta = momenta.to(torch.float32).contiguous()
        output = torch.empty((len(momenta), columns), dtype=torch.float32, device="cpu")
        with lock:
            if not handle:
                raise RuntimeError("Lund worker group is closed")
            status = lib.jet_workers_run(handle, blocks, mass_fit, momenta.data_ptr(), len(momenta),
                                         momenta.shape[1], output.data_ptr())
        if status > 0:
            raise ValueError(f"Invalid four-momenta at jet {status} in the current Lund batch")
        if status < 0:
            raise RuntimeError("Compiled Lund extraction failed")
        return output

    try:
        yield rows
    finally:
        with lock:
            lib.jet_workers_destroy(handle)
            handle = None
