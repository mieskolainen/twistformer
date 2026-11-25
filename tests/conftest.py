# Physics and workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

import torch
import pytest


@pytest.fixture(autouse=True)
def small_cpu_tests():
    """
    Isolate random generators, CPU threads, CUDA precision and compiled model caches
    """
    torch.compiler.reset()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    backends = (torch.backends.cuda.matmul, torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
    precision = [backend.fp32_precision for backend in backends]
    yield
    for backend, original in zip(backends, precision, strict=True):
        if backend.fp32_precision != original:
            backend.fp32_precision = original
