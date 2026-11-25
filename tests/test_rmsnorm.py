# RMS normalization independent of native Torch RMSNorm APIs
#
# mikael.mieskolainen@cern.ch (2026)

import subprocess
import sys
import pytest
import torch
from twistformer.layers.common import RMSNorm


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("affine", [False, True])
def test_rmsnorm_reference_gradients_and_state(shape, affine):
    layer = RMSNorm(shape, eps=1e-6, elementwise_affine=affine, dtype=torch.float64)
    reference = torch.nn.RMSNorm(
        shape, eps=1e-6, elementwise_affine=affine, dtype=torch.float64
    )
    if affine:
        with torch.no_grad():
            layer.weight.uniform_(0.5, 1.5)
    reference.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn(4, *shape, dtype=torch.float64, requires_grad=True)
    actual, expected = layer(x), reference(x)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, *layer.parameters()))
    expected_grad = torch.autograd.grad(
        expected.square().sum(), (x, *reference.parameters())
    )
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e)
    assert set(layer.state_dict()) == ({"weight"} if affine else set())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_low_precision_accumulation(dtype):
    layer = RMSNorm(4, eps=1e-6)
    x = torch.tensor(
        [[1000.0, -2000.0, 3000.0, -4000.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=dtype):
        actual = layer(x)
    expected = x.double() / (x.double().square().mean(-1, keepdim=True) + 1e-6).sqrt()
    torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)
    assert actual.dtype == dtype and actual.isfinite().all()
    actual.sum().backward()
    assert x.grad.isfinite().all() and layer.weight.grad.isfinite().all()


def test_rmsnorm_import_without_native_apis():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
del torch.nn.RMSNorm
del torch.nn.functional.rms_norm
from twistformer import build_experiment, parse_config
from twistformer.data.batching import collate_samples
config = parse_config({
    'driver': {'name': 'synthetic', 'train_size': 4, 'max_tokens': 3},
    'model': {'norm': 'rms', 'head_norm': 'rms', 'width': 8, 'heads': 2, 'depth': 1},
})
experiment = build_experiment(config)
batch = collate_samples(experiment.driver.dataset('train')[:2])
loss = experiment.task.compute_loss(experiment.model(batch), batch).loss
loss.backward()
assert loss.isfinite()
assert all(p.grad is None or p.grad.isfinite().all() for p in experiment.model.parameters())
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
