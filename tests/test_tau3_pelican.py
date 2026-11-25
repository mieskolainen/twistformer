# Tau PELICAN detector inputs and physical regression targets
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import pytest
import torch

from drivers.tau3_classification.geometry import hit_momenta
from twistformer import build_experiment, load_config
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tau_pelican_detector_inputs_and_gradients(task):
    """
    Retain detector orientation, set symmetry and finite derivatives

    Args:
        task:  Tau classification or Gaussian mixture regression
    """
    cfg = load_config(f"configs/tau3_{task}/pelican.yml")
    exp = build_experiment(cfg)
    model = exp.model.eval()
    hits = torch.tensor([
        [1200.0, 50.0, 80.0, 5.0],
        [1100.0, -60.0, 100.0, 2.0],
        [1300.0, 100.0, -50.0, 1.0],
    ])
    regression = task == "regression"
    target = torch.arange(6).float() if regression else torch.tensor(1)
    sample = Sample(hits, target, p4=hit_momenta(hits))
    samples = [sample, replace(sample, features=hits[:1], p4=sample.p4[:1])]
    pre = Preprocessor().fit(samples, regression=regression)
    batch = collate_samples([pre.transform(item) for item in samples])
    torch.testing.assert_close(batch.p4[0], sample.p4)
    prediction = model(batch)
    value = prediction.component_mean if regression else prediction.logits
    assert value.shape == ((2, 5, 6) if regression else (2, 3))
    exp.task.compute_loss(prediction, batch).loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(gradient.isfinite().all() for gradient in grads)
    assert model.backbone.auxiliary.embed[0].weight.grad.abs().sum() > 0
    order = torch.tensor([2, 0, 1])
    shuffled = model(replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    ))
    torch.testing.assert_close(
        shuffled.component_mean if regression else shuffled.logits,
        value, atol=2e-5, rtol=2e-5,
    )
    dirty = model(replace(
        batch,
        features=batch.features.masked_fill(~batch.mask[..., None], float("nan")),
        p4=batch.p4.masked_fill(~batch.mask[..., None], float("nan")),
    ))
    torch.testing.assert_close(dirty.component_mean if regression else dirty.logits, value)
    rotated = hits.clone()
    rotated[:, :2] = torch.stack((-hits[:, 1], hits[:, 0]), -1)
    shifted = replace(sample, features=rotated, p4=hit_momenta(rotated))
    result = model(collate_samples([pre.transform(shifted)]))
    assert not torch.allclose(
        result.component_mean if regression else result.logits,
        value[:1], atol=1e-5, rtol=1e-5,
    )
