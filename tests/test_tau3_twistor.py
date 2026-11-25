# Tau energy-flow proxies and detector twistor validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
import pytest
import torch

from drivers.tau3_classification.geometry import hit_momenta
from twistformer import (
    load_config,
    parse_config,
    build_experiment,
    Trainer,
    load_predictor,
)
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor
from twistformer.geometry.lorentz import dot, validate_p4
from twistformer.geometry.twistors import reference_metric, twistors_from_p4


def test_null_proxies_preserve_energy_and_raw_geometry():
    """
    Preserve hit energy and nominal-vertex directions through normalization

    Returns:
        None: Proxies are null and preprocessing leaves them unchanged
    """
    origin = [10.0, 20.0, 30.0]
    hits = torch.tensor([[13.0, 24.0, 30.0, 5.0], [10.0, 20.0, 28.0, 2.0]])
    p4 = hit_momenta(hits, origin)
    expected = torch.tensor([[5.0, 3.0, 4.0, 0.0], [2.0, 0.0, 0.0, -2.0]]).double()
    torch.testing.assert_close(p4, expected)
    torch.testing.assert_close(dot(p4, p4), torch.zeros(2).double())
    sample = Sample(hits, p4=p4)
    pre = Preprocessor().fit([sample])
    transformed = pre.transform(sample)
    assert torch.equal(transformed.p4, expected)
    assert not torch.equal(transformed.features, hits)
    batch = collate_samples([transformed])
    validate_p4(batch.p4, batch.mask)


@pytest.mark.parametrize(
    "hit", [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, -1], [1, 0, float("nan"), 1]]
)
def test_invalid_proxy_hits_rejected(hit):
    """
    Reject undefined directions and nonphysical hit energies

    Args:
        hit:  Invalid x,y,z,E token
    Returns:
        None: Proxy construction raises ValueError
    """
    with pytest.raises(ValueError):
        hit_momenta(torch.tensor([hit]).float())


def small_config(regression=False):
    """
    Build a compact tau hybrid for CPU validation

    Args:
        regression:  Select six-vector Gaussian mixture regression
    Returns:
        config: Validated tau prototype with short training settings
    """
    task = "regression" if regression else "classification"
    data = load_config(f"configs/tau3_{task}/twistformer.yml").to_dict()
    data["driver"]["files"] = ["a", "b", "c"]
    data["model"].update(
        width=8,
        heads=2,
        depth=2,
        pair_channels=4,
        representation_channels=2,
        head_widths=[8],
    )
    data["training"].update(
        device="cpu",
        compile=False,
        workers=0,
        epochs=1,
        warmup_epochs=0,
        batch_size=12,
        eval_batch_size=12,
        amp=False,
    )
    return parse_config(data)


@pytest.mark.parametrize("regression", [False, True])
def test_hybrid_gradients_permutations_and_detector_orientation(regression):
    """
    Train both branches while retaining detector orientation and set symmetry

    Args:
        regression:  Exercise classification or joint mixture outputs
    Returns:
        None: Finite gradients, permutation symmetry and orientation sensitivity
    """
    exp = build_experiment(small_config(regression))
    hits = torch.tensor(
        [
            [1200.0, 50.0, 80.0, 5.0],
            [1100.0, -60.0, 100.0, 2.0],
            [1300.0, 100.0, -50.0, 1.0],
        ]
    )
    target = torch.arange(6).float() if regression else torch.tensor(1)
    sample = Sample(hits, target, p4=hit_momenta(hits))
    samples = [sample, replace(sample, features=hits[:1], p4=sample.p4[:1])]
    pre = Preprocessor().fit(samples, regression=regression)
    batch = collate_samples([pre.transform(s) for s in samples])
    exp.model.eval()
    prediction = exp.model(batch)
    loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    for branch in (exp.model.backbone.detector, exp.model.backbone.twistor):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    result = prediction.component_mean if regression else prediction.logits
    order = torch.tensor([2, 0, 1])
    permuted = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    other = exp.model(permuted)
    torch.testing.assert_close(
        result,
        other.component_mean if regression else other.logits,
        atol=2e-5,
        rtol=2e-5,
    )
    rotated_hits = hits.clone()
    rotated_hits[:, :2] = torch.stack((-hits[:, 1], hits[:, 0]), dim=-1)
    rotated = replace(sample, features=rotated_hits, p4=hit_momenta(rotated_hits))
    shifted = exp.model(collate_samples([pre.transform(rotated)]))
    shifted_result = shifted.component_mean if regression else shifted.logits
    assert not torch.allclose(result[:1], shifted_result, atol=1e-5, rtol=1e-5)
    dirty = replace(
        batch,
        features=batch.features.masked_fill(~batch.mask[..., None], 1234),
        p4=batch.p4.masked_fill(~batch.mask[..., None], 1234),
    )
    padded = exp.model(dirty)
    torch.testing.assert_close(
        result, padded.component_mean if regression else padded.logits
    )


@pytest.mark.parametrize("opening", [0.0, 1e-8, 1e-4])
def test_collinear_energy_flow_is_finite(opening):
    """
    Mask unresolved null references and retain finite hybrid derivatives

    Args:
        opening:  Transverse angular perturbation of one hit
    Returns:
        None: Stable geometry and parameter gradients for narrow showers
    """
    exp = build_experiment(small_config())
    hits = torch.tensor([[1200.0, 800.0, 700.0, 50.0]], dtype=torch.float64).repeat(
        64, 1
    )
    hits[0, 1] += 1200 * opening
    p4 = hit_momenta(hits)
    batch = collate_samples([Sample(hits.float(), torch.tensor(0), p4=p4)])
    tw, metric = twistors_from_p4(batch.p4, batch.mask)
    assert torch.isfinite(tw.Z).all() and torch.isfinite(metric).all()
    if opening == 0:
        assert not reference_metric(batch.p4.sum(1))[2].any()
        assert not tw.Z.count_nonzero()
    prediction = exp.model(batch)
    exp.task.compute_loss(prediction, batch).loss.backward()
    assert torch.isfinite(prediction.logits).all()
    assert all(
        torch.isfinite(p.grad).all()
        for p in exp.model.parameters()
        if p.grad is not None
    )


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("reference", [None, "time", "beam_plane"])
def test_tau_proxy_training_and_checkpoint(tmp_path, monkeypatch, regression, reference):
    """
    Exercise proxy datasets, training and unlabeled checkpoint inference

    Args:
        tmp_path:  Isolated run directory
        monkeypatch:  Replace ROOT I/O with aligned detector chunks
        regression:  Select classification or six-vector regression
        reference:  Optional laboratory reference override
    Returns:
        None: Both drivers preserve proxies and reload hybrid predictions
    """
    pytest.importorskip("awkward")
    pytest.importorskip("uproot")
    from drivers.tau3_classification.driver import Driver
    from drivers.tau3_classification.reader import Tau3Chunk
    from drivers.tau3_regression import reader

    def chunks(read_targets=True):
        """
        Construct varying calorimeter hits and optional independent truth labels

        Args:
            read_targets:  Include six-vector targets
        Returns:
            chunk: Compact synthetic detector events
        """
        hits = np.tile(
            [[1200.0, 50.0, 80.0, 5.0], [1100.0, -60.0, 100.0, 2.0]], (12, 1)
        ).astype(np.float32)
        hits[:, 1] += np.repeat(np.arange(12), 2)
        return Tau3Chunk(
            features=hits,
            offsets=np.arange(13) * 2,
            weights=np.ones(12),
            entries=np.arange(12),
            targets=np.arange(72).reshape(12, 6).astype(np.float32)
            if read_targets
            else None,
        )

    def read_classification(self, path, apply_fiducial=True):
        """
        Supply calorimeter chunks to the classification adapter

        Args:
            self:  Classification driver
            path:  Mock input identifier
            apply_fiducial:  Apply acceptance for labeled training samples
        Returns:
            chunks: Iterator over detector events
        """
        yield chunks(False)

    def read_regression(*args, read_targets=True, **kwargs):
        """
        Supply calorimeter chunks with optional regression truth

        Args:
            args:  ROOT source arguments
            read_targets:  Include regression targets
            kwargs:  Reader selection arguments
        Returns:
            chunks: Iterator over detector events
        """
        yield chunks(read_targets)

    monkeypatch.setattr(Driver, "_read", read_classification)
    monkeypatch.setattr(reader, "iter_tau3_regression", read_regression)
    cfg = small_config(regression)
    if reference is not None:
        cfg.model.reference_token = reference
    exp = build_experiment(cfg)
    for split in ("train", "val", "test"):
        sample = exp.driver.dataset(split)[0]
        torch.testing.assert_close(sample.p4, hit_momenta(sample.features))
    checkpoint = Trainer(tmp_path / "run").fit(exp)
    predictor = load_predictor(checkpoint, "cpu")
    app = predictor.experiment.driver
    dataset = app.prediction_dataset("unlabeled")
    sample = dataset[0]
    assert sample.target is None
    torch.testing.assert_close(sample.p4, hit_momenta(sample.features))
    batch = collate_samples([predictor.preprocessor.transform(sample)])
    restored = predictor.experiment.model
    restored.eval()
    prediction = restored(batch)
    value = prediction.component_mean if regression else prediction.logits
    assert torch.isfinite(value).all()
    clone = build_experiment(predictor.experiment.config).model.eval()
    clone.load_state_dict(restored.state_dict())
    other = clone(batch)
    torch.testing.assert_close(
        value, other.component_mean if regression else other.logits
    )


@pytest.mark.parametrize("regression", [False, True])
def test_hit_only_root_inference(tmp_path, regression):
    """
    Construct proxy inputs from a ROOT file containing no MC branches

    Args:
        tmp_path:  Isolated ROOT fixture directory
        regression:  Select classification or regression input adapter
    Returns:
        None: Unlabeled detector events retain energy and hit directions
    """
    ak = pytest.importorskip("awkward")
    uproot = pytest.importorskip("uproot")
    from drivers.tau3_classification.reader import ECAL_BRANCHES_BY_MODE

    columns = ECAL_BRANCHES_BY_MODE["sim"]
    path = tmp_path / "hits.root"
    with uproot.recreate(path) as root:
        root.mktree(
            "events",
            {
                columns["x"]: ak.Array([[1200.0, 1100.0], [1300.0]]),
                columns["y"]: ak.Array([[50.0, -60.0], [100.0]]),
                columns["z"]: ak.Array([[80.0, 100.0], [-50.0]]),
                columns["e"]: ak.Array([[5.0, 2.0], [1.0]]),
            },
        )
    app = build_experiment(small_config(regression)).driver
    samples = app.prediction_dataset(str(path))
    assert len(samples) == 2
    for sample in samples:
        assert sample.target is None
        torch.testing.assert_close(sample.p4, hit_momenta(sample.features))


def test_tau_proxy_steering_validation():
    """
    Reject inactive or malformed proxy and hybrid steering

    Returns:
        None: Invalid configurations fail before reading detector data
    """
    data = small_config().to_dict()
    data["model"]["family"] = "transformer"
    with pytest.raises(ValueError, match="twistformer"):
        parse_config(data)
    for origin in ([1.0, 2.0], [0.0, 0.0, float("nan")]):
        cfg = small_config()
        cfg.driver["proxy_origin"] = origin
        with pytest.raises(ValueError):
            build_experiment(cfg)
    cfg = small_config()
    cfg.driver["momentum_proxy"] = False
    with pytest.raises(ValueError, match="requires explicit p4"):
        build_experiment(cfg)
