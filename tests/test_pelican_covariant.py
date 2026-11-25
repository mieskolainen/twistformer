# Covariant PELICAN four-vector outputs and workflows
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
import pytest
import torch
from twistformer import Trainer, build_experiment, load_predictor, parse_config
from twistformer.api.contracts import InputSpec, Prediction, Sample, TaskSpec
from twistformer.api.config import PreprocessingConfig
from twistformer.data.batching import collate_samples
from twistformer.evaluation.runner import evaluate
from twistformer.evaluation.metrics import regression_metrics
from twistformer.layers.pelican import ParticleReadout
from twistformer.models.factory import Model
from twistformer.training.checkpoint import read_checkpoint


def configuration(vectors=1, **model):
    """
    Build a small covariant regression experiment

    Args:
        vectors:  Number of target four-vectors
        model:  Model overrides
    Returns:
        config: Validated experiment configuration
    """
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 12,
                "val_size": 6,
                "test_size": 4,
                "max_tokens": 5,
                "target_vectors": vectors,
            },
            "model": {
                "family": "pelican",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "norm": "rms",
                "dropout": 0.0,
                **model,
            },
            "task": {"name": "covariant_regression"},
            "preprocessing": {"normalize_targets": False},
            "training": {
                "device": "cpu",
                "epochs": 2,
                "batch_size": 4,
                "eval_batch_size": 4,
                "save_every": 1,
            },
        }
    )


def lorentz_matrix():
    """
    Construct a boost along a generic direction followed by a spatial rotation

    Returns:
        matrix: Proper orthochronous Lorentz transformation (4,4)
    """
    beta = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
    beta2 = beta.square().sum()
    gamma = (1 - beta2).rsqrt()
    boost = torch.eye(4, dtype=torch.float64)
    boost[0, 0] = gamma
    boost[0, 1:] = boost[1:, 0] = -gamma * beta
    boost[1:, 1:] += (gamma - 1) / beta2 * beta[:, None] * beta[None]
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[1:3, 1:3] = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    return rotation @ boost


@pytest.mark.parametrize("multiplicity", [False, True])
def test_particle_maps_match_kronecker_contractions(multiplicity):
    """
    Check all five 2->1 maps and gradients against independent index contractions

    Args:
        multiplicity:  Enable learned constituent-count powers
    Returns:
        None: Pair-to-particle maps and gradients match their contraction basis
    """
    torch.manual_seed(21)
    layer = ParticleReadout(2, 3, 2.0, multiplicity).double()
    x = torch.randn(2, 4, 4, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    i, j, k = torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij")
    basis = torch.stack(
        [
            (i == j) & (i == k),
            i == j,
            i == k,
            j == k,
            torch.ones_like(i, dtype=torch.bool),
        ]
    ).double()
    assert torch.linalg.matrix_rank(basis.flatten(1)) == 5
    valid = mask[:, :, None] & mask[:, None, :]
    operations = torch.einsum(
        "aijk,bjkc->baic", basis, x.masked_fill(~valid[..., None], 0)
    )
    count = mask.sum(1).double()
    order = torch.tensor([0, 1, 1, 1, 2])
    operations = operations / count[:, None, None, None].pow(order[None, :, None, None])
    if multiplicity:
        operations = operations * (count[:, None, None, None] / 2).pow(
            layer.alpha[None, :, None]
        )
    expected = torch.einsum("baic,adc->bid", operations, layer.weight) + layer.bias
    expected = expected.masked_fill(~mask[..., None], 0)
    actual = layer(x, mask)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *layer.parameters())
    found = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    wanted = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(found, wanted):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("vectors", [1, 2])
@pytest.mark.parametrize("amp", [False, True])
def test_covariant_output_boost_rotation_permutation_padding_and_gradients(
    vectors, amp
):
    """
    Transform predictions as four-vectors and exclude arbitrary padding

    Args:
        vectors:  Number of target four-vectors
        amp:  Enable BF16 learned channel projections
    Returns:
        None: Outputs are covariant and permutation invariant with finite gradients
    """
    exp = build_experiment(configuration(vectors))
    exp.model.eval()
    batch = collate_samples(exp.driver.dataset("train")[:3])
    batch = replace(batch, p4=batch.p4.double())
    transform = lorentz_matrix()
    metric = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
    torch.testing.assert_close(
        transform.T @ metric @ transform, metric, atol=1e-7, rtol=1e-7
    )
    order = torch.arange(batch.mask.shape[1] - 1, -1, -1)
    permuted = replace(
        batch,
        p4=batch.p4[:, order],
        features=batch.features[:, order],
        mask=batch.mask[:, order],
    )
    padded = replace(
        batch,
        p4=torch.nn.functional.pad(batch.p4, (0, 0, 0, 2), value=float("nan")),
        features=torch.nn.functional.pad(batch.features, (0, 0, 0, 2)),
        mask=torch.nn.functional.pad(batch.mask, (0, 2)),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        prediction = exp.model(batch)
        assert prediction.std is None and prediction.mean.shape == (3, 4 * vectors)
        boosted = exp.model(replace(batch, p4=batch.p4 @ transform.T)).mean
        expected = (prediction.mean.reshape(3, vectors, 4) @ transform.T).flatten(1)
        tolerance = 0.015 if amp else 2e-5
        torch.testing.assert_close(boosted, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            exp.model(permuted).mean, prediction.mean, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            exp.model(padded).mean, prediction.mean, atol=tolerance, rtol=tolerance
        )
        loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    assert loss.isfinite()
    for name, parameter in exp.model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


def test_covariant_coefficients_use_unscaled_momenta_and_signed_weights():
    """
    Verify four-vector ordering and unrestricted coefficients in physical units

    Returns:
        None: Constant coefficients give exact signed sums independent of input scaling
    """
    exp = build_experiment(configuration(2, pelican_momentum_scale=0.02))
    with torch.no_grad():
        exp.model.head.network[-1].weight.zero_()
        exp.model.head.network[-1].bias.copy_(torch.tensor([0.25, -0.5]))
    samples = [
        Sample(torch.tensor(p), p4=torch.tensor(p))
        for p in (
            [[2.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]],
        )
    ]
    batch = collate_samples(samples)
    predicted = exp.model(batch).mean.reshape(2, 2, 4)
    expected = batch.p4.sum(1)[:, None] * torch.tensor([0.25, -0.5])[None, :, None]
    torch.testing.assert_close(predicted, expected)
    exp.model.backbone.momentum_scale = 1.0
    torch.testing.assert_close(exp.model(batch).mean.reshape(2, 2, 4), expected)


@pytest.mark.parametrize("fusion", ["tokens", "readout"])
def test_covariant_global_and_auxiliary_scalars(fusion):
    """
    Preserve covariance with explicit invariant particle and event scalars

    Args:
        fusion:  Event-feature injection stage
    Returns:
        None: Scalar context changes coefficients without changing covariance
    """
    cfg = configuration(auxiliary_features=["charge"], global_fusion=fusion)
    spec = InputSpec(("charge",), "p4", global_features=("mass",))
    task = TaskSpec("covariant_regression", ("E", "px", "py", "pz"))
    model = Model(spec, task, cfg.model, cfg.task).eval()
    p4 = torch.tensor([[3.0, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    batch = collate_samples(
        [
            Sample(
                torch.tensor([[1.0], [-1.0]]),
                p4=p4,
                global_features=torch.tensor([4.0]),
            )
        ]
    )
    transform = lorentz_matrix()
    original = model(batch).mean
    transformed = model(replace(batch, p4=batch.p4 @ transform.T)).mean
    torch.testing.assert_close(
        transformed, original @ transform.T, atol=2e-5, rtol=2e-5
    )


def test_covariant_mse_missing_targets_and_metrics():
    """
    Reduce observed component errors without inventing Gaussian widths

    Returns:
        None: Weighted MSE and physical residual metrics match explicit values
    """
    exp = build_experiment(configuration())
    batch = collate_samples(exp.driver.dataset("train")[:2])
    target = torch.tensor([[2.0, float("nan"), 1.0, 5.0], [float("nan")] * 4])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    batch = replace(
        batch, targets=target, target_mask=mask, sample_weights=torch.tensor([2.0, 3.0])
    )
    mean = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]], requires_grad=True
    )
    result = exp.task.compute_loss(Prediction(mean=mean), batch)
    torch.testing.assert_close(result.loss, torch.tensor(2.5))
    result.loss.backward()
    assert mean.grad.isfinite().all() and not mean.grad[1].count_nonzero()
    metrics = regression_metrics(
        {
            "targets": target.numpy(),
            "target_mask": mask.numpy(),
            "mean": mean.detach().numpy(),
            "weights": np.array([2.0, 3.0]),
        },
        exp.driver.task_spec,
    )
    assert metrics["p0_E"]["rmse"] == 1.0 and metrics["p0_py"]["rmse"] == 2.0
    assert "nll" not in metrics["p0_E"] and "coverage_1sigma" not in metrics["p0_E"]


@pytest.mark.parametrize(
    "change",
    [
        {"model": {"family": "transformer"}},
        {"preprocessing": {"normalize_targets": True}},
        {"task": {"mixture_components": 2}},
        {"task": {"min_std": 0.1}},
    ],
)
def test_covariant_configuration_rejects_incompatible_options(change):
    """
    Reject scalar normalization, Gaussian widths and unsupported encoders

    Args:
        change:  Incompatible configuration override
    Returns:
        None: Invalid combinations fail before training
    """
    config = configuration().to_dict()
    for section, options in change.items():
        config[section].update(options)
    with pytest.raises(ValueError):
        parse_config(config)


def test_covariant_schema_and_preprocessing_guards():
    """
    Reject incomplete four-vectors and target transforms that break covariance

    Returns:
        None: Four-vector targets retain consistent units and physical coordinates
    """
    with pytest.raises(ValueError, match="complete"):
        TaskSpec("covariant_regression", ("px", "py", "pz"))
    with pytest.raises(ValueError, match="same unit"):
        TaskSpec(
            "covariant_regression", ("E", "px", "py", "pz"), ("GeV", "GeV", "GeV", "mm")
        )
    exp = build_experiment(configuration())
    samples = exp.driver.dataset("train")
    pre = exp.driver.create_preprocessor().fit(samples, regression=True)
    torch.testing.assert_close(pre.transform(samples[0]).target, samples[0].target)
    with pytest.raises(ValueError, match="physical"):
        exp.driver.create_preprocessor(
            config=PreprocessingConfig(normalize_targets=True)
        )
    invalid = {**pre.state, "target_mean": torch.ones(4)}
    with pytest.raises(ValueError, match="identity"):
        exp.driver.create_preprocessor(invalid)


def test_covariant_train_resume_predict_and_evaluate(tmp_path):
    """
    Train and restore covariant outputs through the public API and evaluation plots

    Args:
        tmp_path:  Temporary run and evaluation directories
    Returns:
        None: Exact resume, covariant physical predictions and residual plots work
    """
    cfg = configuration(pelican_triangle=True, pelican_triangle_channels=4)
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "full").fit(exp)
    Trainer(
        tmp_path / "resumed", resume=tmp_path / "full/checkpoints/epoch_00000.pt"
    ).fit(build_experiment(cfg))
    first = read_checkpoint(tmp_path / "full/checkpoints/last.pt")
    second = read_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    for name, tensor in first["model"].items():
        torch.testing.assert_close(tensor, second["model"][name], atol=0, rtol=0)
    predictor = load_predictor(best, "cpu")
    samples = exp.driver.dataset("test")
    predictions = predictor.predict(samples)
    assert predictions.mean.shape == (4, 4) and predictions.std is None
    transform = lorentz_matrix().float()
    boosted = [replace(sample, p4=sample.p4 @ transform.T) for sample in samples]
    torch.testing.assert_close(
        predictor.predict(boosted).mean,
        predictions.mean @ transform.T,
        atol=2e-5,
        rtol=2e-5,
    )
    metrics = evaluate(predictor, samples, tmp_path / "evaluation", batch_size=2)
    assert np.isfinite(metrics["preprocessed_loss"])
    with np.load(tmp_path / "evaluation/predictions.npz") as data:
        assert "mean" in data and "std" not in data
    assert (tmp_path / "evaluation/plots/residual_0.png").exists()
    assert not list((tmp_path / "evaluation/plots").glob("pull*"))
