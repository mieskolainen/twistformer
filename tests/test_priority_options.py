# Auxiliary scalar inputs, attention ties and persistent training state
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
from dataclasses import replace
import csv
import math
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from twistformer import (
    InputSpec,
    Sample,
    Trainer,
    build_experiment,
    load_predictor,
    parse_config,
)
from twistformer.api.contracts import TaskSpec
from twistformer.data.batching import collate_samples
from twistformer.geometry.lorentz import from_matrix, matrix
from twistformer.geometry.twistors import twistors_from_p4, projective_normalize
from twistformer.layers.attention import TriangleAttention, tied_topk
from twistformer.models.factory import Model
from twistformer.models.twistformer import TwistorBlock
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.ema import ModelEMA
from twistformer.training.schedulers import make_scheduler, step_scheduler


@pytest.mark.parametrize("family", ["twistformer", "pelican"])
def test_auxiliary_scalar_inputs_covariance_masking_and_gradients(family):
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "auxiliary_features": ["charge"],
                **(
                    {"twistor_qk": True, "reference_token": "event"}
                    if family == "twistformer"
                    else {}
                ),
            },
        }
    )
    schema = InputSpec(("unused", "charge"), "p4")
    model = Model(
        schema, TaskSpec("classification", ("a", "b")), cfg.model, cfg.task
    ).eval()
    p = torch.tensor([[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]])
    batch = collate_samples(
        [
            Sample(torch.tensor([[7.0, 1.0], [8.0, -1.0], [9.0, 0.0]]), p4=p),
            Sample(torch.tensor([[7.0, -1.0], [8.0, 1.0]]), p4=p[:2]),
        ]
    )
    batch.features.requires_grad_()
    expected = model(batch).logits
    expected.square().sum().backward()
    assert batch.features.grad[..., 1].abs().sum() > 0
    assert not batch.features.grad[..., 0].count_nonzero()
    assert not batch.features.grad[~batch.mask].count_nonzero()
    assert not torch.allclose(
        model(replace(batch, features=batch.features + 2)).logits, expected
    )
    s = torch.tensor([[1.2, 0.2j], [0.0, 1 / 1.2]], dtype=torch.complex128)
    boosted = from_matrix(s @ matrix(batch.p4.double()) @ s.mH).float()
    torch.testing.assert_close(
        model(replace(batch, p4=boosted)).logits, expected, atol=2e-5, rtol=2e-5
    )
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 2), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 2)),
        mask=F.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(model(padded).logits, expected, atol=2e-6, rtol=2e-5)
    order = torch.tensor([2, 0, 1])
    shuffled = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    torch.testing.assert_close(model(shuffled).logits, expected, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="Unknown auxiliary_features"):
        Model(
            InputSpec(("x",), "p4"),
            TaskSpec("classification", ("a", "b")),
            cfg.model,
            cfg.task,
        )


def test_topk_boundary_includes_all_ties_without_padding():
    scores = torch.tensor(
        [[3.0, 2.0, 2.0, 1.0, -torch.inf], [5.0, 4.0, 3.0, 2.0, -torch.inf]],
        requires_grad=True,
    )
    selected, indices = tied_topk(scores, 2)
    assert selected.shape == (2, 3)
    assert set(indices[0].tolist()) == {0, 1, 2}
    assert torch.isfinite(selected[1]).sum() == 2
    selected.masked_fill(~torch.isfinite(selected), 0).sum().backward()
    torch.testing.assert_close(
        scores.grad,
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("distinct", [False, True])
def test_triangle_padding_does_not_expand_candidates(monkeypatch, shared, distinct):
    """
    Exclude padded queries from triangle candidate counts

    Args:
        monkeypatch:  Record the selected candidate widths
        shared:  Share the two candidate sets
        distinct:  Exclude equal particle indices
    Returns:
        None: Sparse candidates preserve predictions and gradients
    """
    import twistformer.layers.attention as attention

    torch.manual_seed(42)
    layer = TriangleAttention(
        8, 2, k=3, query_chunk=8, shared=shared, distinct=distinct
    ).eval()
    reference = deepcopy(layer)
    reference.fixed_candidates = True
    mask = torch.arange(32)[None] < torch.tensor([16, 15])[:, None]
    x = torch.randn(2, 32, 8).masked_fill(~mask[..., None], 0).requires_grad_()
    widths = []

    def record_width(*args, **kwargs):
        """
        Record the candidate axis after masked selection

        Args:
            args:  Candidate scores and selection options
            kwargs:  Query validity mask
        Returns:
            selected: Candidate scores and indices
        """
        selected = tied_topk(*args, **kwargs)
        widths.append(selected[0].shape[-1])
        return selected

    monkeypatch.setattr(attention, "tied_topk", record_width)
    actual = layer(x, mask)
    assert max(widths) == 3
    assert widths[-1] == 1
    expected = reference(x, mask)
    torch.testing.assert_close(actual, expected)
    assert not actual[~mask].count_nonzero()
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, *layer.parameters()))
    expected_grad = torch.autograd.grad(
        expected.square().sum(), (x, *reference.parameters())
    )
    for found, wanted in zip(actual_grad, expected_grad, strict=True):
        assert found.isfinite().all()
        torch.testing.assert_close(found, wanted, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("distinct", [False, True])
def test_triangle_ties_match_full_candidates_and_gradients(shared, distinct):
    torch.manual_seed(14)
    layer = TriangleAttention(8, 2, k=2, shared=shared, distinct=distinct).eval()
    with torch.no_grad():
        # All candidate scores tie while values and pair compatibilities differ
        layer.projections.weight[:8].zero_()
    dense = deepcopy(layer)
    dense.kj = dense.kk = 5
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    actual, expected = layer(x, mask), dense(x, mask)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (x, *layer.parameters()))
    ge = torch.autograd.grad(expected.square().sum(), (x, *dense.parameters()))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
    order = torch.tensor([3, 1, 4, 0, 2])
    shuffled = layer(x[:, order], mask[:, order])
    torch.testing.assert_close(shuffled, actual[:, order], atol=2e-6, rtol=2e-5)
    gp = torch.autograd.grad(shuffled.square().sum(), (x, *layer.parameters()))
    for a, e in zip(ga, gp, strict=True):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
    padded = layer(F.pad(x, (0, 0, 0, 3)), F.pad(mask, (0, 3)))
    torch.testing.assert_close(padded[:, :5], actual, atol=2e-6, rtol=2e-5)


def training_config(**options):
    """
    Construct a small reproducible training experiment

    Args:
        options:  Training option overrides
    Returns:
        config: Validated experiment configuration
    """
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 12,
                "val_size": 6,
                "test_size": 6,
                "max_tokens": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.1},
            "training": {
                "epochs": 5,
                "device": "cpu",
                "batch_size": 4,
                "eval_batch_size": 3,
                "save_every": 1,
                "keep_checkpoints": 0,
                **options,
            },
        }
    )


def test_ema_parameters_and_buffers():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(model, 0.75)
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(4)
        model[1].running_mean.fill_(3)
        model[1].num_batches_tracked.fill_(7)
    ema.update(model)
    for name, p in ema.module.named_parameters():
        torch.testing.assert_close(p, before[name] + 1)
        assert not p.requires_grad
    torch.testing.assert_close(ema.module[1].running_mean, model[1].running_mean)
    assert ema.module[1].num_batches_tracked == 7 and ema.updates == 1
    assert not ema.module.training


@pytest.mark.parametrize("kind", ["none", "cosine", "thirds", "plateau"])
def test_scheduler_rates_and_resume(kind):
    cfg = training_config(
        epochs=8,
        learning_rate=1.0,
        scheduler=kind,
        warmup_epochs=2,
        min_learning_rate=0.01,
        plateau_patience=0,
        plateau_threshold=0.0,
    ).training
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=1.0)
    scheduler = make_scheduler(optimizer, cfg)
    rates = []
    for epoch in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        step_scheduler(scheduler, 1.0)
        if epoch == 2:
            saved_optimizer = deepcopy(optimizer.state_dict())
            saved_scheduler = deepcopy(scheduler.state_dict())
    expected = {
        "none": [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cosine": [0.5, 1.0]
        + [0.01 + 0.99 * (1 + math.cos(math.pi * e / 6)) / 2 for e in range(6)],
        "thirds": [0.5, 1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01],
        "plateau": [0.5, 1.0, 1.0, 1.0, 0.5, 0.25, 0.125, 0.0625],
    }[kind]
    np.testing.assert_allclose(rates, expected)
    resumed_optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=1.0)
    resumed = make_scheduler(resumed_optimizer, cfg)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)
    for rate in rates[3:]:
        assert resumed_optimizer.param_groups[0]["lr"] == rate
        resumed_optimizer.step()
        step_scheduler(resumed, 1.0)


def test_plateau_uses_monitor_direction():
    cfg = training_config(
        scheduler="plateau", monitor="val_accuracy", mode="max", plateau_patience=0
    ).training
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=cfg.learning_rate)
    scheduler = make_scheduler(optimizer, cfg)
    for accuracy in (0.5, 0.6, 0.7):
        step_scheduler(scheduler, accuracy)
    assert optimizer.param_groups[0]["lr"] == cfg.learning_rate
    step_scheduler(scheduler, 0.6)
    assert optimizer.param_groups[0]["lr"] == cfg.learning_rate * cfg.plateau_factor


@pytest.mark.parametrize(("options", "rounds"), [
    ({}, 8),
    ({"scheduler": "none", "warmup_epochs": 2}, 8),
    ({"scheduler": "cosine", "warmup_epochs": 2}, 8),
    ({"scheduler": "cosine", "scheduler_interval": "step", "max_steps": 5}, 5),
    ({"scheduler": "cosine_restarts", "warmup_epochs": 2}, 8),
    ({"scheduler": "thirds", "warmup_epochs": 2}, 8),
    ({"scheduler": "plateau", "monitor": "val_auc", "mode": "max", "plateau_patience": 0, "patience": 3}, 4),
])
def test_xgboost_applies_torch_schedule(tmp_path, options, rounds):
    """
    Compare the native rate and logged rate with the Torch rate at each round

    Args:
        tmp_path:  Training record directory
        options:  Scheduler and stopping settings
        rounds:  Expected number of boosting rounds
    """
    import json
    from types import SimpleNamespace

    xgb = pytest.importorskip("xgboost")
    from twistboard.outputs import read_history
    from twistformer.training.xgboost import BoostingHistory

    config = parse_config({"driver": {"name": "synthetic"}, "model": {"family": "xgboost"},
                           "training": {"epochs": 8, "learning_rate": .1, "min_learning_rate": .001, **options}})
    training = config.training
    assert training.scheduler == options.get("scheduler", "none")
    optimizer = torch.optim.SGD([torch.zeros(())], lr=training.learning_rate)
    scheduler = make_scheduler(optimizer, training)
    expected = []
    for _ in range(rounds):
        expected.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        step_scheduler(scheduler, .5)

    native_rates = []

    class NativeRates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            """Read the rate before the next tree update"""
            booster = json.loads(model.save_config())["learner"]["gradient_booster"]
            native_rates.append(float(booster["tree_train_param"]["learning_rate"]))
            return False

    timer = SimpleNamespace(record={"attempts": [{"started_at": "attempt"}]}, elapsed=lambda: 0, persist=lambda: None)
    history = BoostingHistory(tmp_path, training, timer, "val_scores")
    matrix = xgb.DMatrix(np.zeros((4, 1)), label=[0, 0, 1, 1], nthread=1)
    model = xgb.train({"eta": history.learning_rate, "nthread": 1, "eval_metric": ["logloss", "error", "auc"]},
                      matrix, num_boost_round=history.round_limit,
                      evals=[(matrix, name) for name in ("train", "val", "val_scores")],
                      callbacks=[history, NativeRates()], verbose_eval=False)
    assert model.num_boosted_rounds() == rounds
    np.testing.assert_allclose(native_rates, expected)
    np.testing.assert_allclose([row["lr"] for row in read_history(tmp_path)], expected)


@pytest.mark.parametrize("saved_epoch", [1, 3, 4, 7, 8, 15, 16, 30, 31, 32, 33])
def test_pelican_schedule_and_resume(saved_epoch):
    """
    Check cosine restarts and resume across all schedule transitions

    Args:
        saved_epoch:  Zero-based epoch saved before resume
    Returns:
        None: Both parameter groups retain the reference learning rates
    """
    cfg = training_config(
        epochs=35,
        scheduler="cosine_restarts",
        warmup_epochs=4,
        learning_rate=0.001,
    ).training
    groups = [
        {"params": [nn.Parameter(torch.ones(()))], "lr": lr} for lr in (0.001, 0.0002)
    ]
    optimizer = torch.optim.SGD(groups)
    scheduler = make_scheduler(optimizer, cfg)
    reference_optimizer = torch.optim.SGD(
        [{"params": [nn.Parameter(torch.ones(()))], "lr": lr} for lr in (0.001, 0.0002)]
    )
    reference = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        reference_optimizer,
        T_0=4,
        T_mult=2,
    )
    expected = [[lr * factor / 4 for lr in (0.001, 0.0002)] for factor in (1, 2, 3, 4)]
    for _ in range(28):
        expected.append(reference.get_last_lr())
        reference_optimizer.step()
        reference.step()
    last_cosine = expected[-1]
    expected.extend([[lr * 0.5**step for lr in last_cosine] for step in (1, 2, 3)])
    for epoch, rates in enumerate(expected):
        np.testing.assert_allclose([g["lr"] for g in optimizer.param_groups], rates)
        optimizer.step()
        step_scheduler(scheduler, 1.0)
        if epoch == saved_epoch:
            saved_optimizer = deepcopy(optimizer.state_dict())
            saved_scheduler = deepcopy(scheduler.state_dict())
    resumed = make_scheduler(optimizer, cfg)
    optimizer.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)
    for rates in expected[saved_epoch + 1 :]:
        np.testing.assert_allclose([g["lr"] for g in optimizer.param_groups], rates)
        optimizer.step()
        step_scheduler(resumed, 1.0)


@pytest.mark.parametrize(
    "kind", ["none", "cosine", "cosine_restarts", "thirds", "plateau"]
)
def test_ema_and_schedule_resume_exact(tmp_path, kind):
    cfg = training_config(
        scheduler=kind, warmup_epochs=2, decay_epochs=1, ema=True, ema_decay=0.8
    )
    root = tmp_path / "full"
    Trainer(root).fit(build_experiment(cfg))
    for checkpoint_epoch in (0, 2):
        resumed = tmp_path / f"resume_{checkpoint_epoch}"
        Trainer(
            resumed, resume=root / f"checkpoints/epoch_{checkpoint_epoch:05d}.pt"
        ).fit(build_experiment(cfg))
        reference = read_checkpoint(root / "checkpoints/last.pt")
        restored = read_checkpoint(resumed / "checkpoints/last.pt")
        for component in ("model", "ema"):
            a = (
                reference[component]
                if component == "model"
                else reference[component]["model"]
            )
            b = (
                restored[component]
                if component == "model"
                else restored[component]["model"]
            )
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        assert (
            reference["ema"]["updates"]
            == restored["ema"]["updates"]
            == reference["step"]
        )
        assert reference["scheduler"] == restored["scheduler"]
    with (root / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["lr"]) == cfg.training.learning_rate / 2
    predictor = load_predictor(root / "checkpoints/last.pt", "cpu")
    for name, value in predictor.model.state_dict().items():
        torch.testing.assert_close(
            value, reference["ema"]["model"][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("ema_eval", [False, True])
def test_ema_inference_selection_and_fresh_process(tmp_path, ema_eval):
    cfg = training_config(epochs=2, ema=True, ema_decay=0.8, ema_eval=ema_eval)
    checkpoint = Trainer(tmp_path / "run").fit(build_experiment(cfg))
    state = read_checkpoint(checkpoint)
    expected = state["ema"]["model"] if ema_eval else state["model"]
    predictor = load_predictor(checkpoint, "cpu")
    for name, value in predictor.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    output = tmp_path / "fresh.npz"
    subprocess.run(
        [
            sys.executable,
            "scripts/predict.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    prediction = predictor.predict(
        predictor.experiment.driver.dataset("test"), batch_size=3
    )
    with np.load(output) as saved:
        np.testing.assert_allclose(
            saved["logits"], prediction.logits.numpy(), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "section,options",
    [
        ("model", {"family": "twistformer", "twistor_geometry_scale": -1.0}),
        ("model", {"family": "twistformer", "twistor_representation_scale": -1.0}),
        ("model", {"twistor_qk": True}),
        ("model", {"family": "twistformer", "twistor_residual_scaling": "invalid"}),
        ("model", {"auxiliary_features": ["charge"]}),
        ("model", {"family": "pelican", "auxiliary_features": ["charge", "charge"]}),
        ("training", {"warmup_epochs": 30}),
        ("training", {"ema_decay": 1.0}),
        ("training", {"min_learning_rate": 2.0}),
        ("training", {"plateau_factor": 1.0}),
        ("training", {"scheduler": "cosine_restarts", "restart_period": 0}),
        ("training", {"scheduler": "cosine_restarts", "restart_period": 1.5}),
        ("training", {"scheduler": "cosine_restarts", "restart_mult": 0}),
        ("training", {"scheduler": "cosine_restarts", "restart_mult": 1.5}),
        ("training", {"scheduler": "cosine_restarts", "decay_epochs": -1}),
        ("training", {"scheduler": "cosine_restarts", "decay_epochs": 30}),
        ("training", {"scheduler": "cosine_restarts", "decay_epochs": 1.5}),
        ("training", {"scheduler": "cosine_restarts", "decay_gamma": 0}),
        ("training", {"scheduler": "cosine_restarts", "decay_gamma": 1.1}),
    ],
)
def test_priority_config_validation(section, options):
    with pytest.raises(ValueError):
        parse_config({"driver": {"name": "synthetic"}, section: options})


@pytest.mark.parametrize("family", ["twistformer", "pelican"])
def test_auxiliary_normalization_checkpoint_roundtrip(tmp_path, family):
    p4 = np.tile(
        np.array([[[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0]]], dtype=np.float32),
        (8, 1, 1),
    )
    features = np.arange(32, dtype=np.float32).reshape(8, 2, 2)
    path = tmp_path / "events.npz"
    np.savez(path, features=features, p4=p4, targets=np.arange(8) % 2)
    cfg = parse_config(
        {
            "driver": {
                "name": "generic",
                "files": {split: str(path) for split in ("train", "val", "test")},
                "features": ["unused", "charge"],
                "geometry": "p4",
                "outputs": ["a", "b"],
            },
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "auxiliary_features": ["charge"],
            },
            "training": {"epochs": 1, "batch_size": 4, "ema": True, "device": "cpu"},
        }
    )
    checkpoint = Trainer(tmp_path / "run").fit(build_experiment(cfg))
    predictor = load_predictor(checkpoint, "cpu")
    np.testing.assert_allclose(
        predictor.preprocessor.state["feature_mean"].numpy(), features.mean((0, 1))
    )
    samples = predictor.experiment.driver.dataset("test")
    prediction = predictor.predict(samples, batch_size=4).logits
    modified = [
        replace(sample, features=sample.features + torch.tensor([0.0, 10.0]))
        for sample in samples
    ]
    assert not torch.allclose(
        predictor.predict(modified, batch_size=4).logits, prediction
    )
    second = load_predictor(checkpoint, "cpu")
    torch.testing.assert_close(
        second.predict(samples, batch_size=4).logits, prediction, atol=0, rtol=0
    )


def test_ema_skips_overflowed_optimizer_steps(tmp_path, monkeypatch):
    cfg = training_config(epochs=1, amp=True, amp_dtype="fp16", ema=True)
    experiment = build_experiment(cfg)
    original_scaler = torch.amp.GradScaler
    monkeypatch.setattr(
        torch.amp,
        "GradScaler",
        lambda *a, **kw: original_scaler(*a, **kw, init_scale=128.0),
    )
    calls = 0

    def overflow_once(gradient):
        """
        Inject one overflowing gradient into a real scaled optimizer step

        Args:
            gradient:  Parameter gradient
        Returns:
            gradient: Infinite values on the first backward call only
        """
        nonlocal calls
        calls += 1
        return torch.full_like(gradient, torch.inf) if calls == 1 else gradient

    next(experiment.model.parameters()).register_hook(overflow_once)
    Trainer(tmp_path / "run").fit(experiment)
    state = read_checkpoint(tmp_path / "run/checkpoints/last.pt")
    assert calls == 3
    assert state["step"] == state["ema"]["updates"] == 2
    assert state["scaler"]["scale"] == 64.0


def test_twistor_qk_transport_scores_and_zero_update_scales(monkeypatch):
    import importlib

    module = importlib.import_module("twistformer.models.twistformer")
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "twistor_qk": True,
                "representation_channels": 2,
            },
        }
    ).model
    block = TwistorBlock(cfg).eval()
    with torch.no_grad():
        block.step.bias.fill_(0.4)
    p4 = torch.tensor(
        [[[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    tw, metric = twistors_from_p4(p4, mask)
    z = projective_normalize(tw.Z, metric)
    h = torch.randn(1, 3, 8)
    channels = z[..., None] * torch.randn(1, 3, 1, 2, dtype=torch.float64)
    captured = {}
    attend = module.scaled_attention
    transport = module.transport_channels

    def capture_attention(q, k, v, valid, bias, dropout=0.0, backend="sdpa"):
        """
        Record the scalar attention distribution

        Args:
            q, k, v:  Scalar attention projections
            valid:  Valid particle mask
            bias:  Geometric scores
            dropout:  Scalar attention dropout probability
            backend:  Configured attention kernel
        Returns:
            context: Original scalar attention result
        """
        captured["weights"] = (
            ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5 + bias)
            .softmax(-1)
            .double()
            .mean(1)
        )
        return attend(q, k, v, valid, bias, dropout=dropout, backend=backend)

    def capture_transport(z, overlap, attention, values, mask, mode):
        """
        Check that geometric transport shares the scalar attention distribution

        Args:
            z, overlap:  Normalized twistors and overlaps
            attention:  Transport weights
            values:  Complex representation channels
            mask, mode:  Valid entries and transport convention
        Returns:
            values: Original transported channels
        """
        torch.testing.assert_close(attention, captured["weights"])
        return transport(z, overlap, attention, values, mask, mode)

    monkeypatch.setattr(module, "scaled_attention", capture_attention)
    monkeypatch.setattr(module, "transport_channels", capture_transport)
    _, changed_z, changed_channels = block(h, z, metric, mask, channels)
    assert not torch.allclose(changed_z, z)
    assert not torch.allclose(changed_channels, channels)
    block.geometry_scale = block.representation_scale = 0.0
    _, unchanged_z, unchanged_channels = block(h, z, metric, mask, channels)
    torch.testing.assert_close(unchanged_z, z)
    torch.testing.assert_close(unchanged_channels, channels)
