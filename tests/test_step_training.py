# Lion updates and optimizer-step training budgets
# Reference: https://arxiv.org/abs/2302.06675
#
# mikael.mieskolainen@cern.ch (2026)

import pytest
import torch

from twistformer import Trainer, build_experiment, parse_config, load_predictor
from twistformer.api.contracts import LossResult
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.engine import clip_gradients, make_optimizer, normalize_gradients
from twistformer.training.optimizers import Lion
from twistformer.training.schedulers import make_scheduler


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_muon_adamw_matches_native_updates_after_resume(device):
    """
    Match scheduled matrix and auxiliary updates across checkpoint restore

    Args:
        device:  CPU or CUDA test device
    """
    from copy import deepcopy

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.LayerNorm(3), torch.nn.Embedding(4, 3))
    model.register_parameter("blade_weights", torch.nn.Parameter(torch.ones(3, 2, 4)))
    model.to(device)
    reference = deepcopy(model)
    cfg = parse_config({"driver": {"name": "synthetic"}, "training": {
        "optimizer": "muon_adamw", "learning_rate": 0.002, "weight_decay": 0.03,
        "decay_norm_bias": False, "optimizer_betas": [0.85, 0.98],
        "fused_optimizer": device == "cuda",
    }}).training
    optimizer = make_optimizer(model, cfg)
    muon = torch.optim.Muon([reference[0].weight], lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                            adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW([
        {"params": [reference.blade_weights, reference[2].weight], "weight_decay": cfg.weight_decay},
        {"params": [reference[0].bias, *reference[1].parameters()], "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=tuple(cfg.optimizer_betas), eps=cfg.optimizer_eps, fused=cfg.fused_optimizer)
    for step in range(2):
        if step == 1:
            checkpoint = deepcopy(optimizer.state_dict())
            model = deepcopy(model)
            optimizer = make_optimizer(model, cfg)
            optimizer.load_state_dict(checkpoint)
        for current in (optimizer, muon, adamw):
            for group in current.param_groups:
                group["lr"] = cfg.learning_rate / (step + 1)
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            gradient = torch.randn_like(expected)
            parameter.grad = gradient.clone()
            expected.grad = gradient
        optimizer.step()
        muon.step()
        adamw.step()
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, expected)


def test_gradient_clipping_recovers_finite_norm_overflow():
    """
    Clip finite large gradients without overflowing float32 norm reduction
    """
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, 1e30)
    model.bias.grad = torch.full_like(model.bias, -1e30)
    norm = clip_gradients(model, 1.5)
    assert torch.isfinite(norm)
    clipped = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    assert clipped.norm().item() == pytest.approx(1.5)
    assert clipped.tolist() == pytest.approx([3**-0.5 * 1.5, 3**-0.5 * 1.5, -3**-0.5 * 1.5])


def test_gradient_clipping_reports_invalid_parameters():
    """
    Report invalid gradients without changing finite parameter gradients
    """
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.ones_like(model.bias)
    with pytest.raises(FloatingPointError, match="Nonfinite gradients.*weight"):
        clip_gradients(model, 1.5)
    torch.testing.assert_close(model.bias.grad, torch.ones_like(model.bias))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_gradient_normalization(device):
    """
    Match scalar division for mixed real and complex gradient groups

    Args:
        device:  CPU or CUDA tensor device
    Returns:
        None: Gradient values and storage match the in-place reference
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    parameters, expected, pointers = [], [], []
    for dtype in (torch.float32, torch.float32, torch.float64, torch.complex64):
        parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype, device=device))
        parameter.grad = torch.randn(3, 2, dtype=dtype, device=device).T
        expected.append(parameter.grad.clone() / 7.5)
        pointers.append(parameter.grad.data_ptr())
        parameters.append(parameter)
    unused = torch.nn.Parameter(torch.ones(1, device=device))
    normalize_gradients(iter([*parameters, unused]), 7.5)
    for parameter, reference, pointer in zip(parameters, expected, pointers, strict=True):
        torch.testing.assert_close(parameter.grad, reference)
        assert parameter.grad.data_ptr() == pointer
    assert unused.grad is None
    normalize_gradients([unused], 7.5)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_accumulation_prevents_update(tmp_path, monkeypatch, nonfinite):
    """
    Reject nonfinite loss before updating a partial accumulation window

    Args:
        tmp_path:  Training output directory
        monkeypatch:  Replace the first loss and guard optimizer updates
        nonfinite:  Invalid loss contribution
    Returns:
        None: All parameters remain unchanged after the rejected window
    """
    cfg = parse_config({
        "driver": {"name": "synthetic", "train_size": 5, "val_size": 2, "max_tokens": 3},
        "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
        "training": {"device": "cpu", "epochs": 1, "batch_size": 2, "accumulation": 4, "optimizer": "sgd"},
    })
    experiment = build_experiment(cfg)
    expected = [parameter.detach().clone() for parameter in experiment.model.parameters()]
    compute_loss = experiment.task.compute_loss
    calls = []

    def invalid_loss(prediction, batch):
        """
        Add a nonfinite first loss with otherwise finite gradients

        Args:
            prediction:  Model predictions
            batch:  Weighted event batch
        Returns:
            loss: Original weighted loss with one invalid contribution
        """
        result = compute_loss(prediction, batch)
        calls.append(len(batch.mask))
        return LossResult(result.numerator + nonfinite, result.normalizer) if len(calls) == 1 else result

    def forbidden_step(*args, **kwargs):
        """
        Fail if an invalid loss reaches the optimizer
        """
        pytest.fail("Nonfinite loss reached the optimizer")

    monkeypatch.setattr(experiment.task, "compute_loss", invalid_loss)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden_step)
    with pytest.raises(FloatingPointError, match="Nonfinite loss in accumulation window"):
        Trainer(tmp_path / "run").fit(experiment)
    assert calls == [2, 2, 1]
    for parameter, reference in zip(experiment.model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)


def test_lion_reference_updates_and_state_reload():
    """
    Check sign momentum, decoupled decay and restored optimizer updates
    """
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Lion([parameter], lr=0.1, betas=(0.9, 0.99), weight_decay=0.2)
    momentum = torch.zeros(2)
    expected = parameter.detach().clone()
    for gradient in (torch.tensor([2.0, -1.0]), torch.tensor([-0.01, 3.0])):
        parameter.grad = gradient
        direction = (0.9 * momentum + 0.1 * gradient).sign()
        expected = expected * 0.98 - 0.1 * direction
        momentum = 0.99 * momentum + 0.01 * gradient
        optimizer.step()
        torch.testing.assert_close(parameter, expected)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], momentum)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = Lion([restored_parameter])
    import copy

    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parameter.grad = restored_parameter.grad = torch.tensor([-2.0, 4.0])
    optimizer.step()
    restored.step()
    torch.testing.assert_close(restored_parameter, parameter)


def test_zero_weight_window_skips_update(tmp_path, monkeypatch):
    """
    Skip empty likelihood windows and update the final partial window

    Args:
        tmp_path:  Training output directory
        monkeypatch:  Set the first accumulation window to zero weight
    Returns:
        None: Only the two positively weighted windows advance the optimizer
    """
    cfg = parse_config({
        "driver": {"name": "synthetic", "train_size": 5, "val_size": 2, "max_tokens": 3},
        "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
        "training": {"device": "cpu", "epochs": 1, "batch_size": 1, "accumulation": 2, "optimizer": "sgd"},
    })
    experiment = build_experiment(cfg)
    compute_loss = experiment.task.compute_loss
    calls = []

    def zero_first_window(prediction, batch):
        """
        Remove the first two batches from the weighted likelihood

        Args:
            prediction:  Model predictions
            batch:  Weighted event batch
        Returns:
            loss: Zero weight for the first accumulation window
        """
        result = compute_loss(prediction, batch)
        calls.append(len(batch.mask))
        return LossResult(result.numerator * 0, result.normalizer * 0) if len(calls) <= 2 else result

    monkeypatch.setattr(experiment.task, "compute_loss", zero_first_window)
    Trainer(tmp_path / "run").fit(experiment)
    state = read_checkpoint(tmp_path / "run/checkpoints/last.pt")
    assert state["step"] == 2


def test_step_cosine_and_resume_state():
    """
    Check cosine rates at optimizer boundaries and preserve scheduler phase
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "training": {
                "max_steps": 4,
                "scheduler_interval": "step",
                "learning_rate": 0.1,
            },
        }
    ).training
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = Lion([p], lr=cfg.learning_rate)
    schedule = make_scheduler(optimizer, cfg)
    rates = []
    for _ in range(2):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()
    replacement = make_scheduler(optimizer, cfg)
    replacement.load_state_dict(schedule.state_dict())
    optimizer.param_groups[0]["lr"] = schedule.get_last_lr()[0]
    for _ in range(2):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        replacement.step()
    assert rates == pytest.approx([0.1, 0.085355339, 0.05, 0.014644661])
    assert replacement.get_last_lr() == [0.0]


def test_training_stops_at_optimizer_budget_with_accumulation(tmp_path):
    """
    Stop within an epoch at successful optimizer steps and reload predictions
    """
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 20,
                "val_size": 4,
                "test_size": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
            "task": {"balance_classes": True},
            "training": {
                "epochs": 5,
                "batch_size": 2,
                "eval_batch_size": 4,
                "max_steps": 3,
                "scheduler_interval": "step",
                "optimizer": "lion",
                "accumulation": 2,
                "device": "cpu",
            },
        }
    )
    root = tmp_path / "run"
    best = Trainer(root).fit(build_experiment(cfg))
    state = read_checkpoint(root / "checkpoints/last.pt")
    assert state["step"] == 3 and state["epoch"] == 0
    assert state["scheduler"]["last_epoch"] == 3
    assert state["optimizer"]["param_groups"][0]["lr"] == 0.0
    predictor = load_predictor(best, "cpu")
    prediction = predictor.predict(predictor.experiment.driver.dataset("test"))
    assert prediction.logits.shape == (4, 1)
    torch.testing.assert_close(
        prediction.probabilities[:, 1], prediction.logits[:, 0].sigmoid()
    )
    torch.testing.assert_close(
        predictor.predict_prob(predictor.experiment.driver.dataset("test")),
        prediction.probabilities,
    )
    # Completed step budgets cannot restart extra optimizer updates
    assert (
        Trainer(root, resume=root / "checkpoints/last.pt").fit(build_experiment(cfg))
        == best
    )
