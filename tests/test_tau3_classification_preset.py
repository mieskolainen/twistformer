# Tau3 preset and shared optimizer validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
import pytest
import torch
from twistformer import load_config, build_experiment, Trainer, load_predictor
from twistformer.api.config import parse_config, TrainingConfig
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples, drop_tokens
from twistformer.data.preprocessing import Preprocessor
from twistformer.training.engine import make_optimizer, make_loader, configure_precision
from twistformer.training.checkpoint import read_checkpoint
from twistformer.evaluation.runner import evaluate


def test_tau3_preset_optimizer_groups_and_runtime(monkeypatch):
    """
    Check the tau classifier architecture and optimizer settings

    Args:
        monkeypatch:  Runtime patch fixture
    """
    cfg = load_config("configs/tau3_classification/transformer.yml")
    exp = build_experiment(cfg)
    assert cfg.model.width == 64 and cfg.model.expansion == 2
    assert cfg.model.depth == 6 and cfg.model.heads == 4
    assert cfg.model.pooling == "mean" and cfg.model.dropout == 0.01
    assert cfg.model.norm == cfg.model.head_norm == "rms"
    assert cfg.model.embedding == [64, 64, 128]
    assert cfg.model.embedding_norm == cfg.model.ffn_norm == "layer"
    assert cfg.model.ffn_activation == "geglu"
    assert cfg.model.head_layers == 3 and cfg.model.head_expansion == 2
    assert cfg.model.head_residual and cfg.model.head_zero_init
    assert cfg.driver["input_mode"] == "sim"
    reco = load_config("configs/tau3_classification/transformer.yml", ["driver.input_mode=reco-cluster"])
    assert build_experiment(reco).driver.options.input_mode == "reco-cluster"
    assert not exp.model.head.linear.weight.count_nonzero()
    assert not exp.model.head.linear.bias.count_nonzero()
    assert cfg.task.balance_classes
    assert cfg.training.token_dropout == 0.01
    assert cfg.training.compile and cfg.training.compile_mode == "default"
    assert cfg.training.amp_dtype == "bf16"
    assert cfg.training.save_every == 1 and cfg.training.keep_checkpoints == 0
    assert cfg.evaluation.efficiencies == [0.3, 0.5, 0.9]
    optimizer = make_optimizer(exp.model, cfg.training)
    muon, adamw = optimizer.optimizers
    assert isinstance(muon, torch.optim.Muon) and isinstance(adamw, torch.optim.AdamW)
    for group in adamw.param_groups:
        assert group["betas"] == (0.9, 0.98) and group["eps"] == 1e-8
    for group in optimizer.param_groups:
        assert group["lr"] == 3e-4
        for parameter in group["params"]:
            name = next(
                name for name, p in exp.model.named_parameters() if p is parameter
            )
            excluded = (
                parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
            )
            assert group["weight_decay"] == (0 if excluded else 0.01)
    configure_precision(cfg.training)
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch.backends.cudnn.conv.fp32_precision == "tf32"
    configure_precision(replace(cfg.training, tf32=False))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = make_loader([], cfg.training, train=False)
    assert loader.num_workers == 1 and loader.prefetch_factor == 2


def test_token_dropout_retains_support_and_alignment():
    sample = Sample(
        torch.arange(24).reshape(6, 4).float(),
        p4=torch.arange(24).reshape(6, 4).float(),
        token_weights=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    batch = collate_samples(
        [
            sample,
            replace(
                sample,
                features=sample.features[:1],
                p4=sample.p4[:1],
                token_weights=torch.ones(1),
            ),
        ]
    )
    original = batch.features.clone()
    result = drop_tokens(batch, 0.999, min_keep=3)
    assert result.mask.sum(1).tolist()[0] >= 3 and result.mask.sum(1)[1] == 1
    assert not (result.mask & ~batch.mask).any()
    assert (result.token_weights.sum(1) > 0).all()
    torch.testing.assert_close(result.features, result.p4)
    torch.testing.assert_close(batch.features, original)
    assert drop_tokens(batch, 0) is batch


def test_class_balance_is_weighted_fitted_and_checkpointable():
    samples = [
        Sample(torch.ones(2, 1) * i, torch.tensor(label), weight=weight)
        for i, (label, weight) in enumerate([(0, 1.0), (0, 3.0), (1, 2.0)])
    ]
    pre = Preprocessor(normalize_features=False).fit(samples, class_count=2)
    torch.testing.assert_close(pre.state["class_counts"], torch.tensor([2, 1]))
    torch.testing.assert_close(pre.state["class_weights"], torch.tensor([0.75, 1.5]))
    assert sum(pre.transform(sample).weight for sample in samples[:2]) == 3
    assert pre.transform(samples[2]).weight == 3
    restored = Preprocessor(pre.state)
    assert restored.transform(replace(samples[2], weight=4)).weight == 6
    assert restored.transform(replace(samples[2], target=None, weight=4)).weight == 4
    with pytest.raises(ValueError, match="positive training weight"):
        Preprocessor().fit(samples[:2], class_count=2)
    with pytest.raises(ValueError, match="classification"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "task": {"name": "gaussian_regression", "balance_classes": True},
            }
        )


@pytest.mark.parametrize("family", ["transformer", "transformer_projective", "pelican"])
def test_tau3_class_balancing_training_and_reload(tmp_path, monkeypatch, family):
    """
    Train tau classifiers and restore class weights and evaluation outputs

    Args:
        tmp_path:  Temporary run directory
        monkeypatch:  Replace ROOT reads with detector hits
        family:  Classification model family
    """
    pytest.importorskip("awkward")
    pytest.importorskip("uproot")
    from drivers.tau3_classification.driver import Driver
    from drivers.tau3_classification.reader import Tau3Chunk

    def read(self, path):
        size = int(path)
        x = np.arange(size * 12, dtype=np.float32).reshape(size, 3, 4) / 100
        yield Tau3Chunk(
            x.reshape(-1, 4),
            np.arange(size + 1) * 3,
            np.ones(size),
            np.arange(size),
        )

    monkeypatch.setattr(Driver, "_read", read)
    cfg = load_config(
        f"configs/tau3_classification/{family}.yml",
        [
            'driver.files=["12", "16", "20"]',
            "training.device=cpu",
            "training.workers=0",
            "training.compile=false",
            "training.epochs=1",
            "training.warmup_epochs=0",
            "training.batch_size=16",
        ],
    )
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "run").fit(exp)
    state = read_checkpoint(best)
    assert state["config"]["model"] == cfg.to_dict()["model"]
    assert state["model"]["head.linear.weight"].count_nonzero()
    torch.testing.assert_close(
        state["preprocessing"]["class_counts"], torch.tensor([8, 11, 14])
    )
    predictor = load_predictor(best, "cpu")
    torch.testing.assert_close(
        predictor.preprocessor.state["class_weights"],
        state["preprocessing"]["class_weights"],
    )
    result = evaluate(
        predictor,
        predictor.experiment.driver.dataset("test"),
        tmp_path / "evaluation",
        plots=False,
    )
    for name in exp.driver.task_spec.outputs:
        assert [
            point["requested_efficiency"]
            for point in result["classes"][name]["operating_points"]
        ] == cfg.evaluation.efficiencies
    adam_betas = [group["betas"] for group in state["optimizer"]["param_groups"] if "betas" in group]
    assert adam_betas and all(betas == (0.9, 0.98) for betas in adam_betas)
    configure_precision(TrainingConfig())
