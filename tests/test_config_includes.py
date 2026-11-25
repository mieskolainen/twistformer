# Shared YAML steering
#
# mikael.mieskolainen@cern.ch (2026)

from pathlib import Path
from dataclasses import replace

import pytest
import yaml

from drivers.base import build_driver
from twistformer.api.config import (
    EvaluationConfig,
    load_config,
    load_yaml_mapping,
    save_config,
)
from twistformer.cli.evaluate import evaluation_override


@pytest.mark.parametrize("card,key,expected", [
    ("tau3_classification/transformer", "driver.signal_class", None),
    ("toptag/transformer", "driver.signal_class", 1),
    ("pi0gamma/transformer_angular", "driver.signal_class", 1),
    ("synthetic/gatr", "driver.signal_class", 1),
    ("generic/parametric", "driver.signal_class", 1),
    ("generic/reweighting", "driver.source.signal_class", 1),
])
def test_training_signal_class(card, key, expected):
    """
    Read the signal index from the training card and apply a class override

    Args:
        card:  Classification config below configs
        key:  Signal index in the source driver
        expected:  Signal index in the preset
    """
    file = Path("configs") / f"{card}.yml"
    config = load_config(file)
    assert config.driver.get("source", config.driver)["signal_class"] == expected
    assert build_driver(config.driver, config.task).task_spec.signal_class == expected
    config = load_config(file, [f"{key}=0"])
    assert build_driver(config.driver, config.task).task_spec.signal_class == 0


@pytest.mark.parametrize(
    "group,name,reference",
    [
        ("toptag", "transformer", "gatr_projective"),
        ("toptag", "twistformer", "gatr_projective"),
    ],
)
def test_transformer_training_matches_gatr(group, name, reference):
    """
    Match optimizer budgets, warmup and learning-rate schedules to GATr

    Args:
        group:  Physics driver directory
        name:  Model card
        reference:  GATr reference card for the same task
    """
    root = Path(__file__).resolve().parents[1] / "configs" / group
    actual = load_config(root / f"{name}.yml").training
    expected = load_config(root / f"{reference}.yml").training
    for key in (
        "epochs",
        "max_steps",
        "batch_size",
        "accumulation",
        "learning_rate",
        "optimizer",
        "optimizer_betas",
        "optimizer_eps",
        "weight_decay",
        "decay_norm_bias",
        "scheduler",
        "scheduler_interval",
        "warmup_epochs",
        "min_learning_rate",
        "grad_clip",
        "token_dropout",
        "token_min_keep",
    ):
        assert getattr(actual, key) == getattr(expected, key), key


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tau3_comparison_inputs_and_training(task):
    """
    Verify shared tau event selections and model-specific training schedules

    Args:
        task:  Tau classification or regression
    """
    root = Path(__file__).resolve().parents[1] / "configs" / f"tau3_{task}"
    reference = load_config(root / "common.yml")
    for name in ("transformer", "twistformer", "pelican", "gatr_projective"):
        actual = load_config(root / f"{name}.yml")
        assert actual.driver == reference.driver
        expected = reference.training
        if name in {"transformer", "twistformer"}:
            expected = replace(expected, learning_rate=3e-4, scheduler="none", warmup_epochs=0)
        elif name == "gatr_projective" and task == "classification":
            expected = replace(expected, epochs=50)
        assert actual.training == expected
        assert actual.task == reference.task
        assert actual.evaluation == reference.evaluation
    assert reference.driver["max_hits"] == 1024
    assert reference.driver["momentum_proxy"]


@pytest.mark.parametrize(
    "group,name",
    [
        ("toptag", "transformer"),
        ("toptag", "pelican"),
        ("toptag", "twistformer"),
        ("tau3_classification", "transformer"),
        ("tau3_regression", "transformer"),
        ("tau3_classification", "twistformer"),
        ("tau3_regression", "twistformer"),
        ("tau3_classification", "pelican"),
        ("tau3_regression", "pelican"),
    ],
)
def test_shared_evaluation(group, name):
    """
    Check that physics cards inherit their shared evaluation bins
    """
    root = Path(__file__).resolve().parents[1] / "configs" / group
    shared = load_yaml_mapping(root / "evaluation.yml")["evaluation"]
    card = root / f"{name}.yml"
    assert "evaluation" not in yaml.safe_load(card.read_text())
    assert load_config(card).evaluation == EvaluationConfig(**shared)


def test_nested_includes_and_override_priority(tmp_path, monkeypatch):
    """
    Resolve relative includes before local steering and CLI overrides
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "base.yml").write_text(
        "evaluation:\n  batch_size: 2048\n  efficiencies: [0.3, 0.5]\n"
    )
    (shared / "evaluation.yml").write_text(
        "include: base.yml\nevaluation:\n  efficiencies: [0.5]\n"
    )
    (tmp_path / "extra.yml").write_text("evaluation:\n  batch_size: 1024\n")
    card = tmp_path / "train.yml"
    card.write_text(
        "include: [shared/evaluation.yml, extra.yml]\n"
        "driver:\n  name: synthetic\n"
        "evaluation:\n  batch_size: 512\n"
    )
    monkeypatch.chdir(shared)
    assert load_config(card).evaluation.batch_size == 512
    config = load_config(card, ["evaluation.batch_size=256"])
    assert config.evaluation.batch_size == 256
    assert config.evaluation.efficiencies == [0.5]
    card.write_text("include: [shared/evaluation.yml, extra.yml]\n")
    assert load_yaml_mapping(card)["evaluation"]["batch_size"] == 1024

    saved = tmp_path / "resolved.yml"
    save_config(config, saved)
    (shared / "base.yml").unlink()
    assert "include" not in yaml.safe_load(saved.read_text())
    assert load_config(saved).to_dict() == config.to_dict()


@pytest.mark.parametrize("override", ["training..epochs=2", "training.epochs.value=2"])
def test_malformed_override_path(tmp_path, override):
    """Reject empty option names and nested overrides of scalar settings"""
    card = tmp_path / "card.yml"
    card.write_text("driver: {name: synthetic}\ntraining: {epochs: 3}\n")
    with pytest.raises(ValueError, match="Overrides|must be a mapping"):
        load_config(card, [override])


def test_evaluation_only_include(tmp_path):
    """
    Use the same include rules for standalone evaluation steering
    """
    (tmp_path / "base.yml").write_text(
        "evaluation:\n  batch_size: 2048\n  efficiencies: [0.3, 0.5]\n"
    )
    card = tmp_path / "evaluation.yml"
    card.write_text("include: base.yml\nevaluation:\n  batch_size: 1024\n")
    config = evaluation_override(card, EvaluationConfig())
    assert config.batch_size == 1024
    assert config.efficiencies == [0.3, 0.5]


def test_include_cycle(tmp_path):
    """
    Reject recursive steering with a readable error
    """
    first = tmp_path / "first.yml"
    first.write_text("include: second.yml\n")
    (tmp_path / "second.yml").write_text("include: first.yml\n")
    with pytest.raises(ValueError, match="YAML include cycle"):
        load_yaml_mapping(first)


@pytest.mark.parametrize(
    "document", ["[]", "", "include: null", "include: [4]", "include: ''"]
)
def test_invalid_include_document(tmp_path, document):
    """
    Reject malformed steering and include paths
    """
    card = tmp_path / "invalid.yml"
    card.write_text(document)
    with pytest.raises(ValueError, match="YAML"):
        load_yaml_mapping(card)


def test_missing_include(tmp_path):
    """
    Report missing steering instead of using incomplete defaults
    """
    card = tmp_path / "train.yml"
    card.write_text("include: missing.yml\n")
    with pytest.raises(FileNotFoundError, match="missing.yml"):
        load_yaml_mapping(card)
