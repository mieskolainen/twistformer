# Density-ratio closure and shared classifier training
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from twistformer import (
    Sample, parse_config, build_experiment, Trainer, load_predictor,
    ClassifierReweighter, FoldReweighter, WeightedDataset, fit_reweighter,
)
from twistformer.training.checkpoint import read_checkpoint
from twistformer.evaluation.runner import collect_predictions


class Posterior:
    def __init__(self, probability):
        """
        Store an analytic posterior table

        Args:
            probability:  Class probabilities in event order
        Returns:
            None: Probability table is stored
        """
        self.probability = probability

    def predict_prob(self, samples, batch_size=1024):
        """
        Return the analytic class posterior

        Args:
            samples:  Input events
            batch_size:  Unused batch limit
        Returns:
            probability: Fixed posterior table
        """
        return self.probability


def test_density_ratio_closure_and_physical_weights():
    """
    Recover equal two-bin densities from unequal priors and event weights
    """
    samples = [Sample(torch.tensor([[float(i % 2)]]), torch.tensor(i // 2), weight=w, id=str(i))
               for i, w in enumerate([2.0, 2.0, 1.0, 3.0])]
    # Prior ratio 3:1 with p_reference=(1/2,1/2) and p_source=(1/4,3/4)
    probability = torch.tensor([[6 / 7, 1 / 7], [2 / 3, 1 / 3]] * 2)
    ratio = ClassifierReweighter(Posterior(probability), [0.75, 0.25])
    factors = ratio.factors(samples)
    torch.testing.assert_close(factors, torch.tensor([1., 1., 2., 2 / 3], dtype=torch.float64))
    weighted = WeightedDataset(samples, factors)
    torch.testing.assert_close(torch.tensor([weighted[i].weight for i in range(4)]), torch.full((4,), 2.0))
    assert [sample.weight for sample in samples] == [2, 2, 1, 3]
    assert [sample.id for sample in weighted.__getitems__([3, 0, 3])] == ["3", "0", "3"]
    capped = ClassifierReweighter(Posterior(torch.tensor([[1.0, 0.0]] * 4)), [1, 1], max_weight=7)
    assert capped.factors(samples).tolist() == [1, 1, 7, 7]


def test_multiclass_density_ratio_reference():
    """
    Match three densities to a nonzero reference class
    """
    density = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=torch.float64)
    priors = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    posterior = (density * priors[:, None]).T
    posterior /= posterior.sum(1, keepdim=True)
    samples = [Sample(torch.tensor([[float(bin_index)]]), torch.tensor(label),
                      weight=float(density[label, bin_index]), id=f"{label}:{bin_index}")
               for label in range(3) for bin_index in range(2)]
    ratio = ClassifierReweighter(Posterior(posterior.repeat(3, 1)), priors, reference_class=1)
    weighted = WeightedDataset(samples, ratio.factors(samples))
    assert [sample.weight for sample in weighted] == pytest.approx([0.5] * 6)


@pytest.mark.parametrize("probability", [torch.ones(2, 3), torch.ones(2, 2), torch.tensor([[float("nan"), 0], [0, 1]])])
def test_invalid_probabilities(probability):
    """
    Reject invalid classifier posterior outputs

    Args:
        probability:  Invalid posterior table
    """
    samples = [Sample(torch.ones(1, 1), torch.tensor(i), id=str(i)) for i in range(2)]
    with pytest.raises(ValueError, match="predict_prob"):
        ClassifierReweighter(Posterior(probability), [1, 1]).factors(samples)


def density_config(tmp_path, balance):
    """
    Make distinct source splits with weighted kinematic inputs

    Args:
        tmp_path:  Input directory
        balance:  Enable classifier class balancing
    Returns:
        config: Shared training config with the scalar observable driver
    """
    files = {}
    for split, offset in (("train", 0), ("val", 30), ("test", 60)):
        path = tmp_path / f"{split}.npz"
        np.savez(path, features=np.ones((24, 2, 1), dtype=np.float32),
                 global_features=np.column_stack((np.arange(24) + offset, np.arange(24) % 3)),
                 targets=np.arange(24) % 2, weights=np.where(np.arange(24) % 2, 3., 1.),
                 ids=np.array([f"{split}:{i}" for i in range(24)]))
        files[split] = str(path)
    return parse_config({
        "driver": {"name": "observables", "source": {
            "name": "generic", "files": files, "features": ["energy"],
            "global_features": ["pt", "rapidity"], "outputs": ["background", "signal"],
        }, "variables": ["global:pt", "global:rapidity"]},
        "model": {"width": 8, "depth": 1, "heads": 2, "dropout": 0.0},
        "task": {"balance_classes": balance, "class_weights": [2., 1.]},
        "training": {"epochs": 1, "batch_size": 16, "device": "cpu", "workers": 0},
    })


@pytest.mark.parametrize("balance", [False, True])
def test_fold_training_final_fit_and_reload(tmp_path, balance):
    """
    Exclude held-out events from fitting and retain physical evaluation weights

    Args:
        tmp_path:  Input and reweighting output directory
        balance:  Enable density classifier class balancing
    """
    config = density_config(tmp_path, balance)
    if balance:
        from twistformer.api.config import save_config
        from twistformer.cli.train import train

        card = tmp_path / "density.yml"
        save_config(config, card)
        train(["--config", str(card), "--output", str(tmp_path / "density"), "--reweight-folds", "3"])
        manifest_path = tmp_path / "density" / "reweighting.json"
    else:
        manifest_path = fit_reweighter(config, tmp_path / "density", folds=3)
    manifest = json.loads(manifest_path.read_text())
    app = build_experiment(config).driver
    source = app.source.dataset("train")
    expected = torch.zeros(len(source), dtype=torch.float64)
    for fold, record in enumerate(manifest["folds"]):
        path = manifest_path.parent / record["checkpoint"]
        predictor = load_predictor(path, "cpu")
        assert hasattr(predictor, "predict_prob")
        state = read_checkpoint(path)
        assert "optimizer" in state
        selected = state["config"]["driver"]["event_ids"]
        held_out = {key for key, value in manifest["held_out"].items() if value == fold}
        assert not held_out.intersection(selected["train"] + selected["val"])
        assert not set(selected["train"]).intersection(selected["val"])
        assert all(key.startswith("train:") for key in selected["train"] + selected["val"])
        train_pt = [int(key.split(":")[1]) for key in selected["train"]]
        assert state["preprocessing"]["feature_mean"][0].item() == pytest.approx(np.mean(train_pt))
        np.testing.assert_allclose(record["priors"], [2 / 3, 1 / 3] if balance else [2 / 5, 3 / 5], rtol=1e-6)
        subset = [app.project(source)[i] for i in range(len(source)) if source[i].id in held_out]
        values = ClassifierReweighter(predictor, record["priors"]).factors(subset)
        indices = [i for i in range(len(source)) if source[i].id in held_out]
        expected[indices] = values
    frozen = FoldReweighter(manifest_path)
    weighted = frozen.apply(source, app.source.input_spec, app.task_spec)
    torch.testing.assert_close(weighted.factors, expected)
    validation = app.source.dataset("val")
    weighted_val = frozen.apply(validation, app.source.input_spec, app.task_spec)
    expected_val = torch.stack([
        ClassifierReweighter(load_predictor(manifest_path.parent / record["checkpoint"], "cpu"), record["priors"])
        .factors(app.project(validation)) for record in manifest["folds"]
    ]).mean(0)
    torch.testing.assert_close(weighted_val.factors, expected_val)
    steering = config.to_dict()
    steering["driver"] = config.driver["source"]
    steering["reweighting"]["path"] = str(manifest_path)
    final = build_experiment(parse_config(steering))
    checkpoint = Trainer(tmp_path / "final").fit(final)
    predictor = load_predictor(checkpoint, "cpu")
    torch.testing.assert_close(predictor.preprocessor.state["reweighting_digest"], frozen.digest)
    predictions, _ = collect_predictions(predictor, final.driver.dataset("test"))
    np.testing.assert_array_equal(predictions["physical_weights"], np.where(np.arange(24) % 2, 3, 1))
    before = predictor.predict_prob(validation)
    manifest_path.rename(manifest_path.with_suffix(".hidden"))
    torch.testing.assert_close(load_predictor(checkpoint, "cpu").predict_prob(validation), before)
    manifest_path.with_suffix(".hidden").rename(manifest_path)
    manifest["max_weight"] = 50
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed since"):
        Trainer(tmp_path / "resume", resume=checkpoint).fit(build_experiment(parse_config(steering)))
    with (manifest_path.parent / manifest["folds"][0]["checkpoint"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        FoldReweighter(manifest_path)


def test_reweighting_validation_and_input_projection(tmp_path):
    """
    Reject signed densities and retain only declared scalar inputs

    Args:
        tmp_path:  Source input directory
    """
    config = density_config(tmp_path, True)
    app = build_experiment(config).driver
    raw = app.source.dataset("train")[0]
    projected = app.project([replace(raw, observables={"pt": 123., "truth": 999.})])[0]
    assert projected.features.shape == (1, 2) and projected.global_features is None
    assert projected.p4 is None and projected.parameters is None
    assert projected.features[0, 0] == raw.global_features[0]
    for invalid in ([0, 1], [float("nan"), 1], [1]):
        with pytest.raises(ValueError, match="priors"):
            ClassifierReweighter(Posterior(None), invalid)
    with pytest.raises(TypeError, match="predict_prob"):
        ClassifierReweighter(object(), [1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        ClassifierReweighter(Posterior(None), [1, 1]).factors([replace(raw, weight=-1)])
    with pytest.raises(ValueError, match="unique"):
        ClassifierReweighter(Posterior(None), [1, 1]).factors([raw, raw])
    config.task.label_smoothing = 0.1
    with pytest.raises(ValueError, match="label_smoothing"):
        fit_reweighter(config, tmp_path / "invalid")
