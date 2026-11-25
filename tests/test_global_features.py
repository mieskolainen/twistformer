# Global event feature contracts and checkpoint persistence
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import subprocess
import sys
import numpy as np
import pytest
import torch
from twistformer import (
    InputSpec,
    Sample,
    parse_config,
    build_experiment,
    Trainer,
    load_predictor,
)
from twistformer.data.batching import collate_samples, drop_tokens
from twistformer.data.preprocessing import Preprocessor
from twistformer.evaluation.runner import evaluate


def make_data(tmp_path, globals=True):
    paths = {}
    for split, shift in [("train", 0.0), ("val", 20.0), ("test", 40.0)]:
        p = np.tile(
            np.array(
                [[[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]],
                np.float32,
            ),
            (6, 1, 1),
        )
        values = dict(features=p.copy(), p4=p, targets=np.arange(6) % 2)
        if globals:
            values["global_features"] = np.stack(
                [np.arange(6) + shift, np.ones(6) * 7], -1
            ).astype(np.float32)
        path = tmp_path / f"{split}.npz"
        np.savez(path, **values)
        paths[split] = str(path)
    return paths


def config(paths, family="transformer", normalize=True):
    options = {"family": family, "width": 8, "heads": 2, "depth": 2, "dropout": 0.0}
    if family == "twistformer":
        options.update(
            pair_channels=3,
            representation_channels=2,
            twistor_transport="frame",
            reference_token="event",
            pooling="pma",
        )
    return parse_config(
        {
            "driver": {
                "name": "generic",
                "files": paths,
                "features": ["E", "px", "py", "pz"],
                "geometry": "p4",
                "outputs": ["a", "b"],
                "global_features": ["context", "constant"],
                "global_units": ["GeV", ""],
            },
            "model": options,
            "preprocessing": {"normalize_globals": normalize},
            "training": {
                "device": "cpu",
                "epochs": 1,
                "batch_size": 2,
                "eval_batch_size": 2,
            },
        }
    )


@pytest.mark.parametrize("fusion", ["readout", "tokens"])
@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican"]
)
def test_global_feature_model_gradients_and_effect(tmp_path, family, fusion):
    cfg = config(make_data(tmp_path), family)
    cfg.model.global_fusion = fusion
    exp = build_experiment(cfg)
    data = exp.driver.dataset("train")
    pre = exp.driver.create_preprocessor(config=cfg.preprocessing).fit(data)
    batch = collate_samples([pre.transform(data[i]) for i in range(3)])
    prediction = exp.model(batch)
    exp.task.compute_loss(prediction, batch).loss.backward()
    module = exp.model.global_fuse if fusion == "readout" else exp.model.global_embed
    assert module[0].weight.grad.abs().sum() > 0
    changed = replace(batch, global_features=batch.global_features + 2)
    assert not torch.allclose(exp.model(changed).logits, prediction.logits)
    dropped = drop_tokens(batch, 0.9)
    torch.testing.assert_close(dropped.global_features, batch.global_features)
    torch.testing.assert_close(batch.to("cpu").global_features, batch.global_features)
    with pytest.raises(ValueError, match="Global feature shape"):
        exp.model(replace(batch, global_features=None))
    with pytest.raises(ValueError, match="Global feature shape"):
        exp.model(replace(batch, global_features=torch.ones(3, 4)))


@pytest.mark.parametrize("normalize", [True, False])
def test_globals_training_reload_and_fresh_prediction(tmp_path, normalize):
    cfg = config(make_data(tmp_path), "twistformer", normalize)
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "run").fit(exp)
    predictor = load_predictor(best, "cpu")
    state = predictor.preprocessor.state
    torch.testing.assert_close(
        state["global_mean"], torch.tensor([2.5, 7.0]) if normalize else torch.zeros(2)
    )
    expected_std = (
        torch.tensor([np.sqrt(35 / 12), 1.0], dtype=torch.float32)
        if normalize
        else torch.ones(2)
    )
    torch.testing.assert_close(state["global_std"], expected_std)
    test = predictor.experiment.driver.dataset("test")
    predictions = predictor.predict(test, batch_size=2)
    result = evaluate(predictor, test, tmp_path / "evaluation", plots=False)
    assert np.isfinite(result["preprocessed_loss"])
    output = tmp_path / "fresh.npz"
    subprocess.run(
        [
            sys.executable,
            "scripts/predict.py",
            "--checkpoint",
            str(best),
            "--input",
            cfg.driver["files"]["test"],
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output) as saved:
        np.testing.assert_allclose(
            saved["logits"], predictions.logits.numpy(), atol=1e-6, rtol=1e-5
        )


def test_global_stats_are_per_event_and_invalid_contracts(tmp_path):
    samples = [
        Sample(torch.zeros(n, 1), global_features=torch.tensor([v]))
        for n, v in [(1, 1.0), (9, 5.0)]
    ]
    pre = Preprocessor().fit(samples)
    torch.testing.assert_close(pre.state["global_mean"], torch.tensor([3.0]))
    torch.testing.assert_close(pre.state["global_std"], torch.tensor([2.0]))
    for bad in [None, torch.ones(2), torch.ones(1, 1), torch.tensor([float("nan")])]:
        with pytest.raises(ValueError):
            collate_samples([samples[0], replace(samples[1], global_features=bad)])
        with pytest.raises(ValueError):
            Preprocessor(global_dim=1).fit(
                [samples[0], replace(samples[1], global_features=bad)]
            )
    with pytest.raises(ValueError, match="uniquely"):
        InputSpec(("x",), global_features=("g", "g"))
    with pytest.raises(ValueError, match="units"):
        InputSpec(("x",), global_features=("g",), global_units=("a", "b"))
    exp = build_experiment(config(make_data(tmp_path, globals=False)))
    with pytest.raises(ValueError, match="NPZ requires global_features"):
        exp.driver.dataset("train")
