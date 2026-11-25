# Batched event fetching and deterministic worker reuse
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import fields, replace
import os
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from drivers.tau3_classification.driver import Tau3Dataset
from drivers.tau3_regression.driver import Tau3RegressionDataset
from twistformer import Trainer, build_experiment, parse_config
from twistformer.api.config import TrainingConfig
from twistformer.api.contracts import Sample
from twistformer.api.predictor import Predictor
from twistformer.data.batching import collate_samples
from twistformer.data.augmentation import prepare_training_dataset
from twistformer.data.loading import set_loader_epoch
from twistformer.data.preprocessing import PreparedDataset, Preprocessor
from twistformer.training.distributed import DistributedRuntime, PaddedTrainingDataset
from twistformer.training.engine import make_loader
from twistformer.training.checkpoint import read_checkpoint


def test_run_seed_sets_python_numpy_and_torch():
    """
    Reproduce all three global streams when the experiment is rebuilt
    """
    seed = 37
    config = parse_config({"driver": {"name": "synthetic"}, "training": {"seed": seed}})
    first = build_experiment(config)
    assert random.random() == random.Random(seed).random()
    assert np.random.random() == np.random.RandomState(seed).random_sample()
    draw = torch.rand(4)
    random.random()
    np.random.random()
    torch.rand(10)
    second = build_experiment(config)
    assert random.random() == random.Random(seed).random()
    assert np.random.random() == np.random.RandomState(seed).random_sample()
    torch.testing.assert_close(torch.rand(4), draw, rtol=0, atol=0)
    torch.testing.assert_close(next(first.model.parameters()), next(second.model.parameters()), rtol=0, atol=0)


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("proxy", [False, True])
def test_tau_batch_matches_indexed_events(regression, proxy):
    """
    Preserve shuffled duplicates, geometry, truth and source buffers
    """
    features = np.arange(32, dtype=np.float32).reshape(8, 4) + 1
    chunk = SimpleNamespace(
        features=features,
        offsets=np.array([0, 2, 3, 8]),
        entries=np.array([7, 11, 19]),
        weights=np.array([1.0, -0.2, 3.0]),
        targets=np.arange(18, dtype=np.float32).reshape(3, 6),
    )
    indices = np.array([[0, 0], [1, 2], [0, 1]])
    if regression:
        dataset = Tau3RegressionDataset([("a", chunk), ("b", chunk)], indices, proxy)
    else:
        dataset = Tau3Dataset([("a", 0, chunk), ("b", 1, chunk)], indices, proxy)
    original = features.copy()
    selected = [2, 0, 1, 2]
    actual = dataset.__getitems__(selected)
    for sample, index in zip(actual, selected, strict=True):
        expected = dataset[index]
        for field in fields(Sample):
            value, reference = getattr(sample, field.name), getattr(
                expected, field.name
            )
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, reference)
            else:
                assert value == reference
    actual[0].features.zero_()
    assert actual[-1].features.abs().sum() > 0
    np.testing.assert_array_equal(chunk.features, original)


@pytest.mark.parametrize("regression", [False, True])
def test_batch_preprocessing_matches_scalar(regression):
    """
    Keep normalization, masks, globals and class weights unchanged
    """
    samples = [
        Sample(
            torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) + i,
            (
                torch.tensor([float(i), float(i + 1)])
                if regression
                else torch.tensor(i % 2)
            ),
            p4=torch.ones(n, 4, dtype=torch.float64),
            weight=i + 0.25,
            id=str(i),
            global_features=torch.tensor([float(i), 2.0]),
            target_mask=torch.tensor([True, i != 2]) if regression else None,
            token_weights=torch.arange(n, dtype=torch.float32),
        )
        for i, n in enumerate([2, 5, 1, 3])
    ]
    pre = Preprocessor().fit(
        samples, regression=regression, class_count=0 if regression else 2
    )
    prepared = PreparedDataset(samples, pre)
    order = [3, 0, 2, 3]
    actual = collate_samples(prepared.__getitems__(order))
    expected = collate_samples([prepared[index] for index in order])
    for field in fields(actual):
        value, reference = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, equal_nan=True)
        else:
            assert value == reference


def test_dataset_bulk_fetch_and_custom_preprocessor():
    """
    Use native batch fetching and retain driver transform overrides
    """

    class BulkDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            raise AssertionError("Scalar fetch is not expected")

        def __getitems__(self, indices):
            return [Sample(torch.ones(2, 1), id=str(i)) for i in indices]

    class CustomPreprocessor(Preprocessor):
        def transform(self, sample):
            return replace(sample, features=sample.features + 7)

    prepared = PreparedDataset(BulkDataset(), CustomPreprocessor())
    batch = next(
        iter(make_loader(prepared, TrainingConfig(device="cpu"), batch_size=3))
    )
    assert batch.ids == ("0", "1", "2")
    assert (batch.features == 8).all()


class RandomEvents(Dataset):
    """
    Exercise all three worker random generators and native batched fetching
    """

    def __len__(self):
        return 13

    def __getitem__(self, index):
        raise AssertionError("The loader must use batched fetching")

    def __getitems__(self, indices):
        return [
            Sample(
                torch.tensor(
                    [[random.random(), np.random.random(), torch.rand(()).item()]]
                ),
                id=str(index),
                observables={"worker_pid": float(os.getpid())},
            )
            for index in indices
        ]


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_persistent_workers_resume_epoch(distributed, device):
    """
    Reuse processes while reproducing shuffled events and worker RNG on resume
    """
    # Event loading stays on CPU for both training devices
    cfg = TrainingConfig(workers=2, batch_size=4, device=device, pin_memory=False)
    runtime = DistributedRuntime(torch.device(device), 0, 2, distributed)
    loader = make_loader(RandomEvents(), cfg, train=True, runtime=runtime)
    if distributed or device == "cuda":
        assert loader.multiprocessing_context.get_start_method() == "spawn"
    else:
        assert loader.multiprocessing_context is None
    first = list(loader)
    set_loader_epoch(loader, 1)
    second = list(loader)
    fresh = make_loader(RandomEvents(), cfg, train=True, epoch=1, runtime=runtime)
    resumed = list(fresh)
    assert [batch.ids for batch in first] != [batch.ids for batch in second]
    assert [batch.ids for batch in second] == [batch.ids for batch in resumed]
    assert set(torch.cat([b.observables["worker_pid"] for b in first]).tolist()) == set(
        torch.cat([b.observables["worker_pid"] for b in second]).tolist()
    )
    for actual, expected in zip(second, resumed, strict=True):
        torch.testing.assert_close(actual.features, expected.features, rtol=0, atol=0)
        torch.testing.assert_close(actual.sample_weights, expected.sample_weights)


@pytest.mark.parametrize("workers", [0, 1])
def test_cuda_validation_workers(workers):
    """
    Load ordered validation events without inheriting the CUDA process state
    """
    cfg = TrainingConfig(workers=workers, device="cuda", pin_memory=False)
    loader = make_loader(RandomEvents(), cfg)
    if workers:
        assert loader.multiprocessing_context.get_start_method() == "spawn"
    else:
        assert loader.multiprocessing_context is None
        assert not loader.persistent_workers
    batches = list(loader)
    assert [event for batch in batches for event in batch.ids] == [
        str(index) for index in range(13)
    ]
    assert all(batch.features.device.type == "cpu" for batch in batches)


def test_batched_distributed_padding():
    """
    Retain the original event weight when a repeated rank-padding event is zeroed
    """
    sample = Sample(torch.ones(2, 1), weight=2.5)
    dataset = PaddedTrainingDataset([sample])
    first, padding = dataset.__getitems__([(0, False), (0, True)])
    assert first.weight == sample.weight == 2.5
    assert padding.weight == 0


def test_predictor_loader_uses_shared_options():
    """
    Apply training loader options to ordered signed-weight inference
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 8},
            "model": {"width": 8, "heads": 2, "depth": 1},
            "training": {"device": "cpu", "workers": 1, "eval_prefetch_factor": 3},
        }
    )
    experiment = build_experiment(cfg)
    samples = experiment.driver.dataset("train")
    pre = Preprocessor().fit(samples)
    predictor = Predictor(experiment, pre, "cpu")
    loader = predictor.loader(
        [replace(samples[0], weight=-1.0), samples[1]], batch_size=2
    )
    assert loader.num_workers == 1 and loader.persistent_workers
    assert loader.prefetch_factor == 3 and not loader.pin_memory
    batch = next(iter(loader))
    assert batch.ids == (samples[0].id, samples[1].id)
    assert batch.sample_weights[0] == -1


def random_augmentation(sample):
    """
    Apply worker-local random feature noise for resume validation
    """
    noise = random.random() + np.random.random() + torch.rand_like(sample.features)
    return replace(sample, features=sample.features + 0.01 * noise)


@pytest.mark.parametrize("augmentation", ["online", "dataset", "none"])
def test_training_resume_with_persistent_workers(tmp_path, monkeypatch, augmentation):
    """
    Preserve model updates when resuming seeded worker augmentation
    """
    monkeypatch.setattr("twistboard.experiments.training_plots", lambda root: None)
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 12,
                "val_size": 4,
                "max_tokens": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.1},
            "training": {
                "epochs": 2,
                "workers": 2,
                "batch_size": 4,
                "device": "cpu",
                "save_every": 1,
                "augmentation": augmentation,
            },
        }
    )
    full = build_experiment(cfg)
    full.driver.augment = random_augmentation
    Trainer(tmp_path / "full").fit(full)
    resumed = build_experiment(cfg)
    resumed.driver.augment = random_augmentation
    Trainer(
        tmp_path / "resumed", resume=tmp_path / "full/checkpoints/epoch_00000.pt"
    ).fit(resumed)
    reference = read_checkpoint(tmp_path / "full/checkpoints/last.pt")
    actual = read_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    for name, value in reference["model"].items():
        torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["dataset", "online", "none"])
def test_augmentation_stage_and_rng(mode):
    """
    Select fixed or fresh driver transforms without changing labels or weights

    Args:
        mode:  Training augmentation stage
    Returns:
        None: Cache draws, validation exclusion and RNG state are checked
    """
    samples = [Sample(torch.ones(2, 1), target=torch.tensor(i), weight=i + 1, id=str(i)) for i in range(2)]
    prep = Preprocessor(normalize_features=False).fit(samples)
    calls = []

    def augment(sample):
        """
        Draw one feature shift from each supported CPU random generator

        Args:
            sample:  Raw training event
        Returns:
            sample: Shifted event with unchanged identity and weight
        """
        calls.append(sample.id)
        return random_augmentation(sample)

    config = TrainingConfig(augmentation=mode, seed=17)
    torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    prepared = prepare_training_dataset(samples, prep, augment, config)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert len(calls) == (2 if mode == "dataset" else 0)
    first, second = prepared[0], prepared.__getitems__([0])[0]
    assert first.id == second.id == samples[0].id
    assert first.weight == second.weight == samples[0].weight
    torch.testing.assert_close(first.target, second.target)
    if mode == "online":
        assert len(calls) == 2 and not torch.equal(first.features, second.features)
    else:
        torch.testing.assert_close(first.features, second.features)
    if mode == "dataset":
        random.random()
        np.random.random()
        torch.rand(10)
        restored = prepare_training_dataset(
            samples, prep, augment, config,
            augment_dataset=lambda dataset: [augment(sample) for sample in dataset],
        )
        torch.testing.assert_close(first.features, restored[0].features)
    elif mode == "none":
        torch.testing.assert_close(first.features, samples[0].features)
        assert not calls
    validation = PreparedDataset(samples, prep)
    torch.testing.assert_close(validation[0].features, samples[0].features)


def test_invalid_augmentation_stage():
    """
    Reject an unknown augmentation stage in the steering card
    """
    with pytest.raises(ValueError, match="training.augmentation"):
        parse_config({"driver": {"name": "synthetic"}, "training": {"augmentation": "invalid"}})
