# Weighted training histograms and their Twistboard view
#
# mikael.mieskolainen@cern.ch (2026)

import json
from types import SimpleNamespace

import numpy as np
import pytest

from twistboard.server import training_distribution
from twistformer.training.feature_distributions import FEATURE_BLOCK, write_feature_distributions


def test_training_feature_histograms(tmp_path):
    """
    Compare blocked histograms with weighted classes in overlapping category groups

    Args:
        tmp_path:  Temporary dashboard output directory
    """
    output = tmp_path / "run"
    output.mkdir()
    (output / "experiment.json").write_text('{}')
    values = np.arange(8, dtype=np.float32)
    features = np.tile(values[:, None], (1, FEATURE_BLOCK + 1))
    features[:, -1] = 3
    labels = np.arange(8) % 2
    weights = np.asarray([1, 2, 3, 1, 0, 2, 1, 3], dtype=np.float32)
    energies = np.asarray([0., 7.49, 7.5, 14.99, 15., -1., 0., np.nan])
    cells = np.asarray([1., 20., 32., 64., np.inf, 20., 32., 21.])
    chunks = tuple((features[part], labels[part], weights[part], {"truth_energy": energies[part], "n_cells": cells[part]})
                   for part in (slice(0, 3), slice(3, None)))
    batches = SimpleNamespace(names=[f"graph_{index}" for index in range(features.shape[1])], count=len(values),
                              arrays=lambda: iter(chunks))
    factors = [0.5, 2.0]
    categories = [{"name": "truth_energy", "observable": "truth_energy", "edges": [0., 7.5, 15.]},
                  {"name": "n_cells", "observable": "n_cells", "edges": [20, 32], "underflow": True, "overflow": True}]
    observables = {"truth_energy": {"unit": "GeV"}, "n_cells": {}}
    write_feature_distributions(output, batches, ('pi0', 'gamma'), 7, categories, observables, factors)
    record = json.loads((output / "plots/training_distributions.json").read_text())
    assert record["features"] == batches.names
    assert record["classes"] == ["pi0", "gamma"]
    assert record["category_groups"] == categories
    assert record["categories"] == ["All training events", "Truth energy 0–7.5 GeV", "Truth energy 7.5–15 GeV",
                                    "N cells < 20", "N cells 20–32", "N cells ≥ 32"]
    selections = [np.ones(len(values), dtype=bool), (energies >= 0) & (energies < 7.5),
                  (energies >= 7.5) & (energies < 15), cells < 20, (cells >= 20) & (cells < 32),
                  (cells >= 32) & np.isfinite(cells)]
    assert record["events"] == [[int((keep & (labels == label)).sum()) for label in (0, 1)] for keep in selections]
    for index in (0, FEATURE_BLOCK):
        result = training_distribution(tmp_path, output.name, index)
        for category, keep in enumerate(selections):
            for label in (0, 1):
                selected = keep & (labels == label)
                expected, _ = np.histogram(features[selected, index], bins=result["edges"],
                                           weights=weights[selected] * factors[label])
                np.testing.assert_array_equal(result["weights"][category][label], expected)
    with pytest.raises(ValueError, match="Unknown training variable"):
        training_distribution(tmp_path, output.name, len(batches.names))
