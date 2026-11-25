# Weighted class histograms of XGBoost training inputs
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
from tqdm.auto import tqdm

from twistboard.outputs import write_json

FEATURE_BLOCK = 64


def write_feature_distributions(output, batches, classes, bins, categories, observables, class_weights):
    """
    Save training distributions in compact blocks for on-demand plotting

    Args:
        output:  Training run directory
        batches:  Prepared physical training rows before augmentation
        classes:  Class names in label order
        bins:  Number of equal-width bins per feature
        categories:  Optional half-open bins of driver observables
        observables:  Driver definitions with physical units
        class_weights:  Explicit class multipliers used by the XGBoost loss
    """
    width = len(batches.names)
    low = np.full(width, np.inf)
    high = np.full(width, -np.inf)
    with tqdm(total=batches.count, desc="Training feature ranges", unit="event", dynamic_ncols=True) as progress:
        for features, _, _, _ in batches.arrays():
            np.minimum(low, features.min(axis=0), out=low)
            np.maximum(high, features.max(axis=0), out=high)
            progress.update(len(features))
    constant = low == high
    padding = np.maximum(0.5, np.abs(low[constant]) * 1e-6)
    low[constant] -= padding
    high[constant] += padding
    edges = np.linspace(low, high, bins + 1, axis=-1)
    edges[:, -1] = high

    labels_by_category = ["All training events"]
    groups = []
    for category in categories:
        unit = observables[category["observable"]].get("unit", "")
        name = category["name"].replace("_", " ").capitalize()
        bounds = category["edges"]
        labels = [f"{first:g}–{last:g}" for first, last in zip(bounds, bounds[1:])]
        if category.get("underflow"):
            labels.insert(0, f"< {bounds[0]:g}")
        if category.get("overflow"):
            labels.append(f"≥ {bounds[-1]:g}")
        start = len(labels_by_category)
        labels_by_category.extend(f"{name} {label} {unit}".rstrip() for label in labels)
        groups.append((start, len(labels_by_category), category))

    shape = (len(labels_by_category), len(classes), bins)
    counts = np.zeros(shape[:2], dtype=np.int64)
    histograms = np.zeros((width, *shape), dtype=np.float64)
    feature_block = np.arange(FEATURE_BLOCK, dtype=np.intp)
    with tqdm(total=batches.count, desc="Training feature histograms", unit="event", dynamic_ncols=True) as progress:
        for features, labels, weights, observable_values in batches.arrays():
            if class_weights:
                weights = (weights * np.take(class_weights, labels)).astype(np.float32)

            if groups:
                selections = []
                for start, stop, category in groups:
                    column = observable_values[category["observable"]]
                    interval = np.searchsorted(category["edges"], column, side="right") + category.get("underflow", False)
                    selected = np.isfinite(column) & (interval > 0) & (interval <= stop - start)
                    if start == 1:
                        # The first group uses bin zero for events outside its intervals
                        interval[~selected] = 0
                        selections.append((slice(0, stop), slice(None), interval * len(classes) + labels))
                    else:
                        selections.append((slice(start, stop), selected, (interval[selected] - 1) * len(classes) + labels[selected]))
            else:
                selections = [(slice(0, 1), slice(None), labels)]

            blocks = []
            for categories_slice, rows, event_bins in selections:
                count = counts[categories_slice]
                count += np.bincount(event_bins, minlength=count.size).reshape(count.shape)
                blocks.append((categories_slice, rows, (event_bins * bins)[:, None], weights[rows, None],
                               feature_block * (count.size * bins)))

            for start in range(0, width, FEATURE_BLOCK):
                stop = min(start + FEATURE_BLOCK, width)
                columns = slice(start, stop)
                values = features[:, columns]
                scaled = (values - low[columns]) * (bins / (high[columns] - low[columns]))
                indexes = scaled.clip(0, bins - 1).astype(np.intp)
                offsets = feature_block[:stop - start]
                block_edges = edges[columns]
                indexes -= (indexes > 0) & (values < block_edges[offsets, indexes])
                indexes += (indexes < bins - 1) & (values >= block_edges[offsets, indexes + 1])
                for categories_slice, rows, row_bins, selected_weights, feature_offsets in blocks:
                    indexes_in_group = row_bins + feature_offsets[:stop - start]
                    indexes_in_group += indexes[rows]
                    block = histograms[columns, categories_slice]
                    block += np.bincount(indexes_in_group.ravel(),
                                         weights=np.broadcast_to(selected_weights, indexes_in_group.shape).ravel(),
                                         minlength=block.size).reshape(block.shape)
            progress.update(len(features))
    if groups:
        # Only the first group partitions the sample used for the inclusive sum
        first_stop = groups[0][1]
        counts[0] = counts[:first_stop].sum(axis=0)
        histograms[:, 0] = histograms[:, :first_stop].sum(axis=1)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    file = plots / "training_distributions.npz"
    temporary = plots / ".training_distributions.npz.tmp"
    arrays = {}
    for block, start in enumerate(range(0, width, FEATURE_BLOCK)):
        columns = slice(start, start + FEATURE_BLOCK)
        arrays[f"edges_{block}"] = edges[columns]
        arrays[f"weights_{block}"] = histograms[columns]
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(file)
    write_json(plots / "training_distributions.json", {
        "schema_version": 1,
        "stage": "preprocessed_before_augmentation",
        "features": batches.names,
        "classes": list(classes),
        "categories": labels_by_category,
        "category_groups": categories,
        "events": counts.tolist(),
        "feature_block": FEATURE_BLOCK,
    })
