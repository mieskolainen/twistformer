# Training histories and checkpoint inventory
#
# mikael.mieskolainen@cern.ch (2026)

import csv
import json
import math
import os
from pathlib import Path
from .plotting import PLOT_COLORS, save_figure
from .events import active_events

def write_json(path, data):
    """
    Replace a JSON record atomically

    Args:
        path:  Destination file
        data:  JSON-compatible record
    Returns:
        None: The record is persisted
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temporary, path)

def append_epoch_metrics(path, row):
    """
    Append an epoch and extend older CSV headers when new metrics appear

    Args:
        path:  Training metrics file
        row:  Epoch counters and scalar metrics
    """
    fields = list(row)
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            previous = reader.fieldnames or []
            fields = previous + [key for key in row if key not in previous]
            if fields != previous:
                temporary = path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="") as output:
                    writer = csv.DictWriter(output, fields, restval="nan")
                    writer.writeheader()
                    writer.writerows(reader)
                    writer.writerow(row)
                os.replace(temporary, path)
                return
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fields, restval="nan")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

def read_history(root):
    """
    Read complete finite epoch records, including during training

    Args:
        root:  Experiment directory
    Returns:
        rows: Ordered numeric epoch records
    """
    path = Path(root) / "metrics.csv"
    if not path.exists():
        events = Path(root) / "events.jsonl"
        if not events.is_file() or events.is_symlink():
            return []
        epochs = {}
        for event in active_events(events):
            epoch = event.get("epoch")
            if event.get("type") != "scalar" or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
                continue
            row = epochs.setdefault(epoch, {"epoch": epoch - 1})
            row["step"] = event.get("step")
            row.update({key: value for key, value in event.get("values", {}).items() if key in {"train_loss", "val_loss", "val_accuracy", "val_auc", "lr"} and isinstance(value, (int, float)) and math.isfinite(value)})
        return [epochs[key] for key in sorted(epochs)]
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                values = {key: float(value) for key, value in row.items()}
                if "val_auc" in values and not math.isfinite(values["val_auc"]):
                    values.pop("val_auc")
                if all(math.isfinite(value) for value in values.values()):
                    rows.append(values)
            except (TypeError, ValueError):
                continue
    return rows

def checkpoint_index(root, monitor, mode):
    """
    Index checkpoint files against their validation epoch

    Args:
        root:  Experiment directory
        monitor:  Validation metric used for model selection
        mode:  min or max
    Returns:
        None: checkpoints/index.json is updated
    """
    root = Path(root)
    rows = read_history(root)
    if not rows:
        return
    select = min if mode == "min" else max
    best = select(rows, key=lambda row: row[monitor])
    by_epoch = {int(row["epoch"]): row for row in rows}
    records = []
    for path in sorted((root / "checkpoints").glob("*.pt")):
        if path.stem == "best":
            row = best
        elif path.stem == "last":
            row = rows[-1]
        elif path.stem.startswith("epoch_"):
            row = by_epoch.get(int(path.stem.split("_")[1]), {})
        else:
            continue
        records.append(
            {
                "file": path.name,
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "metric": row.get(monitor),
                "bytes": path.stat().st_size,
            }
        )
    write_json(root / "checkpoints/index.json", {"monitor": monitor, "mode": mode, "checkpoints": records})

def training_plots(root):
    """
    Plot epoch losses and learning rate from saved history

    Args:
        root:  Experiment directory
    Returns:
        None: PNG and PDF figures are written under plots/
    """
    root = Path(root)
    rows = read_history(root)
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "plots"
    output.mkdir(exist_ok=True)
    epochs = [row["epoch"] + 1 for row in rows]
    for name, keys, ylabel in [("loss", ("train_loss", "val_loss"), "Loss"), ("learning_rate", ("lr",), "Learning rate")]:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.set_prop_cycle(color=PLOT_COLORS)
        for key in keys:
            axis.plot(epochs, [row[key] for row in rows], label=key, marker="." if len(rows) < 20 else None)
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            temporary = output / f".{name}.{extension}"
            save_figure(fig, temporary)
            os.replace(temporary, output / f"{name}.{extension}")
        plt.close(fig)
