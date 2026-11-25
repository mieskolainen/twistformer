# Common-sample model diagnostic overlays
#
# mikael.mieskolainen@cern.ch (2026)

from pathlib import Path
import numpy as np
from twistboard.plotting import plot_style, save_figure
from .plots import roc_class_label, roc_legend_label

@plot_style
def plot_comparison(selection, spec, output):
    """
    Export ROC, rejection, precision-recall and calibration overlays

    Args:
        selection: Comparison metrics and curves for one event selection
        spec: Common class schema
        output: Plot directory
    Returns:
        None: PNG and PDF figures are saved
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    indices = [spec.signal_class] if len(spec.outputs) == 2 and spec.signal_class is not None else range(len(spec.outputs))
    classification = spec.name == "classification"
    kinds = ("roc", "rejection", "precision_recall", "calibration") if classification else ("residual", "rmse", "coverage_1sigma")
    for index in indices:
        unit = f" [{spec.units[index]}]" if spec.units and spec.units[index] else ""
        for kind in kinds:
            if kind == "calibration" and index != indices[0]:
                continue
            fig, axis = plt.subplots()
            for n, model in enumerate(selection["models"]):
                curve = model["curves"].get(str(index))
                if not classification:
                    profile = (curve or {}).get(kind, {"x": [], "y": []})
                    x, y = np.asarray(profile["x"], dtype=float), np.asarray(profile["y"], dtype=float)
                elif kind == "calibration":
                    bins = model["metrics"].get("calibration", [])
                    x = np.asarray([b["confidence"] for b in bins], dtype=float)
                    y = np.asarray([b["accuracy"] for b in bins], dtype=float)
                elif not curve:
                    continue
                elif kind == "roc":
                    x, y = np.asarray(curve["fpr"]), np.asarray(curve["tpr"])
                elif kind == "rejection":
                    x, background = np.asarray(curve["tpr"]), np.asarray(curve["fpr"])
                    y = np.divide(1.0, background, out=np.full_like(background, np.nan), where=(background > 0) & (background <= 1))
                    y[(x < 0) | (x > 1)] = np.nan
                else:
                    x, y = np.asarray(curve["recall"]), np.asarray(curve["precision"])
                label = model["label"]
                if kind in ("roc", "rejection"):
                    label = roc_legend_label(label, model["metrics"]["classes"][spec.outputs[index]]["auc"])
                axis.plot(x, y, label=label, linestyle=["-", "--", "-.", ":"][n % 4])
            if kind in ("roc", "calibration"):
                axis.plot(
                    [0, 1], [0, 1], color="gray", linestyle="--",
                    label=roc_legend_label("random", 0.5) if kind == "roc" else "ideal",
                )
            if kind == "rejection":
                axis.set_yscale("log")
            if kind == "roc":
                axis.set_ylim(0, 1)
            labels = {
                "roc": (r"Background efficiency $\epsilon_B$", r"Signal efficiency $\epsilon_S$"),
                "rejection": (r"Signal efficiency $\epsilon_S$", r"Background rejection $1/\epsilon_B$"),
                "precision_recall": ("Recall", "Precision"),
                "calibration": ("Confidence", "Accuracy"),
                "residual": (f"Prediction − target{unit}", "Weighted density"),
                "rmse": (f"Target{unit}", f"RMSE{unit}"),
                "coverage_1sigma": (f"Target{unit}", "One-sigma interval coverage"),
            }
            axis.set_xlabel(labels[kind][0])
            axis.set_ylabel(labels[kind][1])
            if kind == "roc" and selection.get("id", "inclusive") != "inclusive":
                axis.set_title(selection["label"])
            if classification and kind != "roc":
                axis.set_title(
                    f"{selection['label']} - {roc_class_label(spec.outputs, index)}"
                    if kind != "calibration"
                    else f"{selection['label']} - Top-label calibration"
                )
            if not classification:
                axis.set_title(f"{selection['label']} - {spec.outputs[index]}")
            axis.grid(True, alpha=0.2)
            axis.legend(fontsize="small")
            fig.tight_layout()
            for extension in ("png", "pdf"):
                name = "calibration" if kind == "calibration" else f"{kind}_{'class' if classification else 'target'}_{index}"
                save_figure(fig, output / f"{name}.{extension}")
            plt.close(fig)
