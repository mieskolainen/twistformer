# Prediction diagnostics and evaluation
#
# mikael.mieskolainen@cern.ch (2026)

"""Standalone PNG/PDF plots. import plotting dependencies only during evaluation"""

from pathlib import Path
from twistboard.plotting import plot_style, save_figure
from twistformer.metrics import confusion_matrix_statistics
import numpy as np


def roc_class_label(outputs, index):
    """
    Name the signal and background in a ROC curve

    Args:
        outputs:  Ordered class names
        index:  Signal class index
    Returns:
        label:  Signal versus background class or remaining classes
    """
    background = outputs[1 - index] if len(outputs) == 2 else "rest"
    return f"{outputs[index]} vs {background}"


def roc_legend_label(label, area):
    """
    Label a ROC curve with its weighted AUC

    Args:
        label:  Class, category or model label
        area:  AUC from the curve's event selection, or None when undefined
    Returns:
        text: Legend label with four decimal places or an unavailable marker
    """
    value = f"{area:.4f}" if area is not None and np.isfinite(area) else "unavailable"
    return f"{label} (AUC = {value})"


@plot_style
def plot_results(data, spec, metrics, curves, output, *, selection_label=None):
    """
    Plot classification or Gaussian regression diagnostics

    Args:
        data:  Prediction and target arrays in physical units
        spec:  TaskSpec defining output names
        metrics:  Task metrics dictionary
        curves:  ROC and precision-recall arrays
        output:  Plot output directory
        selection_label:  Optional physical category label for exported figures
    Returns:
        None: PNG and PDF figures are written to disk
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, *, close=True):
        if selection_label:
            plt.gcf().suptitle(selection_label, fontsize=10)
        plt.tight_layout()
        for extension in ("png", "pdf"):
            save_figure(plt.gcf(), output / f"{name}.{extension}")
        if close:
            plt.close()

    if spec.name == "classification":

        def roc_plot(series, name):
            """
            Plot signal efficiency against logarithmic background efficiency

            Args:
                series:  Background efficiencies, signal efficiencies and labels
                name:  Output figure stem
            Returns:
                None: Exact zero background is clipped only for display
            """
            plt.figure()
            signed = bool((np.asarray(data["weights"]) < 0).any())
            positive = [x[x > 0].min() for x, *_ in series if (x > 0).any()]
            xmin = min(1e-3, min(positive) / 2) if positive else 1e-3
            for x, y, label, area, curve, prefix in series:
                (line,) = plt.plot(x if signed else np.maximum(x, xmin), y, label=roc_legend_label(label, area))
                if prefix + "band_fpr" in curve:
                    bx = curve[prefix + "band_fpr"]
                    lower, upper = (curve[prefix + "tpr_lower"], curve[prefix + "tpr_upper"])
                    if np.isfinite(lower).any():
                        plt.fill_between(
                            bx if signed else np.maximum(bx, xmin), lower, upper, color=line.get_color(), alpha=0.20, linewidth=0
                        )

            random_efficiency = np.linspace(0, 1, 256) if signed else np.geomspace(xmin, 1, 256)
            plt.plot(random_efficiency, random_efficiency, color="gray", linestyle="--", label=roc_legend_label("random", 0.5))
            if not signed:
                plt.xscale("log")
                plt.xlim(xmin, 1)
            plt.ylim(0, 1)
            plt.xlabel(r"Background efficiency $\epsilon_B$")
            plt.ylabel(r"Signal efficiency $\epsilon_S$")
            plt.grid(True, which="both", alpha=0.2)
            plt.legend()
            save(name)
            if not signed:
                plt.figure()
                for x, y, label, area, curve, prefix in series:
                    rejection = np.divide(1, x, out=np.full_like(x, np.nan), where=x > 0)
                    (line,) = plt.plot(y, rejection, label=roc_legend_label(label, area))
                    if prefix + "band_fpr" in curve:
                        bx = curve[prefix + "band_fpr"]
                        selected = bx >= x[x > 0].min() if (x > 0).any() else np.zeros_like(bx, dtype=bool)
                        plt.fill_betweenx(
                            1 / bx[selected],
                            curve[prefix + "tpr_lower"][selected],
                            curve[prefix + "tpr_upper"][selected],
                            color=line.get_color(),
                            alpha=0.20,
                            linewidth=0,
                        )
                plt.yscale("log")
                plt.xlim(0, 1)
                limits = plt.ylim(bottom=1)
                random_efficiency = np.geomspace(1 / limits[1], 1, 256)
                plt.plot(random_efficiency, 1 / random_efficiency, color="gray", linestyle="--", label=roc_legend_label("random", 0.5))
                plt.ylim(limits)
                plt.xlabel(r"Signal efficiency $\epsilon_S$")
                plt.ylabel(r"Background rejection $1/\epsilon_B$")
                plt.grid(True, which="both", alpha=0.2)
                plt.legend()
                save(name.replace("roc", "rejection", 1))

        indices = [spec.signal_class] if len(spec.outputs) == 2 and spec.signal_class is not None else list(curves)
        roc_plot(
            [
                (
                    curves[i]["fpr"],
                    curves[i]["tpr"],
                    roc_class_label(spec.outputs, i),
                    metrics["classes"][spec.outputs[i]]["auc"],
                    curves[i],
                    "",
                )
                for i in indices
                if i in curves and "fpr" in curves[i]
            ],
            "roc",
        )
        if len(spec.outputs) > 2:
            for i, curve in curves.items():
                roc_plot(
                    [
                        (
                            curve[f"background_{j}_fpr"],
                            curve[f"background_{j}_tpr"],
                            f"{spec.outputs[i]} vs {background}",
                            metrics["classes"][spec.outputs[i]]["background_auc"][background],
                            curve,
                            f"background_{j}_",
                        )
                        for j, background in enumerate(spec.outputs)
                        if f"background_{j}_fpr" in curve
                    ],
                    f"roc_class_{i}",
                )
        plt.figure()
        signed = bool((np.asarray(data["weights"]) < 0).any())
        for i, curve in curves.items():
            if "recall" not in curve:
                continue
            (line,) = plt.plot(curve["recall"], curve["precision"], label=spec.outputs[i])
            if "pr_band_recall" in curve:
                plt.fill_between(
                    curve["pr_band_recall"],
                    curve["pr_precision_lower"],
                    curve["pr_precision_upper"],
                    color=line.get_color(),
                    alpha=0.20,
                    linewidth=0,
                )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if not signed:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        plt.grid(True, which="both", alpha=0.2)
        if curves:
            plt.legend()
        save("precision_recall")
        statistics = confusion_matrix_statistics(
            data["targets"],
            data["probabilities"].argmax(axis=1),
            labels=list(range(len(spec.outputs))),
            sample_weight=data["weights"],
        )
        cm = statistics["counts"]
        classes = len(spec.outputs)
        plt.figure(figsize=(max(8, classes * 2.8 + 2), max(6.5, classes * 2.6)))
        image = plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for (row, col), value in np.ndenumerate(cm):
            color = "white" if image.norm(value) > 0.55 else "black"
            plt.text(
                col,
                row,
                format_yield_uncertainty(value, np.sqrt(statistics["sumw2"][row, col])),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
        plt.xticks(range(len(spec.outputs)), spec.outputs, rotation=30)
        plt.yticks(range(len(spec.outputs)), spec.outputs)
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        save("confusion")
        plot_confusion_efficiency_purity(statistics, spec.outputs)
        save("confusion_normalized")
        for index, name in enumerate(spec.outputs):
            plt.figure()
            for label, label_name in enumerate(spec.outputs):
                selected = data["targets"] == label
                plt.hist(
                    data["probabilities"][selected, index],
                    weights=data["weights"][selected],
                    bins=np.linspace(0, 1, 31),
                    histtype="step",
                    label=label_name,
                )
            plt.xlabel(r"$P$")
            plt.xlim(0, 1)
            plt.title(f"Classifier output: {name}")
            plt.ylabel("Weighted events")
            plt.legend()
            save(f"score_{index}", close=False)
            if (np.asarray(data["weights"]) < 0).any():
                plt.yscale("symlog")
            else:
                plt.yscale("log", nonpositive="mask")
            save(f"score_{index}_log")

        def calibration_plot(series, name, xlabel, ylabel):
            """
            Draw fixed-bin calibration curves and bootstrap accuracy bands

            Args:
                series:  Named calibration statistics
                name:  Output figure stem
                xlabel:  Probability coordinate label
                ylabel:  Observed fraction label
            Returns:
                None: Calibration figure is exported
            """
            plt.figure()
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal calibration")
            for label, curve in series.items():
                x, y = (np.asarray(curve[k], dtype=float) for k in ("prob_pred", "prob_true"))
                (line,) = plt.plot(x, y, "o-", label=label)
                if curve["n_resamples"]:
                    lower, upper = (np.asarray(curve[k], dtype=float) for k in ("prob_true_lower", "prob_true_upper"))
                    plt.fill_between(x, lower, upper, color=line.get_color(), alpha=0.20, linewidth=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(alpha=0.2)
            save(name)

        if "calibration_classes" in metrics:
            calibration_plot(metrics["calibration_classes"], "calibration", r"$P$", "Observed class fraction")
            calibration_plot(
                {"Top-label accuracy": metrics["calibration_top_label"]},
                "calibration_top_label",
                "Maximum class probability",
                "Accuracy",
            )
        else:
            plt.figure()
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot([r["confidence"] for r in metrics["calibration"]], [r["accuracy"] for r in metrics["calibration"]], "o-")
            plt.xlabel("Confidence")
            plt.ylabel("Accuracy")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            save("calibration")
    else:
        for i, name in enumerate(spec.outputs):
            valid = np.isfinite(data["targets"][:, i])
            if "target_mask" in data:
                valid &= data["target_mask"][:, i]
            y, mu = (data[k][valid, i] for k in ("targets", "mean"))
            std = data["std"][valid, i] if "std" in data else None
            w = data["weights"][valid]
            plt.figure()
            plt.scatter(y, mu, s=3, alpha=0.3)
            plt.xlabel(f"True {name}")
            plt.ylabel(f"Predicted {name}")
            save(f"prediction_{i}")
            diagnostics = [("residual", y - mu)]
            if std is not None:
                diagnostics.append(("pull", (y - mu) / std))
            for title, values in diagnostics:
                plt.figure()
                plt.hist(values, weights=w, bins=40, histtype="step")
                plt.xlabel(f"{name}: {title}")
                plt.ylabel("Weighted events")
                save(f"{title}_{i}")
            bins = metrics[name].get("bins", [])
            for key in ("rmse", "coverage_1sigma") if std is not None else ("rmse",):
                plt.figure()
                plt.plot([r["target"] for r in bins], [r[key] for r in bins], "o-")
                plt.xlabel(f"True {name}")
                plt.ylabel(key)
                save(f"{key}_{i}")

def format_yield_uncertainty(value, error):
    """
    Format weighted yield and uncertainty with a shared LaTeX power of ten

    Args:
        value:  Signed event yield
        error:  Statistical yield uncertainty
    Returns:
        text: Mathtext annotation with two significant uncertainty digits
    """
    magnitude = max(abs(value), abs(error))
    exponent = int(np.floor(np.log10(magnitude))) if magnitude else 0
    scientific = exponent >= 3 or exponent <= -3
    scale = 10.0**exponent if scientific else 1.0
    v, e = value / scale, error / scale
    decimals = max(0, 1 - int(np.floor(np.log10(abs(e))))) if e else 2
    body = rf"{v:.{decimals}f} \pm {e:.{decimals}f}"
    return rf"$({body})\times 10^{{{exponent}}}$" if scientific else rf"${body}$"

def plot_confusion_efficiency_purity(statistics, labels):
    """
    Draw Pandora-style split cells for efficiency and purity with uncertainties

    Args:
        statistics:  Confusion counts and row/column rate intervals
        labels:  Class names in matrix order
    Returns:
        figure: Matplotlib figure with blue upper and green lower half cells
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    count = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, count * 1.8 + 4), max(6.5, count * 1.5 + 2)))
    finite = np.concatenate([statistics[key][np.isfinite(statistics[key])] for key in ("row_rate", "column_rate")])
    norm = plt.Normalize(min(0, finite.min()) if finite.size else 0, max(1, finite.max()) if finite.size else 1)
    for prefix, cmap_name, offset, title in (
        ("row", "Blues", -0.5, "Efficiency: row normalization"),
        ("column", "Greens", 0.0, "Purity: column normalization"),
    ):
        cmap = plt.get_cmap(cmap_name)
        for (row, col), value in np.ndenumerate(statistics[prefix + "_rate"]):
            available = np.isfinite(value)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row + offset),
                    1,
                    0.5,
                    facecolor=cmap(norm(value)) if available else "0.96",
                    edgecolor="0.85",
                    linewidth=0.6,
                )
            )
            if available:
                lower = value - statistics[prefix + "_lower"][row, col]
                upper = statistics[prefix + "_upper"][row, col] - value
                text = rf"${value:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$"
            else:
                text = "n/a"
            ax.text(
                col,
                row + offset + 0.25,
                text,
                ha="center",
                va="center",
                fontsize=max(7, 12 - count / 2),
                color="white" if available and norm(value) > 0.55 else "black",
            )
        scale = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(scale, ax=ax, fraction=0.045, pad=0.07).set_label(title)
    for i in range(count):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="0.15", linewidth=1.6))
    ax.set_xticks(
        range(count),
        [f"{name}\nΣw = {statistics['column_denominator'][i]:.4g}" for i, name in enumerate(labels)],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(count), [f"{name}\nΣw = {statistics['row_denominator'][i]:.4g}" for i, name in enumerate(labels)])
    ax.set(xlim=(-0.5, count - 0.5), ylim=(count - 0.5, -0.5), xlabel="Predicted class", ylabel="True class")
    return fig

@plot_style
def plot_category_rocs(series, spec, output, *, selection_label=None, signed=False):
    """
    Compare bin ROC curves with bootstrap bands using the same class definition

    Args:
        series:  Tuples of bin label and full class-indexed curve arrays in threshold order
        spec:  Classification task schema
        output:  Category overlay figure directory
        selection_label:  Optional physical category title
        signed:  Preserve signed efficiencies on linear axes
    Returns:
        None: One PNG and PDF ROC overlay per signal class
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    indices = [spec.signal_class] if len(spec.outputs) == 2 and spec.signal_class is not None else range(len(spec.outputs))
    for index in indices:
        selected = []
        for label, curves in series:
            if index not in curves or "fpr" not in curves[index]:
                continue
            curve = curves[index]
            # Integrate before display clipping, preserving threshold order for signed weights
            area = float(np.trapezoid(curve["tpr"], curve["fpr"]))
            selected.append((roc_legend_label(label, area), curve))
        if not selected:
            continue
        fig, axis = plt.subplots()
        positive = [curve["fpr"][curve["fpr"] > 0].min() for _, curve in selected if (curve["fpr"] > 0).any()]
        xmin = min(1e-3, min(positive) / 2) if positive else 1e-3
        for n, (label, curve) in enumerate(selected):
            (line,) = axis.plot(
                curve["fpr"] if signed else np.maximum(curve["fpr"], xmin),
                curve["tpr"],
                label=label,
                linestyle=["-", "--", "-.", ":"][(n // 7) % 4],
            )
            if "band_fpr" in curve:
                lower, upper = curve["tpr_lower"], curve["tpr_upper"]
                if np.isfinite(lower).any():
                    axis.fill_between(
                        curve["band_fpr"] if signed else np.maximum(curve["band_fpr"], xmin),
                        lower, upper, color=line.get_color(), alpha=0.20, linewidth=0,
                    )
        random_efficiency = np.linspace(0, 1, 256) if signed else np.geomspace(xmin, 1, 256)
        axis.plot(random_efficiency, random_efficiency, color="gray", linestyle="--", label=roc_legend_label("random", 0.5))
        if not signed:
            axis.set_xscale("log")
            axis.set_xlim(xmin, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel(r"Background efficiency $\epsilon_B$")
        axis.set_ylabel(r"Signal efficiency $\epsilon_S$")
        if selection_label:
            fig.suptitle(selection_label, fontsize=10)
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize="small")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            save_figure(fig, output / f"roc_class_{index}.{extension}")
        plt.close(fig)
