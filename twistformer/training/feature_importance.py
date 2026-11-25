# Ranked split gains for boosted tree inputs
#
# mikael.mieskolainen@cern.ch (2026)

from twistboard.plotting import PLOT_COLORS

FEATURES_PER_PAGE = 32


def feature_importance(booster, features, checkpoint):
    """
    Rank driver inputs by their total gain over all tree splits

    Args:
        booster:  Native XGBoost ensemble
        features:  Input names in driver column order
        checkpoint:  Checkpoint filename
    Returns:
        result:  JSON record with stable ranks and zero scores for unused inputs
    """
    scores = booster.get_score(importance_type="total_gain")
    names = booster.feature_names or [f"f{index}" for index in range(booster.num_features())]
    ranked = [(feature, float(scores.get(name, 0))) for feature, name in zip(features, names, strict=True)]
    ranked.sort(key=lambda entry: -entry[1])
    return {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "importance_type": "total_gain",
        "boosting_rounds": booster.num_boosted_rounds(),
        "features": [{"rank": rank, "feature": feature, "score": score}
                     for rank, (feature, score) in enumerate(ranked, start=1)],
    }


def write_feature_importance_pdf(output, result):
    """
    Write ranked feature bars with at most one page of labels per figure

    Args:
        output:  Destination PDF file
        result:  Ranked feature importance record
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    features = result["features"]
    temporary = output.with_name(f".{output.name}.tmp")
    with PdfPages(temporary) as pdf:
        for start in range(0, len(features), FEATURES_PER_PAGE):
            page = features[start:start + FEATURES_PER_PAGE]
            scores = [entry["score"] for entry in page]
            labels = [f"{entry['rank']}. {entry['feature']}" for entry in page]
            rows = range(len(page))
            figure = Figure(figsize=(12, max(3, len(page) * 0.26 + 1.5)), layout="constrained")
            axis = figure.subplots()
            axis.barh(rows, scores, color=PLOT_COLORS[0])
            axis.set_yticks(rows, labels, fontsize=8, parse_math=False)
            axis.invert_yaxis()
            axis.set_xlim(0, 1.05 * max(scores) if max(scores) > 0 else 1)
            axis.set_xlabel("Total gain (sum over tree splits)")
            axis.set_title(f"XGBoost feature importance · {result['checkpoint']}\n"
                           f"Ranks {start + 1}–{start + len(page)} of {len(features):,}", loc="left")
            axis.set_axisbelow(True)
            axis.grid(axis="x", alpha=0.25)
            pdf.savefig(figure)
            figure.clear()
    temporary.replace(output)
