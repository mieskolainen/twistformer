# Class yields in particle gun energy bins
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
from twistformer.logging import print


def validate_energy_weights(fractions, edges):
    """
    Check the relative yields of the configured gun energies

    Args:
        fractions:  Relative yields per bin
        edges:  Truth energy bin boundaries
    """
    weights = np.asarray(fractions, dtype=float)
    if weights.shape != (len(edges) - 1,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("energy_weights requires one finite nonnegative weight per energy bin")
    if not np.any(weights > 0):
        raise ValueError("energy_weights requires at least one positive bin")


def normalize_energy_weights(weights, edges):
    """
    Convert relative weights to probabilities across the configured gun energies

    Args:
        weights:  Relative nonnegative yields per energy bin
        edges:  Truth energy bin boundaries
    Returns:
        fractions:  Target energy fractions with unit sum
    """
    weights = np.asarray(weights, dtype=float)
    validate_energy_weights(weights, edges)
    scaled = weights / weights.max()
    return scaled / scaled.sum()


def report_energy_weights(setting, weights, fractions):
    """
    Print the input spectrum and the normalized fractions used for event weights

    Args:
        setting:  Steering option name
        weights:  Supplied bin weights
        fractions:  Normalized bin probabilities
    """
    values = np.array2string(fractions, precision=8, separator=", ", max_line_width=160)
    print(f"Normalized {setting} {weights} -> {values} (sum = {fractions.sum():g})")


def spectrum_weights(generator_weights, labels, bins, fractions):
    """
    Set the gun spectrum at fixed class yield for the energies present

    Args:
        generator_weights:  Event weights before spectrum weighting
        labels:  Event classes
        bins:  Gun energy bin indices
        fractions:  Target bin probabilities
    Returns:
        weights:  Event weights for the target spectrum
    """
    fractions = np.asarray(fractions)
    weights = np.array(generator_weights, dtype=float, copy=True)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Gun energy weighting requires finite nonnegative event weights")
    for label in np.unique(labels):
        selected = labels == label
        yields = np.bincount(bins[selected], weights=weights[selected], minlength=len(fractions))
        present = np.bincount(bins[selected], minlength=len(fractions)) > 0
        if ((yields == 0) & present & (fractions > 0)).any():
            raise ValueError("Each populated gun bin with positive energy weight needs positive event yield")
        norm = fractions[present].sum()
        scales = np.zeros_like(yields)
        if norm > 0:
            np.divide(fractions * (yields.sum() / norm), yields, out=scales, where=yields > 0)
        weights[selected] *= scales[bins[selected]]
    return weights
