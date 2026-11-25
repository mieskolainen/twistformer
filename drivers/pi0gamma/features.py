# Cluster cell coordinates and readout geometry
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field
import numpy as np

FEATURES = ("x", "y", "z", "log_energy", "strip", "energy_fraction", "size_theta", "size_phi")
LOCAL_FEATURES = FEATURES[:4]


def cartesian_features(position, energy, geometry):
    """
    Rotate barrel cells into a rigid Cartesian frame at the ECAL inner surface

    Args:
        position:  Cell positions in mm, shape (N,3)
        energy:  Cell energies in GeV, shape (N,)
        geometry:  Readout constants, only the inner radius is needed
    Returns:
        result: Negative beam offset, azimuth displacement, tangent-plane depth,
            log energy and retained cell observables
    """
    position = np.asarray(position, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or position.shape != (len(energy), 3):
        raise ValueError("Cell positions and energies must align")
    if not np.isfinite(energy).all():
        raise ValueError("Cell energies must be finite")
    geometry.validate(local=True)
    keep = energy > 0
    position, energy = position[keep], energy[keep]
    if not len(energy):
        return np.empty((0, 4), dtype=np.float32), {}
    if not np.isfinite(position).all():
        raise ValueError("Positive-energy cell positions must be finite")
    rho = np.hypot(position[:, 0], position[:, 1])
    if (rho <= 0).any():
        raise ValueError("Barrel cells require positive cylindrical radius")
    total = energy.sum()
    fraction = energy / total
    barycentre = fraction @ position
    radius = np.hypot(barycentre[0], barycentre[1])
    if radius < 1e-12 * np.dot(fraction, rho):
        raise ValueError("Cell azimuth barycentre is undefined")
    cosine, sine = barycentre[:2] / radius
    # Right-handed axes are negative beam, azimuth and outward radial
    features = np.column_stack((
        (barycentre[2] - position[:, 2]) / 50,
        (-sine * position[:, 0] + cosine * position[:, 1]) / 50,
        (cosine * position[:, 0] + sine * position[:, 1] - geometry.inner_radius_mm) / 50,
        np.log(energy),
    )).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Cluster features must be finite")
    return features, {"cluster_energy": float(total), "n_cells": float(len(energy))}


def angular_features(position, energy, geometry, *, snap_theta=False):
    """
    Centre retained cells on their energy barycentre in the angular frame

    Args:
        position:  Cell positions in mm, shape (N,3)
        energy:  Cell energies in GeV, shape (N,)
        geometry:  Readout constants, only the inner radius is needed without snapping
        snap_theta:  Snap the theta barycentre to a strip centre for eight-feature mode
    Returns:
        result: Theta offset, phi offset, depth, log energy and retained cell observables
    """
    position = np.asarray(position, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or position.shape != (len(energy), 3):
        raise ValueError("Cell positions and energies must align")
    if not np.isfinite(energy).all():
        raise ValueError("Cell energies must be finite")
    geometry.validate(local=not snap_theta)
    keep = energy > 0
    position, energy = position[keep], energy[keep]
    if not len(energy):
        return np.empty((0, 4), dtype=np.float32), {}
    if not np.isfinite(position).all():
        raise ValueError("Positive-energy cell positions must be finite")
    rho = np.hypot(position[:, 0], position[:, 1])
    if (rho <= 0).any():
        raise ValueError("Barrel cells require positive cylindrical radius")
    theta = np.arctan2(rho, position[:, 2])
    phi = np.arctan2(position[:, 1], position[:, 0])
    total = energy.sum()
    fraction = energy / total
    theta0 = np.dot(fraction, theta)
    if snap_theta:
        theta0 = geometry.theta_offset_rad + np.floor(
            (theta0 - geometry.theta_offset_rad) / geometry.theta_pitch_rad + 0.5
        ) * geometry.theta_pitch_rad
    sine, cosine = np.dot(fraction, np.sin(phi)), np.dot(fraction, np.cos(phi))
    if np.hypot(sine, cosine) < 1e-12:
        raise ValueError("Cell azimuth barycentre is undefined")
    phi0 = np.arctan2(sine, cosine)
    dphi = np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))
    features = np.column_stack((
        geometry.inner_radius_mm * (theta - theta0) / 50,
        geometry.inner_radius_mm * dphi / 50,
        (rho - geometry.inner_radius_mm) / 50,
        np.log(energy),
    )).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Cluster features must be finite")
    return features, {"cluster_energy": float(total), "n_cells": float(len(energy))}


@dataclass
class Geometry:
    inner_radius_mm: float | None = None
    theta_pitch_rad: float | None = None
    phi_pitch_rad: float | None = None
    theta_offset_rad: float | None = None
    layer_shift: int | None = None
    layer_bits: int | None = None
    theta_merge: list[int] = field(default_factory=list)
    phi_merge: list[int] = field(default_factory=list)

    def validate(self, *, local=False):
        """
        Check the readout constants before reading detector cells

        Args:
            local:  Validate only the inner radius for four local features
        Returns:
            None: Invalid or missing geometry raises ValueError
        """
        names = ("inner_radius_mm",) if local else ("inner_radius_mm", "theta_pitch_rad", "phi_pitch_rad")
        for name in names:
            value = getattr(self, name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Set driver.geometry.{name} from the detector readout")
        if local:
            return
        if self.theta_offset_rad is None or not np.isfinite(self.theta_offset_rad):
            raise ValueError("Set driver.geometry.theta_offset_rad to a strip centre in radians")
        if (
            self.layer_shift is None or self.layer_bits is None
            or not 0 <= self.layer_shift < 64 or not 1 <= self.layer_bits <= 64 - self.layer_shift
        ):
            raise ValueError("Set the cellID layer_shift and layer_bits in driver.geometry")
        if (
            not self.theta_merge or len(self.theta_merge) != len(self.phi_merge)
            or min(self.theta_merge + self.phi_merge) < 1
        ):
            raise ValueError("Set positive theta_merge and phi_merge for each readout layer")


def cell_features(position, energy, cell_id, geometry):
    """
    Centre positive-energy cluster cells in the detector angular frame

    Args:
        position:  Cell positions in mm, shape (N,3)
        energy:  Cell energies in GeV, shape (N,)
        cell_id:  Packed readout identifiers, shape (N,)
        geometry:  Validated readout constants
    Returns:
        result: Eight cell features in 50 mm units and cluster observables
    """
    position = np.asarray(position, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    cell_id = np.asarray(cell_id, dtype=np.uint64)
    if position.shape != (len(energy), 3) or cell_id.shape != energy.shape:
        raise ValueError("Cell positions, energies and identifiers must align")
    if not np.isfinite(energy).all():
        raise ValueError("Cell energies must be finite")
    keep = energy > 0
    position, energy, cell_id = position[keep], energy[keep], cell_id[keep]
    if not len(energy):
        return np.empty((0, 8), dtype=np.float32), {}
    angular, obs = angular_features(position, energy, geometry, snap_theta=True)
    fraction = energy / energy.sum()
    g = geometry
    layer = ((cell_id >> np.uint64(g.layer_shift)) & np.uint64((1 << g.layer_bits) - 1)).astype(np.int64)
    if (layer >= len(g.theta_merge)).any():
        raise ValueError("Cell layer is outside the configured readout geometry")
    theta_merge, phi_merge = np.asarray(g.theta_merge)[layer], np.asarray(g.phi_merge)[layer]
    features = np.column_stack((
        angular,
        theta_merge == 1,
        fraction,
        g.inner_radius_mm * g.theta_pitch_rad * theta_merge / 50,
        g.inner_radius_mm * g.phi_pitch_rad * phi_merge / 50,
    )).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Cluster features must be finite")
    return features, obs
