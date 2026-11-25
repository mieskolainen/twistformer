# Check the geometry in the ALLEGRO barrel ECAL ROOT files
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml


CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs/pi0gamma/common.yml"
COLLECTION = "ECalBarrelModuleThetaMergedPositioned"


@dataclass
class Layer:
    hits: int = 0
    radius_sum_mm: float = 0.0
    theta_by_index: dict = field(default_factory=dict)
    example: tuple | None = None

    def spacing(self):
        """
        Measure theta spacing between the closest occupied readout indices

        Returns:
            spacing:  Median angular step for these pairs in rad, or NaN
        """
        indices = sorted(self.theta_by_index)
        if len(indices) < 2:
            return np.nan
        gaps = np.diff(indices)
        angles = [self.theta_by_index[i] for i in indices]
        return np.median(np.abs(np.diff(angles))[gaps == gaps.min()])


def scan(tree, geometry, events):
    """
    Measure the cylindrical radius and polar angles of positioned ECAL cells

    Args:
        tree:  ROOT event tree
        geometry:  Barrel readout geometry from the pi0gamma config
        events:  Number of entries to read from the start of the tree
    Returns:
        layers:  Measurements by zero-based layer index
    """
    keys = set(tree.keys())
    branches = []
    for name in ("cellID", "position.x", "position.y", "position.z"):
        leaf = f"{COLLECTION}.{name}"
        branch = f"{COLLECTION}/{leaf}"
        if branch not in keys and leaf not in keys:
            raise ValueError(f"Missing ROOT branch {leaf}")
        branches.append(branch if branch in keys else leaf)

    layers = {}
    entry = 0
    for arrays in tree.iterate(branches, entry_stop=events, step_size="32 MB", how=tuple):
        counts = ak.to_numpy(ak.num(arrays[0]))
        if any(not np.array_equal(counts, ak.num(axis)) for axis in arrays[1:]):
            raise ValueError("Cell IDs and positions must align in each event")
        cell_id, x, y, z = (ak.to_numpy(ak.flatten(a)) for a in arrays)
        radius = np.hypot(x.astype(np.float64), y)
        if not (np.isfinite(radius) & (radius > 0) & np.isfinite(z)).all():
            raise ValueError("Barrel cell positions must be finite with positive radius")
        layer_id = (cell_id >> geometry["layer_shift"]) & ((1 << geometry["layer_bits"]) - 1)
        # ALLEGRO theta field occupies bits 30 to 39
        theta_index = (cell_id >> 30) & 1023
        theta = np.arctan2(radius, z)
        event_ends = np.cumsum(counts)
        for index in np.unique(layer_id):
            selected = np.flatnonzero(layer_id == index)
            layer = layers.setdefault(int(index), Layer())
            layer.hits += len(selected)
            layer.radius_sum_mm += radius[selected].sum()
            rings, first = np.unique(theta_index[selected], return_index=True)
            layer.theta_by_index.update(zip(rings, theta[selected[first]]))
            if layer.example is None:
                i = selected[0]
                event = entry + np.searchsorted(event_ends, i, side="right")
                layer.example = (event, cell_id[i], x[i], y[i], z[i])
        entry += len(counts)
    return layers


def main():
    """
    Report the measured layer geometry and one hit per observed strip layer
    """
    parser = argparse.ArgumentParser(description="Locate fine theta strips in an ALLEGRO pi0gamma ROOT file")
    parser.add_argument("file", type=Path, help="Reconstructed pi0gamma ROOT file")
    parser.add_argument("--max-events", type=int, help="Scan the first N events (default: all)")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")

    driver = yaml.safe_load(CONFIG_FILE.read_text())["driver"]
    geometry = driver["geometry"]
    try:
        with args.file.open("rb") as input_file, uproot.open(input_file) as root:
            tree = root[driver["tree"]]
            events = min(args.max_events or tree.num_entries, tree.num_entries)
            layers = scan(tree, geometry, events)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"{args.file}: {error}\n")
    if not layers:
        parser.exit(1, f"No positioned ECAL cells in the first {events} events\n")

    print(f"{args.file} | {events} events | {COLLECTION}")
    print("Layer       Hits   Mean r [mm]   Observed theta step [mrad]")
    strips = []
    for index, layer in sorted(layers.items()):
        spacing = layer.spacing()
        radius = layer.radius_sum_mm / layer.hits
        step = f"{1000 * spacing:.3f}" if np.isfinite(spacing) else "unknown"
        print(f"{index:5} {layer.hits:10} {radius:13.3f} {step:>28}")
        if np.isclose(spacing, geometry["theta_pitch_rad"], rtol=0.01, atol=0):
            strips.append((index, radius, layer.example))

    if not strips:
        print("No observed theta step matches the configured fine pitch. Sparse hits can skip adjacent strips.")
    for index, radius, example in strips:
        event, cell_id, x, y, z = example
        depth = radius - geometry["inner_radius_mm"]
        print(f"Strip layer {index} (zero-based), r = {radius:.3f} mm, {depth:.3f} mm behind the active inner face")
        print(f"  Entry {event}, cellID {cell_id}, (x, y, z) = ({x:.3f}, {y:.3f}, {z:.3f}) mm")


if __name__ == "__main__":
    main()
