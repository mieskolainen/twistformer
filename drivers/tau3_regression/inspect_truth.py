# Generator tau record diagnostics for regression truth association
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
import json

import awkward as ak
import numpy as np
import uproot
import yaml

from drivers.tau3_classification.reader import MC_PARTICLE_VARS, MC_RELATION_VARS
from .reader import VERTEX_BRANCHES, ENDPOINT_BRANCHES, tau_targets

def describe_event(arrays, entry):
    """
    Report tau candidates and their direct relations without selecting a target

    Args:
        arrays:  One-event MC particle and parent-relation arrays
        entry:  Original ROOT entry number
    Returns:
        record: Tau and related-particle status, geometry and parent links
    """
    pdg, status, simulator = [ak.to_numpy(arrays[key][0]) for key in MC_PARTICLE_VARS[:3]]
    begin, end, links = [ak.to_numpy(arrays[key][0]) for key in MC_RELATION_VARS]
    parents, children, errors = {}, {i: [] for i in range(len(pdg))}, []
    for i in range(len(pdg)):
        lo, hi = int(begin[i]), int(end[i])
        if not 0 <= lo <= hi <= len(links):
            errors.append(f"Particle {i}: invalid parent offsets [{lo}, {hi})")
            parents[i] = []
            continue
        parents[i] = links[lo:hi].tolist()
        for parent in parents[i]:
            if 0 <= parent < len(pdg):
                children[parent].append(i)
            else:
                errors.append(f"Particle {i}: invalid parent index {parent}")
    taus = np.flatnonzero(np.abs(pdg) == 15).tolist()
    selected = set(taus)
    for i in taus:
        selected.update(parent for parent in parents[i] if 0 <= parent < len(pdg))
        selected.update(children[i])
    records = []
    for i in sorted(selected):
        records.append(
            {
                "index": i,
                "pdg": int(pdg[i]),
                "generator_status": int(status[i]),
                "simulator_status": hex(int(simulator[i]) & 0xFFFFFFFF),
                "created_in_simulation": bool((int(simulator[i]) >> 30) & 1),
                "parents": parents[i],
                "children": children[i],
                "momentum_GeV": [float(arrays[key][0][i]) for key in MC_PARTICLE_VARS[3:]],
                "vertex_mm": [float(arrays[key][0][i]) for key in VERTEX_BRANCHES],
                "endpoint_mm": [float(arrays[key][0][i]) for key in ENDPOINT_BRANCHES],
            }
        )
    return {"entry": int(entry), "particle_count": len(pdg), "tau_indices": taus, "particles": records, "relation_errors": errors}

def main(argv=None):
    """
    Read selected truth entries without streaming calorimeter hits or training

    Args:
        argv:  Command-line arguments or None for process arguments
    Returns:
        None: JSON diagnostics are printed to standard output
    """
    parser = argparse.ArgumentParser(description="Inspect tau truth candidates and their direct relations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Read the first file and tree from a tau regression YAML config")
    source.add_argument("--input", help="Input ROOT path")
    parser.add_argument("--all-files", action="store_true", help="Inspect every file in --config")
    parser.add_argument("--tree", help="Override the configured tree, default events")
    parser.add_argument("--entries", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args(argv)
    if min(args.entries) < 0:
        parser.error("ROOT entries must be nonnegative")
    if args.all_files and not args.config:
        parser.error("--all-files requires --config")
    paths, tree_name = [args.input], "events"
    if args.config:
        with open(args.config) as stream:
            app = yaml.safe_load(stream)["driver"]
        paths, tree_name = (app["files"] if args.all_files else app["files"][:1], app.get("tree", "events"))
    tree_name = args.tree or tree_name
    branches = MC_PARTICLE_VARS + MC_RELATION_VARS + list(VERTEX_BRANCHES + ENDPOINT_BRANCHES)
    reports = []
    for path in paths:
        records = []
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as root:
            tree = root[tree_name]
            for entry in dict.fromkeys(args.entries):
                if entry >= tree.num_entries:
                    parser.error(f"Entry {entry} lies outside {tree.num_entries} stored events in {path}")
                arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="ak")
                record = describe_event(arrays, entry)
                try:
                    record["target"] = tau_targets(arrays, np.array([entry]))[0].tolist()
                    record["selection_error"] = None
                except ValueError as error:
                    record["target"] = None
                    record["selection_error"] = str(error)
                records.append(record)
        reports.append({"input": path, "tree": tree_name, "events": records})
    print(json.dumps({"files": reports} if args.all_files else reports[0], indent=2))

if __name__ == "__main__":
    main()
