# Tau truth targets aligned with calorimeter hit chunks
#
# mikael.mieskolainen@cern.ch (2026)

import awkward as ak
import numpy as np
from drivers.tau3_classification.reader import MC_PARTICLE_VARS, MC_RELATION_VARS, iter_tau3

VERTEX_BRANCHES = tuple(f"MCParticles/MCParticles.vertex.{axis}" for axis in "xyz")
ENDPOINT_BRANCHES = tuple(f"MCParticles/MCParticles.endpoint.{axis}" for axis in "xyz")

def tau_targets(arrays, entries):
    """
    Extract decay-tau momentum and full displacement across generator record copies

    Args:
        arrays:  Selected EDM4hep MC particle branches
        entries:  Original ROOT entry numbers
    Returns:
        targets: Cartesian px,py,pz in GeV and endpoint minus vertex in mm
    """
    tau = (abs(arrays[MC_PARTICLE_VARS[0]]) == 15) & (((arrays[MC_PARTICLE_VARS[2]] >> 30) & 1) == 0)
    counts = ak.to_numpy(ak.sum(tau, axis=1))
    offsets = np.r_[0, np.cumsum(ak.to_numpy(ak.num(tau, axis=1)))]
    candidates = np.flatnonzero(ak.to_numpy(ak.flatten(tau)))
    candidate_offsets = np.r_[0, np.cumsum(counts)]
    first = np.zeros(len(counts), dtype=np.int64)
    last = np.zeros(len(counts), dtype=np.int64)
    single = counts == 1
    first[single] = last[single] = candidates[candidate_offsets[:-1][single]]
    invalid = [(int(entries[i]), "no generator tau") for i in np.flatnonzero(counts == 0)]
    multiple = np.flatnonzero(counts > 1)
    if len(multiple):
        pdg = ak.to_numpy(ak.flatten(arrays[MC_PARTICLE_VARS[0]]))
        generator = ak.to_numpy(ak.flatten(arrays[MC_PARTICLE_VARS[1]]))
        simulator = ak.to_numpy(ak.flatten(arrays[MC_PARTICLE_VARS[2]]))
        decay_particle = (generator > 0) & (((simulator >> 30) & 1) == 0) & (np.abs(pdg) != 15)
        begin, end, parents = (ak.to_numpy(ak.flatten(arrays[key])) for key in MC_RELATION_VARS)
        relation_offsets = np.r_[0, np.cumsum(ak.to_numpy(ak.num(arrays[MC_RELATION_VARS[2]], axis=1)))]
        for event in multiple:
            nodes = candidates[candidate_offsets[event] : candidate_offsets[event + 1]]
            node_set = set(nodes.tolist())
            predecessor, successor = {}, {}
            reason = None
            for node in nodes:
                lo, hi = int(begin[node]), int(end[node])
                relation_count = relation_offsets[event + 1] - relation_offsets[event]
                if not 0 <= lo <= hi <= relation_count:
                    reason = "invalid tau parent offsets"
                    break
                links = parents[relation_offsets[event] + lo : relation_offsets[event] + hi]
                if np.any((links < 0) | (links >= offsets[event + 1] - offsets[event])):
                    reason = "invalid tau parent index"
                    break
                for parent in set((links + offsets[event]).tolist()):
                    if parent not in node_set or pdg[parent] != pdg[node]:
                        continue
                    if node in predecessor or parent in successor:
                        reason = "branching tau copy chain"
                        break
                    predecessor[node], successor[parent] = parent, node
                if reason:
                    break
            roots = node_set - predecessor.keys()
            chains, covered = [], set()
            if reason is None:
                for root in roots:
                    current, visited = root, {root}
                    while current in successor and successor[current] not in visited:
                        current = successor[current]
                        visited.add(current)
                    if current in successor:
                        reason = "cyclic tau copy links"
                        break
                    chains.append((root, current, visited))
                    covered.update(visited)
                if covered != node_set:
                    reason = "cyclic or disconnected tau copy links"
            if reason is None and len(chains) > 1:
                daughters = {node: set() for node in node_set}
                start, stop = offsets[event : event + 2]
                for child in np.flatnonzero(decay_particle[start:stop]) + start:
                    lo, hi = int(begin[child]), int(end[child])
                    if not 0 <= lo <= hi <= relation_count:
                        reason = "invalid generator daughter parent offsets"
                        break
                    links = parents[relation_offsets[event] + lo : relation_offsets[event] + hi]
                    if np.any((links < 0) | (links >= stop - start)):
                        reason = "invalid generator daughter parent index"
                        break
                    for parent in links + start:
                        if parent in daughters:
                            daughters[parent].add(int(pdg[child]))
                active = [chain for chain in chains if any(code != 22 for node in chain[2] for code in daughters[node])]
                if reason is None and len(active) == 1:
                    terminal = active[0][1]
                    decay = daughters[terminal]
                    has_neutrino = (16 if pdg[terminal] == 15 else -16) in decay
                    has_visible = any(abs(code) not in (12, 14, 16, 22) for code in decay)
                    if has_neutrino and has_visible:
                        chains = active
                    else:
                        reason = "stored tau daughters do not identify a visible tau decay"
                elif reason is None:
                    reason = f"{len(active)} tau chains have non-photon generator daughters, expected one"
                if reason is not None:
                    summary = [
                        {
                            "root": int(root - start),
                            "terminal": int(terminal - start),
                            "pdg": int(pdg[terminal]),
                            "terminal_daughters": sorted(daughters[terminal]),
                            "chain_daughters": sorted({code for node in members for code in daughters[node]}),
                        }
                        for root, terminal, members in chains
                    ]
                    reason += f", tau decay records {summary}"
            if reason is None and len(chains) == 1:
                first[event], last[event] = chains[0][:2]
                continue
            invalid.append(
                (
                    int(entries[event]),
                    reason
                    or (
                        f"{len(roots)} unlinked tau roots among {len(nodes)} candidates"
                        if len(roots) > 1
                        else "cyclic or disconnected tau copy links"
                    ),
                )
            )
    if invalid:
        raise ValueError(
            "Tau regression requires exactly one generator tau with an unambiguous decay association, "
            f"invalid ROOT entries and reasons {invalid[:8]}. "
            "Inspect the stored truth with python -m drivers.tau3_regression.inspect_truth "
            "--config configs/tau3_regression/transformer.yml --entries " + " ".join(str(entry) for entry, _ in invalid[:2])
        )
    momentum = np.column_stack([ak.to_numpy(ak.flatten(arrays[key]))[last] for key in MC_PARTICLE_VARS[3:]])
    vertex = np.column_stack([ak.to_numpy(ak.flatten(arrays[key]))[first] for key in VERTEX_BRANCHES])
    endpoint = np.column_stack([ak.to_numpy(ak.flatten(arrays[key]))[last] for key in ENDPOINT_BRANCHES])
    targets = np.column_stack((momentum, endpoint.astype(np.float64) - vertex))
    targets = targets.astype(np.float32)
    invalid = ~np.isfinite(targets).all(axis=1)
    if invalid.any():
        raise ValueError(f"Nonfinite tau targets at ROOT entries {entries[invalid][:8].tolist()}")
    return targets

def iter_tau3_regression(path, tree_name, *, read_targets=True, **options):
    """
    Stream the classification hit representation with six truth coordinates

    Args:
        path:  Input ROOT file
        tree_name:  ROOT event tree name
        read_targets:  Read MC labels and acceptance or use hit-only inference
        options:  Shared tau3 hit selection and streaming parameters
    Yields:
        chunk: Compact hits, six-vector targets, weights and ROOT entry IDs
    """
    yield from iter_tau3(
        path,
        tree_name,
        target_branches=VERTEX_BRANCHES + ENDPOINT_BRANCHES if read_targets else (),
        target_builder=tau_targets if read_targets else None,
        apply_fiducial=read_targets,
        **options,
    )
