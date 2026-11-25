# Standalone CPU clustering and multithreaded Lund feature extraction
#
# mikael.mieskolainen@cern.ch (2026)

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from drivers.toptag.graphs.lund_graphs import LUND_NAMES, lund_rows


def test_standalone_lund_stream(tmp_path):
    """
    Compare C++ observables with Torch and check parallel output across batch boundaries

    Args:
        tmp_path:  Local build directory
    Returns:
        None:  Each input jet has the expected Lund row
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = Path(__file__).resolve().parents[1] / "twistformer/geometry/native"
    executable = tmp_path / "lund_rows"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", *sorted(source.glob("*.cc")),
                    "-o", executable], check=True)
    names = (source / "top_features.txt").read_text().splitlines()
    assert tuple(name for name in names if name and not name.startswith("#")) == LUND_NAMES

    rng = np.random.default_rng(395)
    vectors = rng.normal(size=(8, 18, 3)).astype(np.float32)
    energy = np.sqrt(1. + (vectors**2).sum(axis=2, keepdims=True))
    jets = np.concatenate((energy, vectors), axis=2)
    for index, count in enumerate((0, 1, 2, 3, 6, 8, 12, 18)):
        jets[index, count:] = 0.
    expected = lund_rows(jets)
    order = np.arange(4103) % len(jets)
    data = jets[order].tobytes()
    for threads in (1, 4):
        result = subprocess.run([executable, "18", str(threads)], input=data, capture_output=True, check=True)
        rows = np.frombuffer(result.stdout, dtype=np.float32).reshape(len(order), len(LUND_NAMES))
        np.testing.assert_allclose(rows[:len(jets)], expected, atol=1.e-6, rtol=1.e-5)
        np.testing.assert_array_equal(rows, rows[:len(jets)][order])
    import torch
    from drivers.toptag.graphs.lund_graphs import NATIVE_NAMES, native_top_rows
    result = subprocess.run([executable, "18", "2", "16383"], input=jets.tobytes(), capture_output=True, check=True)
    with native_top_rows(NATIVE_NAMES, 1) as extract:
        reference = extract(torch.from_numpy(jets)).numpy()
    np.testing.assert_allclose(np.frombuffer(result.stdout, dtype=np.float32).reshape(reference.shape),
                               reference, rtol=1.e-6, atol=1.e-6)
    result = subprocess.run([executable, "18"], input=data[:15], capture_output=True)
    assert result.returncode != 0 and b"Truncated input jet" in result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_lund_model_rows(device, monkeypatch, tmp_path):
    """
    Check the default reader against the Torch model inputs for each Lund block

    Args:
        device:     Device for prong observables
        monkeypatch: Isolate the build cache and check backend dispatch
        tmp_path:   Temporary cache directory
    Returns:
        None:  Complete and reordered Lund columns agree for a partial batch of jets
    """
    import torch
    from drivers.toptag import reader
    from twistformer import build_experiment, load_config

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if shutil.which("c++") is None:
        pytest.skip("C++17 compiler required")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config = load_config("configs/toptag/twistgraph_lund_graphs.yml")
    assert config.driver["lund_backend"] == "native"
    names = build_experiment(config).driver.input_spec.features
    lengths = np.array([6, 90, 1, 32])
    generator = torch.Generator().manual_seed(702)
    xyz = torch.randn(int(lengths.sum()), 3, generator=generator)
    jets = reader.Jets.__new__(reader.Jets)
    jets.p4 = torch.cat(((1. + xyz.square().sum(1, keepdim=True)).sqrt(), xyz), 1)
    jets.offsets = np.r_[0, lengths.cumsum()]
    jets.max_particles = 200
    jets.ids = list(range(len(lengths)))
    jets.labels = torch.tensor([1, 0, 1, 0])
    jets.weights = np.array([.5, 2., 1.5, 1.])
    reference = reader.JetGraphs(jets, "lund_graphs", device, graph_features=names, lund_backend="torch")

    def unexpected_torch_lund(*args):
        raise AssertionError("Expected compiled CPU clustering")

    monkeypatch.setattr(reader, "_lund_block", unexpected_torch_lund)
    for selection in (names, ["lund_ca_split1_kt"], ["lund_flow_abs_rapidity", "jet_mass", "lund_kt_split1_kt"],
                      ["jet_mass"]):
        dataset = reader.JetGraphs(jets, "lund_graphs", device, graph_features=selection, lund_threads=2)
        torch.testing.assert_close(dataset.rows, reference.rows[:, :, [names.index(n) for n in selection]],
                                   rtol=1.e-5, atol=2.e-5)
        assert dataset.ids == reference.ids
        torch.testing.assert_close(dataset.labels, reference.labels)
        np.testing.assert_equal(dataset.weights, reference.weights)
        torch.testing.assert_close(dataset.observables, reference.observables, rtol=0, atol=0)


def test_compiled_decay_hypergraphs():
    """
    Compare Durham scales with exhaustive merging and check hypergraph symmetries

    Returns:
        None:  Mixed native blocks preserve schema, particle flow and source order
    """
    import torch
    from drivers.toptag.graphs.lund_graphs import COLOR_NAMES, GROOMED_NAMES, PARTITION_NAMES, PARTICLE_NAMES, REST_NAMES, native_top_rows

    rng = np.random.default_rng(722)
    xyz = rng.normal(size=(12, 11, 3))*20
    p = np.concatenate((np.sqrt(1 + np.square(xyz).sum(-1, keepdims=True)), xyz), axis=-1)
    for i in range(len(p)):
        p[i, i:] = 0.
    momenta = torch.tensor(p, dtype=torch.float32)
    names = REST_NAMES + PARTITION_NAMES + GROOMED_NAMES + COLOR_NAMES + PARTICLE_NAMES
    with native_top_rows(names, 3) as rows:
        result = rows(momenta)
        permuted = rows(momenta[:, list(reversed(range(p.shape[1])))])
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, permuted, rtol=2.e-5, atol=2.e-5)
    for selected in (COLOR_NAMES, GROOMED_NAMES, PARTITION_NAMES, tuple(reversed(REST_NAMES[::13]))):
        with native_top_rows(selected, 1) as rows:
            torch.testing.assert_close(rows(momenta), result[:, [names.index(name) for name in selected]])
    from drivers.toptag.graphs import jet_graph_kernel
    selected = (COLOR_NAMES[0], REST_NAMES[0], PARTICLE_NAMES[7])
    kernel = jet_graph_kernel("rest_hypergraphs", "cpu", True, selected)
    torch.testing.assert_close(kernel(momenta, torch.arange(len(p))), result[:, [names.index(n) for n in selected]])
    counts = result[:, -len(PARTICLE_NAMES):].reshape(len(p), 16, 16, 2)[..., 0].sum(-1)
    torch.testing.assert_close(counts, torch.arange(len(p), dtype=torch.float32)[:, None].expand_as(counts))

    # A common longitudinal boost preserves the rest-frame and transverse observables
    boost = p.copy()
    boost[..., 0] = np.cosh(.6)*p[..., 0] + np.sinh(.6)*p[..., 3]
    boost[..., 3] = np.sinh(.6)*p[..., 0] + np.cosh(.6)*p[..., 3]
    with native_top_rows(REST_NAMES, 2) as rows:
        boosted = rows(torch.tensor(boost, dtype=torch.float32))
        torch.testing.assert_close(boosted, result[:, :len(REST_NAMES)], rtol=3.e-4, atol=3.e-4)

    for i in range(3, len(p)):
        jet = momenta[i, :i].double().numpy()
        energy, spatial = jet[:, 0], jet[:, 1:]
        total = jet.sum(0)
        mass = np.sqrt(total[0]**2 - total[1:] @ total[1:])
        rest_energy = (energy*total[0] - spatial @ total[1:])/mass
        rest_p = spatial - ((rest_energy + energy)/(total[0] + mass))[:, None]*total[1:]
        axes = np.column_stack((rest_energy, rest_p))
        while len(axes) >= 3:
            unit = axes[:, 1:] / np.linalg.norm(axes[:, 1:], axis=1)[:, None]
            distances = 2 * np.minimum.outer(axes[:, 0], axes[:, 0])**2 * (1 - unit @ unit.T)
            distances[np.triu_indices(len(axes))] = np.inf
            hi, lo = np.unravel_index(np.argmin(distances), distances.shape)
            if len(axes) in (3, 4, 6, 8):
                observed = result[i, names.index(f"rest_n{len(axes)}_next_kt")]
                np.testing.assert_allclose(observed, np.sqrt(distances[hi, lo]), rtol=2.e-6)
            axes[lo] += axes[hi]
            axes = np.delete(axes, hi, axis=0)
