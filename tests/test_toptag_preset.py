# Top-tagging preset and kinematic preprocessing validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
import pytest
import torch
from twistformer import load_config, build_experiment, Trainer, load_predictor
from twistformer.api.contracts import Sample
from twistformer.training.checkpoint import read_checkpoint
from twistformer.training.engine import make_optimizer
from twistformer.evaluation.runner import evaluate
from drivers.toptag.features import constituent_features


def test_toptag_preset():
    cfg = load_config("configs/toptag/transformer.yml")
    exp = build_experiment(cfg)
    assert len(exp.driver.input_spec.features) == 15
    assert cfg.model.pair_bias == "jet" and cfg.model.pooling == "attention_query"
    assert (cfg.model.width, cfg.model.depth, cfg.model.heads) == (64, 4, 4)
    assert cfg.model.norm == cfg.model.head_norm == "rms"
    assert cfg.model.ffn_activation == "geglu" and cfg.model.ffn_norm == "none"
    assert cfg.model.embedding == "two_stage" and cfg.model.embedding_norm == "layer"
    assert (
        cfg.model.head_layers == 1
        and cfg.model.head_residual
        and cfg.model.head_zero_init
    )
    assert (
        cfg.model.dropout == cfg.model.attention_dropout == cfg.model.head_dropout == 0
    )
    assert not cfg.preprocessing.normalize_features and cfg.task.balance_classes
    assert cfg.training.epochs == 35
    assert cfg.training.batch_size == 128
    assert cfg.training.eval_batch_size == 2048
    assert cfg.training.learning_rate == 3e-4 and cfg.training.optimizer_betas == [
        0.9,
        0.999,
    ]
    assert cfg.training.optimizer_eps == 1e-8 and cfg.training.weight_decay == 0.01
    assert cfg.training.decay_norm_bias and cfg.training.scheduler == "cosine"
    assert cfg.training.grad_clip == 5 and cfg.training.token_dropout == 0.01
    assert not cfg.training.amp and not cfg.training.compile and not cfg.training.tf32
    assert cfg.evaluation.efficiencies == [0.3, 0.5, 0.9]


@pytest.mark.parametrize("name", ["transformer", "pelican", "twistformer"])
def test_toptag_schedule_presets(name):
    """
    Check the parameter budget and model-specific optimizer recipes

    Args:
        name:  Top-tagging training preset
    Returns:
        None: Presets match the 208k parameter budget and selected training recipe
    """
    config = load_config(f"configs/toptag/{name}.yml")
    model = build_experiment(config).model
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert abs(count / 208000 - 1) < 0.01
    if name == "pelican":
        assert (config.model.width, config.model.pelican_message_width) == (132, 78)
        assert config.model.depth == 5 and not model.backbone.triangles
        assert not list(model.head.hidden.children())
    cfg = config.training
    assert cfg.optimizer == "adamw" and cfg.optimizer_betas == [0.9, 0.999]
    reference = load_config(
        "configs/toptag/pelican.yml" if name == "pelican"
        else "configs/toptag/gatr_projective.yml"
    ).training
    assert cfg.weight_decay == reference.weight_decay and cfg.decay_norm_bias
    optimizer = make_optimizer(model, replace(cfg, fused_optimizer=False))
    assert all(group["weight_decay"] == reference.weight_decay for group in optimizer.param_groups)
    assert cfg.scheduler == reference.scheduler
    assert cfg.epochs == 35 and cfg.warmup_epochs == reference.warmup_epochs
    assert cfg.restart_period == 4 and cfg.restart_mult == 2
    assert cfg.decay_epochs == 3 and cfg.decay_gamma == 0.5
    assert cfg.learning_rate == reference.learning_rate and cfg.min_learning_rate == 0.0


def test_kinematics15_and_rotation():
    cfg = load_config("configs/toptag/transformer.yml")
    app = build_experiment(cfg).driver
    p = torch.tensor([[5.0, 3.0, 0.0, 4.0], [5.0, 0.0, 4.0, 3.0]])
    features = constituent_features(p, "kinematics15")
    torch.testing.assert_close(features[:, :4], p)
    torch.testing.assert_close(features[:, 4], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(features[:, 7], torch.zeros(2))
    torch.testing.assert_close(features[:, 10], torch.tensor([3.0, 4.0]).log())
    torch.testing.assert_close(features[:, 11], torch.tensor([5.0, 5.0]).log())
    torch.testing.assert_close(
        features[:, 12], torch.tensor([3.0 / 5.0, 4.0 / 5.0]).log()
    )
    torch.testing.assert_close(features[:, 13], torch.tensor([0.5, 0.5]).log())
    torch.testing.assert_close(features[:, 14], features[:, 8:10].norm(dim=-1))
    sample = Sample(features.clone(), torch.tensor(1), p.clone())
    rotated = app.augment(sample)
    torch.testing.assert_close(sample.features, features)
    torch.testing.assert_close(sample.p4, p)
    torch.testing.assert_close(
        rotated.features, constituent_features(rotated.p4, "kinematics15")
    )
    torch.testing.assert_close(
        rotated.features[:, [4, 5, 8, 9, 10, 11, 12, 13, 14]],
        features[:, [4, 5, 8, 9, 10, 11, 12, 13, 14]],
        atol=2e-6,
        rtol=2e-5,
    )
    rest = constituent_features(torch.tensor([[2.0, 0.0, 0.0, 0.0]]), "kinematics15")
    assert rest.isfinite().all()
    cfg.driver["rotate"] = False
    assert build_experiment(cfg).driver.augment(sample) is sample


def test_toptag_training_checkpoint_and_evaluation(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    values = np.zeros((16, 800), dtype=np.float32)
    for i in range(16):
        values[i, :3] = [5 + i / 10, 5, 2]
        values[i, 200:203] = [3, 0, 0]
        values[i, 400:403] = [0, 4, 0]
        values[i, 600:603] = [4, 3, 0]
    frame = pd.DataFrame(
        values,
        columns=[f"{name}_{i}" for name in ("E", "PX", "PY", "PZ") for i in range(200)],
    )
    frame["is_signal_new"] = np.arange(16) % 2
    files = {}
    for split, subset in (
        ("train", frame.iloc[:8]),
        ("val", frame.iloc[8:12]),
        ("test", frame.iloc[12:]),
    ):
        path = tmp_path / f"{split}.h5"
        subset.to_hdf(path, key="table", format="table")
        files[split] = str(path)
    cfg = load_config("configs/toptag/transformer.yml")
    cfg.driver["files"] = files
    cfg.training = replace(
        cfg.training,
        device="cpu",
        workers=0,
        epochs=1,
        batch_size=2,
        scheduler="none",
        warmup_epochs=0,
    )
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "run").fit(exp)
    state = read_checkpoint(best)
    assert state["config"]["model"] == cfg.to_dict()["model"]
    torch.testing.assert_close(state["preprocessing"]["feature_mean"], torch.zeros(15))
    torch.testing.assert_close(state["preprocessing"]["feature_std"], torch.ones(15))
    assert state["model"]["head.linear.weight"].count_nonzero()
    assert state["model"]["backbone.bias.to_bias.weight"].count_nonzero()
    predictor = load_predictor(best, "cpu")
    result = evaluate(
        predictor,
        predictor.experiment.driver.dataset("test"),
        tmp_path / "evaluation",
        plots=False,
    )
    assert np.isfinite(result["preprocessed_loss"])
    assert [p["requested_efficiency"] for p in result["signal_operating_points"]] == cfg.evaluation.efficiencies


@pytest.mark.parametrize("basis, width", [
    ("harmonic_graphs", 19), ("local_graphs", 35), ("complete_graphs", 348),
    ("prong_graphs", 427), ("moment_graphs", 562),
])
def test_toptag_graph_basis(basis, width):
    """
    Check graph order, azimuthal wrapping and permutation symmetry on a two-prong jet

    Args:
        basis:  Top-tag graph basis
        width:  Expected number of descriptors
    Returns:
        None:  Graph coefficients agree with the analytic two-prong result
    """
    pytest.importorskip("xgboost")
    from drivers.toptag.graphs import jet_descriptors, jet_graph_kernel

    config = load_config("configs/toptag/twistgraph_moment_graphs.yml", [f"driver.feature_set={basis}"])
    driver = build_experiment(config).driver
    assert driver.options.feature_set == basis
    phi = np.array([np.pi - 0.05, -np.pi + 0.05])
    pt = np.array([2., 1.])
    p4 = np.column_stack((pt, pt * np.cos(phi), pt * np.sin(phi), np.zeros_like(pt)))
    result = jet_descriptors(p4, basis)
    assert torch.is_tensor(result)
    names = driver.input_spec.features
    assert len(names) == width and driver.input_spec.geometry == "none"
    values = dict(zip(names, result, strict=True))
    assert values["edge"] == pytest.approx(4 / 9 * 0.1**2)
    assert values["triangle"] == pytest.approx(0, abs=1e-12)
    assert values["power_phi_k5"] == pytest.approx((5 + 4 * np.cos(0.5)) / 9)
    if "phi_0.2_edge" in values:
        assert values["phi_0.2_edge"] == pytest.approx(7 / 9)
    p4[:, 1:3] = np.column_stack((pt * np.cos(phi + 0.7), pt * np.sin(phi + 0.7)))
    np.testing.assert_allclose(jet_descriptors(p4[::-1], basis), result, atol=1e-7)
    rest = jet_descriptors([[1., 0., 0., 0.]], basis)
    np.testing.assert_array_equal(rest[:3], [1., 0., 1.])
    assert not rest[3:].any()
    # Include a constituent with longitudinal momentum only and a fully padded jet
    jets = [p4, np.vstack((p4, [3., 0., 0., -3.])), np.array([[1., 0., 0., 0.]])]
    batch = torch.zeros(4, 8, 4, dtype=torch.float64)
    for i, jet in enumerate(jets):
        batch[i, :len(jet)] = torch.from_numpy(jet)
    counts = torch.tensor([len(jet) for jet in jets] + [0])
    rows = jet_graph_kernel(basis, "cpu", False)(batch, counts)
    expected = torch.stack([jet_descriptors(jet, basis) for jet in jets])
    torch.testing.assert_close(rows[:3], expected, rtol=1e-6, atol=1e-7)
    assert not rows[3].any()
    mlp = torch.nn.Sequential(torch.nn.Linear(width, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    logits = mlp(rows)
    assert logits.shape == (4, 1) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert mlp[0].weight.grad is not None


def test_three_prong_mass_edges_and_angular_graphs():
    """
    Compare flow contractions and resolved subjet masses to a three-particle jet
    """
    from drivers.toptag.graphs import BASES, jet_descriptors

    pt = np.array([90., 65., 40.])
    eta, phi = np.array([.1, -.12, .3]), np.array([3.03, -3.05, 2.75])
    p = np.stack((pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), 1)
    row = jet_descriptors(p, "prong_graphs")
    values = dict(zip(BASES["prong_graphs"], row, strict=True))
    dphi = (phi[:, None] - phi + np.pi) % (2 * np.pi) - np.pi
    angle = np.hypot(eta[:, None] - eta, dphi)
    w = pt / pt.sum()
    triangle = np.einsum("i,j,k,ij,jk,ki->", w, w, w, angle, angle, angle)
    assert values["angular_beta1_triangle"] == pytest.approx(triangle)
    kernel = np.exp(-angle**2 / (2 * .2**2))
    ends = w * (kernel @ w)
    assert values["flow_r0.2_path3"] == pytest.approx(ends @ kernel @ ends)
    assert values["tau3_beta1"] == pytest.approx(0, abs=1e-12)
    for (i, j), name in zip(((0, 1), (0, 2), (1, 2)), ("prong12", "prong13", "prong23")):
        pair = p[i] + p[j]
        assert values[name + "_mass"] == pytest.approx(np.sqrt(pair[0]**2 - pair[1:] @ pair[1:]))
    np.testing.assert_allclose(jet_descriptors(p[[2, 0, 1]], "prong_graphs"), row, atol=2e-6)


def test_rooted_moments_against_explicit_angular_edges():
    """
    Check mixed vertex weights and shared-root graph contractions

    Returns:
        None:  Graph coefficients match the explicit spherical kernel
    """
    from drivers.toptag.graphs.moment_graphs import MOMENT_NAMES, moment_rows

    points = torch.tensor([[[.2, -.1], [.1, .3], [-.4, .2], [.5, .5]]], dtype=torch.float64)
    w = torch.tensor([[.6, .3, .1, 0.]], dtype=torch.float64)
    result = dict(zip(MOMENT_NAMES, moment_rows(points, w)[0].numpy(), strict=True))
    plane = points[0].numpy() / .25
    norm2 = (plane**2).sum(axis=1, keepdims=True)
    sphere = np.column_stack((2 * plane, 1 - norm2)) / (1 + norm2)
    edge = (1 - sphere @ sphere.T) / 2
    root = w[0].numpy()
    charge = np.sqrt(root)
    charge /= charge.sum()
    leaf = edge @ charge
    triangle = np.einsum("ij,j,jk,k,ki->i", edge, charge, edge, charge, edge)
    prefix = "moment_r0.25_sqrt_pt_"
    assert result[prefix + "star12"] == pytest.approx(root @ leaf**12)
    assert result[prefix + "bridge6_6"] == pytest.approx((root * leaf**6) @ edge @ (root * leaf**6))
    assert result[prefix + "four_triangles"] == pytest.approx(root @ triangle**4)
    assert result[prefix + "triangle_star4"] == pytest.approx(root @ (triangle * leaf**4))


def test_lund_basis_and_graph_feature_selection():
    """
    Compare exact three-body observables and exercise the selected-feature dataset

    Returns:
        None:  Named PyTorch rows match measured subjet kinematics
    """
    pytest.importorskip("xgboost")
    from drivers.toptag.graphs import jet_descriptors
    from drivers.toptag.graphs.lund_graphs import LUND_NAMES, lund_rows
    from drivers.toptag.reader import JetGraphs, Jets

    pt = np.array([100., 55., 20.])
    phi = np.array([.05, -.2, .34])
    eta = np.array([.15, -.01, .4])
    p4 = np.column_stack((pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)))
    w = pt / pt.sum()
    angle = np.hypot(eta[:, None] - eta, phi[:, None] - phi)
    features = lund_rows(p4[None])[0]
    assert len(features) == len(set(LUND_NAMES)) == 836
    fields = dict(zip(LUND_NAMES, features, strict=True))
    assert fields["lund_flow_beta1_ecf3"] == pytest.approx(w.prod() * angle[1, 0] * angle[2, 0] * angle[2, 1])
    assert fields["lund_flow_beta1_ecf4"] == 0
    np.testing.assert_allclose(lund_rows(p4[None, ::-1])[0], features, atol=2e-6)

    config = load_config("configs/toptag/twistgraph_lund_graphs.yml")
    selected = ["lund_flow_beta1_ecf3", "jet_mass", "lund_ca_split1_kt"]
    config.driver["graph_features"] = selected
    driver = build_experiment(config).driver
    assert driver.input_spec.features == tuple(selected)
    jets = Jets.__new__(Jets)
    jets.p4 = torch.tensor(p4, dtype=torch.float32)
    jets.offsets = np.array([0, len(p4)])
    jets.max_particles = 200
    jets.ids, jets.weights, jets.labels = ["jet"], np.ones(1), torch.ones(1, dtype=torch.long)
    dataset = JetGraphs(jets, "lund_graphs", graph_features=selected)
    np.testing.assert_allclose(dataset[0].features[0], [
        fields[selected[0]], jet_descriptors(p4, "prong_graphs")[2], fields[selected[2]],
    ], rtol=3e-6)


def test_toptag_graph_input_packing():
    """
    Check momentum ties, truncation and full-jet observables for a partial block

    Returns:
        None:  Batched and scalar readers select the same constituents
    """
    from drivers.toptag.reader import Jets

    jets = Jets.__new__(Jets)
    lengths = np.array([2, 24, 3, 24])
    jets.offsets = np.r_[0, lengths.cumsum()]
    jets.max_particles = 5
    jets.feature_set = "p4"
    jets.labels = torch.arange(len(lengths)) % 2
    jets.weights = np.ones(len(lengths))
    jets.ids = list(map(str, range(len(lengths))))
    index = torch.arange(sum(lengths), dtype=torch.float32)
    jets.p4 = torch.stack((index + 3, index % 2, torch.zeros_like(index), index), 1)
    jets.p4[jets.offsets[2]:jets.offsets[3], 1] = torch.tensor([3., 2., 1.])
    selected, counts, observables = jets.graph_inputs(1, 4, 4)
    for offset in range(3):
        sample = jets[offset + 1]
        count = len(sample.p4)
        assert counts[offset] == count
        torch.testing.assert_close(selected[offset, :count], sample.p4)
        assert not selected[offset, count:].any()
        for name, values in observables.items():
            assert values[offset] == pytest.approx(sample.observables[name])
    assert not counts[3] and not selected[3].any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def test_lund_tensor_batch_matches_single_jet_rows(device):
    """
    Check PyTorch jet feature columns across a padded batch boundary

    Args:
        device:  Device for the Torch prong kernel
    Returns:
        None:  Host rows match single-jet predictions in requested order
    """
    from drivers.toptag.graphs import jet_graph_kernel

    rng = np.random.default_rng(12)
    p = torch.zeros(4, 90, 4)
    counts = torch.tensor([90, 4, 1, 0])
    for jet, count in enumerate(counts[:3]):
        pt = np.sort(rng.uniform(1., 50., count))[::-1]
        eta, phi = rng.normal(0., .15, (2, count))
        p[jet, :count] = torch.from_numpy(np.column_stack(
            (pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta))))
    model = jet_graph_kernel("lund_graphs", device, False)
    reference = torch.cat([model(p[i:i+1].to(device), counts[i:i+1].to(device)) for i in range(3)])
    result = model(p.to(device), counts.to(device))
    assert result.device.type == device
    torch.testing.assert_close(result[:3], reference, atol=1.e-6, rtol=1.e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def test_lund_multiplicity_batches_preserve_selected_rows(device):
    """Keep source order and selected graph values across compact clustering batches"""
    from drivers.toptag.graphs import BASES, jet_graph_kernel
    from drivers.toptag.reader import _lund_block

    names = ("jet_mass", "d1_g000", "lund_ca_split1_kt", "lund_flow_beta1_ecf3")
    columns = [BASES["lund_graphs"].index(name) for name in names]
    counts = torch.tensor([6, 18, 0, 39, 1, 10, 23, 3, 38, 25, 2, 12])
    p = torch.randn(len(counts), 40, 4, generator=torch.Generator().manual_seed(24))
    p[..., 0] = 1.05 * p[..., 1:].norm(dim=2)
    order = p[..., 1:3].square().sum(2).argsort(dim=1, descending=True)
    p = p.gather(1, order[..., None].expand_as(p))
    p.masked_fill_((torch.arange(40) >= counts[:, None])[..., None], 0)
    full = jet_graph_kernel("lund_graphs", device, False)
    selected = jet_graph_kernel("lund_graphs", device, False, names)
    expected = full(p.to(device), counts.to(device))[:, columns].cpu()
    rows = _lund_block(selected, p, counts, torch.device(device), 4)
    torch.testing.assert_close(rows, expected, atol=2.e-5, rtol=1.e-5)


@pytest.mark.parametrize("normalize,balance", [(False, False), (True, True)])
def test_toptag_packed_histograms(normalize, balance):
    """
    Check prepared jet rows and weighted histogram cuts against the sample loader

    Args:
        normalize:  Standardize the jet observables
        balance:    Fit inverse class yield weights
    Returns:
        None:  Raw and sampled jet tables reproduce the same quantile matrices
    """
    pytest.importorskip("xgboost")
    from torch.utils.data import Subset
    from drivers.toptag.reader import JetGraphs
    from twistformer.data.preprocessing import PreparedDataset
    from twistformer.training.xgboost import ShowerBatches, _packed_reader, shower_matrix

    config = load_config("configs/toptag/twistgraph_lund_graphs.yml", [
        "training.eval_batch_size=8",
        "plots.feature_distributions.categories=[{name: jet_pt, observable: jet_pt, edges: [0, 16, 32]}]",
    ])
    config.preprocessing.normalize_features = normalize
    experiment = build_experiment(config)
    jets = JetGraphs.__new__(JetGraphs)
    count, width = 33, len(experiment.driver.input_spec.features)
    jets.rows = torch.randn(count, 1, width)
    jets.ids = list(range(count))
    jets.labels = torch.arange(count) % 2
    jets.weights = np.linspace(0., 3., count)
    jets.observables = {"jet_pt": torch.arange(count, dtype=torch.float64),
                        "n_constituents": torch.arange(count, dtype=torch.float64) + 1}
    samples = list(jets)
    fit = experiment.driver.create_preprocessor(config=config.preprocessing)
    reference = experiment.driver.create_preprocessor(config=config.preprocessing)
    fit.fit(jets, class_count=2 if balance else 0)
    reference.fit(samples, class_count=2 if balance else 0)
    torch.testing.assert_close(fit.state, reference.state)
    indices = [8, 17, 1, 2, 2, 10, 20, 0, 3, 11, 32]
    packed = PreparedDataset(Subset(jets, indices), fit)
    scalar = PreparedDataset(Subset(samples, indices), reference)
    expected = scalar.__getitems__(range(len(indices)))
    features, labels, weights = _packed_reader(packed)(slice(None))
    np.testing.assert_array_equal(features, torch.cat([s.features for s in expected]))
    np.testing.assert_array_equal(labels, [s.target for s in expected])
    np.testing.assert_array_equal(weights, np.asarray([s.weight for s in expected], dtype=np.float32))
    for dataset in (packed, scalar):
        batches = ShowerBatches(experiment, dataset, ("jet_pt", "n_constituents"))
        columns = [values for _, _, _, values in batches.arrays()]
        for name, values in jets.observables.items():
            np.testing.assert_array_equal(np.concatenate([batch[name] for batch in columns]), values[indices])
    assert _packed_reader(PreparedDataset(jets, fit, augment=lambda sample: sample)) is None
    matrix, original = (shower_matrix(experiment, dataset) for dataset in (packed, scalar))
    assert matrix.feature_names == original.feature_names
    np.testing.assert_array_equal(matrix.get_label(), original.get_label())
    np.testing.assert_array_equal(matrix.get_weight(), original.get_weight())
    for new, old in zip(matrix.get_quantile_cut(), original.get_quantile_cut(), strict=True):
        np.testing.assert_array_equal(new, old)
    jets.rows[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit.fit(jets, class_count=2 if balance else 0)
    with pytest.raises(ValueError, match="finite"):
        _packed_reader(packed)(slice(None))
