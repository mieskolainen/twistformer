# Public experiment API
#
# mikael.mieskolainen@cern.ch (2026)

"""Strict, serializable configuration shared by the Python API and CLI"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import re
import types
from typing import get_args, get_origin, get_type_hints, Union
import yaml
from twistformer.flows.config import FlowConfig

@dataclass
class XGBoostParameters:
    booster: str = "gbtree"
    objective: str = "binary:logistic"
    tree_method: str = "hist"
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    max_leaves: int = 0
    grow_policy: str = "depthwise"
    num_parallel_tree: int = 1
    max_bin: int = 256
    max_cached_hist_node: int | None = None

    def __post_init__(self):
        """
        Validate native binary tree parameters
        """
        if self.booster not in {"gbtree", "dart"} or self.objective != "binary:logistic":
            raise ValueError("XGBoost classification requires binary:logistic and a tree booster")
        if self.tree_method not in {"auto", "hist", "approx", "exact"}:
            raise ValueError("Unknown XGBoost tree_method")
        if min(self.reg_alpha, self.reg_lambda, self.gamma, self.min_child_weight) < 0:
            raise ValueError("XGBoost regularization parameters must be nonnegative")
        if not all(0 < rate <= 1 for rate in (self.subsample, self.colsample_bytree, self.colsample_bylevel, self.colsample_bynode)) or self.max_bin < 2:
            raise ValueError("XGBoost sampling fractions must be in (0,1] and max_bin must be at least 2")
        if self.grow_policy not in {"depthwise", "lossguide"} or self.max_leaves < 0 or self.num_parallel_tree < 1:
            raise ValueError("Invalid XGBoost tree growth settings")
        if self.max_cached_hist_node is not None and self.max_cached_hist_node < 1:
            raise ValueError("max_cached_hist_node must be positive")

@dataclass
class ModelConfig:
    family: str = "transformer"
    xgboost_threads: int = 0
    xgboost_sketch_rows: int = 0
    xgboost_sketch_seed: int | None = None
    xgboost_params: dict = field(default_factory=dict)
    attention_neighbors: int = 0
    attention: str = "standard"
    attention_backend: str = "sdpa"
    projective_channels: int = 2
    projective_coordinates: list[str] = field(default_factory=list)
    projective_input_scale: float = 1.0
    pair_bias: str = "none"
    width: int = 64
    depth: int = 4
    checkpoint_blocks: bool = False
    heads: int = 4
    expansion: int = 2
    norm: str = "layer"
    final_norm: bool = True
    global_fusion: str = "readout"
    parameter_fusion: str = "readout"
    auxiliary_features: list[str] = field(default_factory=list)
    embedding: list[int] | str = "mlp"
    embedding_norm: str = "none"
    ffn_activation: str = "gelu"
    ffn_norm: str = "none"
    head_norm: str = "layer"
    head_layers: int = 0
    head_widths: list[int] = field(default_factory=list)
    head_activation: str = "gelu"
    head_hidden_norm: str = "none"
    head_init: str = "default"
    head_expansion: int = 2
    head_residual: bool = False
    head_dropout: float = 0.0
    head_zero_init: bool = False
    dropout: float = 0.1
    embedding_dropout: float | None = None
    residual_dropout: float | None = None
    attention_dropout: float = 0.0
    pooling: str = "mean"
    pooling_depth: int = 1
    pooling_heads: int | None = None
    pooling_queries: int = 1
    pooling_dropout: float = 0.0
    pair_channels: int = 0
    representation_channels: int = 0
    twistor_transport: str = "phase"
    twistor_precision: str = "float64"
    twistor_separate_norm: bool = False
    twistor_qk: bool = False
    twistor_geometry_scale: float = 0.1
    twistor_representation_scale: float = 0.1
    twistor_residual_scaling: str = "none"
    reference_token: str = "none"
    beam_direction: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    geometry_update: bool = True
    twistor_kind: str = "penrose"
    twistor_detector: bool = False
    three_point: bool = False
    triangle_k: int = 16
    triangle_gamma: float = 0.5
    query_chunk: int = 32
    triangle_kj: int | None = None
    triangle_kk: int | None = None
    triangle_shared_candidates: bool = False
    triangle_distinct: bool = False
    jet_features: list[str] = field(
        default_factory=lambda: ["p4dot", "m2", "dR", "akt", "cos3d", "z_sym", "lund_log_invDR", "lund_log_kt", "eec"]
    )
    jet_eec_betas: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0])
    jet_radius: float = 0.8
    jet_exponent: float = -1.0
    jet_hidden: int = 64
    jet_clamp: float = 3.0
    jet_head_mode: str = "per_head"
    jet_head_gains: bool = True
    jet_scale_init: float = 0.2
    jet_block_scale_init: float = 0.1
    jet_grad_geometry: bool = False
    graph: str = "full"
    neighbors: int = 16
    graph_event_features: bool = False
    graph_pooling: str = "mean"
    graph_gate_hidden: int = 0
    graph_update_scale: float = 0.1
    graph_activation: str = "gelu"
    graph_edge_hidden: int = 0
    graph_mass_statistics: bool = False
    graph_input_features: str = "invariant3"
    graph_transform: str = "asinh"
    pelican_activation: str = "leaky_relu"
    pelican_message_width: int = 0
    pelican_average_nobj: float = 49.0
    pelican_momentum_scale: float = 1.0
    pelican_factorize: bool = True
    pelican_multiplicity: bool = True
    pelican_triangle: bool = False
    pelican_triangle_channels: int = 32
    pelican_triangle_scale: float = 0.1
    gatr_algebra: str = "projective"
    gatr_mv_channels: int = 8
    gatr_input: str = "p4"
    gatr_coordinates: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    gatr_input_scale: float = 1.0
    gatr_center: bool = False
    gatr_spurions: str = "none"
    gatr_reference_tokens: bool = False
    gatr_readout: str = "invariant"
    gatr_global_token: bool = False
    gatr_projective_improved: bool = True
    gatr_lorentz_proper: bool = True
    gatr_multi_query: bool = False
    gatr_attention_ratio: int = 1
    gatr_reinsert: bool = True
    gatr_head_scale: bool = True
    gatr_norm_epsilon: float = 0.01
    gatr_bilinear_norm: bool = True
    gatr_distance_epsilon: float = 0.001
    gatr_checkpoint: bool = False
    gatr_residual_scaling: bool = True

@dataclass
class TaskConfig:
    name: str = "classification"
    lossfunc: str | None = None
    balance_classes: bool = False
    class_weights: list[float] = field(default_factory=list)
    label_smoothing: float = 0.0
    min_std: float = 1e-3
    mixture_components: int = 1

    def __post_init__(self):
        """
        Resolve the loss and require agreement with the prediction type
        """
        losses = {
            "classification": "cross_entropy",
            "gaussian_regression": "gaussian_nll",
            "covariant_regression": "mse",
            "flow_matching": "flow_matching",
            "flow_nll": "flow_nll",
        }
        if self.name not in losses:
            raise ValueError("Unknown task.name")
        expected = losses[self.name]
        if self.lossfunc is None:
            self.lossfunc = expected
        if self.lossfunc != expected:
            raise ValueError(f"task.lossfunc for {self.name} must be {expected}, got {self.lossfunc!r}")

@dataclass
class PreprocessingConfig:
    # Train-token channel z-scores on features, never on separate physical p4
    normalize_features: bool = True
    # Train-event target z-scores for regression, ignored for class labels
    normalize_targets: bool = True
    # Train-event channel z-scores on declared global features
    normalize_globals: bool = True
    normalize_parameters: bool = True
    flow_transforms: list[dict] = field(default_factory=list)

@dataclass
class TrainingConfig:
    epochs: int = 30
    max_steps: int = 0
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    optimizer_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    optimizer_eps: float = 1e-8
    decay_norm_bias: bool = True
    scheduler: str = "cosine"
    scheduler_interval: str = "epoch"
    min_learning_rate: float = 0.0
    warmup_epochs: int = 0
    restart_period: int = 4
    restart_mult: int = 2
    decay_epochs: int = 3
    decay_gamma: float = 0.5
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    ema: bool = False
    ema_decay: float = 0.999
    ema_eval: bool = True
    grad_clip: float = 1.0
    accumulation: int = 1
    workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 4
    eval_prefetch_factor: int = 2
    seed: int = 101
    device: str = "auto"
    strategy: str = "single"
    ddp_find_unused_parameters: bool = True
    ddp_static_graph: bool = False
    ddp_bucket_cap_mb: float = 25.0
    fused_optimizer: bool = False
    amp: bool = False
    amp_dtype: str = "auto"
    tf32: bool = False
    compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool | None = True
    profile_batches: int = 0
    augmentation: str = "online"
    token_dropout: float = 0.0
    token_min_keep: int = 1
    patience: int = 0
    save_every: int = 5
    keep_checkpoints: int = 3
    monitor: str = "val_loss"
    mode: str = "min"

    def __post_init__(self):
        """
        Validate the shared Torch, NumPy and XGBoost seed
        """
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("training.seed must be an integer in [0, 2**32)")

@dataclass
class FeatureDistributionConfig:
    bins: int = 32
    categories: list[dict] = field(default_factory=list)

    def __post_init__(self):
        """Require a nonnegative number of bins"""
        if self.bins < 0:
            raise ValueError("plots.feature_distributions.bins must be nonnegative")

    def resolve_categories(self, driver, observables=None):
        """
        Resolve and validate observable bins against the current driver steering

        Args:
            driver:  Driver configuration mapping
            observables:  Optional driver observable schema
        Returns:
            categories: Category definitions with explicit bin edges
        """
        from twistformer.evaluation.categories import validate_categories

        categories = []
        for category in self.categories:
            edges = category.get("edges")
            if isinstance(edges, str) and edges.startswith("driver."):
                key = edges.removeprefix("driver.")
                if key not in driver:
                    raise ValueError(f"Unknown driver bin edge setting: {edges}")
                category = {**category, "edges": driver[key]}
            categories.append(category)
        validate_categories(categories, observables)
        return categories

@dataclass
class PlotConfig:
    feature_distributions: FeatureDistributionConfig = field(default_factory=FeatureDistributionConfig)

@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 200
    efficiencies: list[float] = field(default_factory=lambda: [0.5, 0.8])
    categories: list[dict] = field(default_factory=list)
    batch_size: int | None = None
    driver: dict[str, list[float] | None] = field(default_factory=dict)
    bootstrap_seed: int = 101

    def __post_init__(self):
        """
        Validate evaluation batch size and bootstrap settings
        """
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 0 or self.bootstrap_samples == 1:
            raise ValueError("Evaluation bootstrap_samples must be zero or an integer >= 2")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("Evaluation bootstrap_seed must be a nonnegative integer")
        if self.batch_size is not None and (type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("Evaluation batch_size must be a positive integer")
        if self.driver.keys() - {"energy_weights"}:
            raise ValueError("Only energy_weights can be overridden in the evaluation driver settings")

@dataclass
class CalibrationConfig:
    temperature_scaling: bool = False
    min_temperature: float = 0.05
    max_temperature: float = 20.0

    def __post_init__(self):
        """Require a positive temperature interval containing the unscaled model"""
        if not 0 < self.min_temperature <= 1 <= self.max_temperature:
            raise ValueError("Temperature bounds must be positive and include one")

@dataclass
class ReweightingConfig:
    path: str = ""
    batch_size: int = 1024

    def __post_init__(self):
        """
        Require a positive density-ratio inference batch size
        """
        if self.batch_size < 1:
            raise ValueError("Reweighting batch_size must be positive")

@dataclass
class Config:
    driver: dict
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reweighting: ReweightingConfig = field(default_factory=ReweightingConfig)
    schema_version: int = 1
    flow: FlowConfig = field(default_factory=FlowConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    def to_dict(self):
        """
        Serialize the resolved configuration with the effective model settings

        Returns:
            config: Plain dictionary suitable for YAML and checkpoints
        """
        config = asdict(self)
        if self.model.family == "xgboost":
            config["model"] = {
                key: value for key, value in config["model"].items()
                if key in {"family", "depth"} or key.startswith("xgboost_")
            }
        return config

def strict_dataclass(cls, values):
    """
    Validate configuration fields and construct their typed record

    Args:
        cls:  Configuration dataclass type
        values:  Dictionary of explicitly supplied fields
    Returns:
        config: Validated dataclass instance
    """
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    values = dict(values)

    def matches(value, expected):
        origin, args = get_origin(expected), get_args(expected)
        if origin in (Union, types.UnionType):
            return any(matches(value, arg) for arg in args)
        if expected is float:
            return type(value) in (int, float) and math.isfinite(value)
        if expected is int:
            return type(value) is int
        if origin is list:
            return isinstance(value, list) and all(matches(v, args[0]) for v in value)
        if origin is dict:
            return isinstance(value, dict) and all(matches(k, args[0]) and matches(v, args[1]) for k, v in value.items())
        return isinstance(value, expected)

    for name, expected in get_type_hints(cls).items():
        if name not in values:
            continue
        if expected is float and isinstance(values[name], str):
            try:
                values[name] = float(values[name])  # YAML scientific notation
            except ValueError:
                pass
        if not matches(values[name], expected):
            raise ValueError(f"{cls.__name__}.{name} must be {expected}, got {values[name]!r}")
    return cls(**values)

def parse_config(data):
    """
    Validate all shared steering options and model compatibility

    Args:
        data:  Nested experiment configuration dictionary
    Returns:
        config: Resolved Config instance
    """
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a mapping")
    data = deepcopy(data)
    training_options = data.get("training", {})
    plots = data.get("plots", {})
    if isinstance(plots, dict) and "feature_distributions" in plots:
        plots["feature_distributions"] = strict_dataclass(FeatureDistributionConfig, plots["feature_distributions"])
    data["plots"] = strict_dataclass(PlotConfig, plots)
    for key, cls in [
        ("model", ModelConfig),
        ("task", TaskConfig),
        ("preprocessing", PreprocessingConfig),
        ("training", TrainingConfig),
        ("evaluation", EvaluationConfig),
        ("calibration", CalibrationConfig),
        ("reweighting", ReweightingConfig),
        ("flow", FlowConfig),
    ]:
        data[key] = strict_dataclass(cls, data.get(key, {}))
    cfg = strict_dataclass(Config, data)
    if cfg.reweighting.path and cfg.task.name != "classification":
        raise ValueError("Density-ratio reweighting requires classification")
    from twistformer.evaluation.categories import validate_categories

    validate_categories(cfg.evaluation.categories)
    if cfg.schema_version != 1:
        raise ValueError("Unsupported configuration schema")
    if not isinstance(cfg.driver, dict) or "name" not in cfg.driver:
        raise ValueError("driver.name is required")
    cfg.plots.feature_distributions.resolve_categories(cfg.driver)
    m, t, tr = cfg.model, cfg.task, cfg.training
    if m.attention_backend not in {"sdpa", "flex", "auto"}:
        raise ValueError("model.attention_backend must be sdpa, flex or auto")
    if m.attention_backend == "flex":
        if m.family not in {"transformer", "gatr"}:
            raise ValueError("FlexAttention is supported by transformer and gatr encoders")
        if m.pair_bias != "none":
            raise ValueError("FlexAttention requires pair_bias=none for learned bias gradients")
        if m.attention_dropout:
            raise ValueError("FlexAttention requires attention_dropout=0")
    if m.attention_neighbors < 0:
        raise ValueError("attention_neighbors must be nonnegative")
    if m.attention_neighbors and (
        m.family != "transformer" or m.reference_token != "none" or m.pair_bias != "none"
        or cfg.preprocessing.normalize_features
    ):
        raise ValueError("attention_neighbors requires transformer, no reference tokens or pair bias, and normalize_features=false")
    if m.family not in {"transformer", "twistformer", "pelican", "triformer", "gatr", "xgboost"}:
        raise ValueError(f"Unknown model family: {m.family}")
    if cfg.driver["name"] == "pi0gamma" and cfg.driver.get("shower_basis") and tr.token_dropout:
        raise ValueError("Shower descriptors require token_dropout=0")
    if m.family == "xgboost":
        if "scheduler" not in training_options:
            tr.scheduler = "none"
        if t.name != "classification" or tr.token_dropout:
            raise ValueError("XGBoost requires classification with token_dropout=0")
        if tr.strategy != "single" or tr.ema or tr.compile:
            raise ValueError("XGBoost requires single-process boosting without EMA or torch.compile")
        if cfg.reweighting.path or t.label_smoothing:
            raise ValueError("XGBoost uses driver weights and unsmoothed binary labels")
        m.xgboost_params = asdict(strict_dataclass(XGBoostParameters, m.xgboost_params))
        if m.xgboost_sketch_rows < 0:
            raise ValueError("xgboost_sketch_rows must be nonnegative")
        if m.xgboost_sketch_rows and m.xgboost_params["tree_method"] != "hist":
            raise ValueError("xgboost_sketch_rows requires tree_method=hist")
        if m.xgboost_sketch_seed is not None and (m.xgboost_sketch_seed < 0 or not m.xgboost_sketch_rows):
            raise ValueError("xgboost_sketch_seed needs a nonnegative seed and positive xgboost_sketch_rows")
        if m.xgboost_threads < 0:
            raise ValueError("xgboost_threads must be nonnegative")
        if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", tr.device):
            raise ValueError("XGBoost device must be cpu, cuda or auto")
    if m.attention not in {"standard", "hyperbolic", "projective"} or m.pair_bias not in {"none", "mandelstam", "jet"}:
        raise ValueError("Invalid attention or pair_bias")
    if m.family != "transformer" and m.attention != "standard":
        raise ValueError("attention is configurable only for transformer")
    if m.family in {"twistformer", "pelican", "gatr"} and m.pair_bias != "none":
        raise ValueError("pair_bias belongs to transformer/triformer")
    defaults = ModelConfig()
    if m.projective_channels < 1:
        raise ValueError("projective_channels must be positive")
    if m.projective_coordinates and (len(m.projective_coordinates) != 3 or len(set(m.projective_coordinates)) != 3):
        raise ValueError("projective_coordinates require three distinct feature names")
    if not math.isfinite(m.projective_input_scale) or m.projective_input_scale <= 0:
        raise ValueError("projective_input_scale must be finite and positive")
    if m.attention == "projective":
        if m.projective_coordinates and cfg.preprocessing.normalize_features:
            raise ValueError("Projective coordinates require normalize_features=false and a common projective_input_scale")
        if not m.projective_coordinates and m.projective_input_scale != 1.0:
            raise ValueError("projective_input_scale requires projective_coordinates")
    elif any(
        getattr(m, name) != getattr(defaults, name)
        for name in ("projective_channels", "projective_coordinates", "projective_input_scale")
    ):
        raise ValueError("projective options require attention=projective")
    from twistformer.geometry.clifford import SIGNATURES

    if m.gatr_algebra not in SIGNATURES:
        raise ValueError("Unknown GATr algebra")
    if m.gatr_mv_channels < 1 or m.gatr_attention_ratio < 1:
        raise ValueError("GATr channel counts and attention ratio must be positive")
    if m.gatr_input not in {"p4", "coordinates"} or m.gatr_readout not in {"invariant", "components", "logits"}:
        raise ValueError("Invalid GATr input or readout")
    if m.gatr_spurions not in {"none", "time", "beam", "beam_plane", "frame"}:
        raise ValueError("Invalid GATr reference channels")
    if len(m.gatr_coordinates) != 3 or len(set(m.gatr_coordinates)) != 3:
        raise ValueError("GATr coordinates require three distinct feature names")
    if any(not math.isfinite(v) or v <= 0 for v in (m.gatr_input_scale, m.gatr_norm_epsilon, m.gatr_distance_epsilon)):
        raise ValueError("GATr scales and normalization floors must be finite and positive")
    if m.family == "gatr":
        if m.gatr_algebra == "lorentz" and (m.gatr_input != "p4" or m.gatr_center):
            raise ValueError("L-GATr requires uncentered four-momenta")
        if m.gatr_algebra != "lorentz" and m.gatr_spurions in {"time", "beam", "beam_plane"}:
            raise ValueError("Time and beam reference channels require the Lorentz algebra")
        if m.gatr_reference_tokens and (not m.gatr_global_token or m.gatr_spurions == "none"):
            raise ValueError("Reference tokens require references and a global readout token")
        if m.gatr_readout == "logits" and (
            t.name != "classification"
            or not m.gatr_global_token
            or m.gatr_algebra != "lorentz"
            or m.global_fusion != "readout"
            or m.head_norm != "none"
            or m.head_layers
            or m.head_widths
        ):
            raise ValueError("GATr logits require Lorentz classification, a global token, no global fusion and no shared head")
        if m.gatr_input == "coordinates" and cfg.preprocessing.normalize_features:
            raise ValueError("GATr coordinates require normalize_features=false and a common gatr_input_scale")
        if m.gatr_global_token and m.pooling != "mean":
            raise ValueError("The GATr global token replaces pooling. Set pooling=mean")
        if m.gatr_algebra != "projective" and (
            m.gatr_projective_improved != defaults.gatr_projective_improved
            or m.gatr_distance_epsilon != defaults.gatr_distance_epsilon
        ):
            raise ValueError("Projective join and distance options require the projective algebra")
        if m.gatr_algebra != "lorentz" and not m.gatr_lorentz_proper:
            raise ValueError("gatr_lorentz_proper requires the Lorentz algebra")
    for option in vars(defaults):
        if option.startswith("gatr_") and m.family != "gatr":
            if getattr(m, option) != getattr(defaults, option):
                raise ValueError(f"{option} is only meaningful for gatr")
        if option.startswith("pelican_") and m.family != "pelican":
            if getattr(m, option) != getattr(defaults, option):
                raise ValueError(f"{option} is only meaningful for pelican")
        if (
            m.family == "pelican"
            and (option.startswith("graph_") or option in {"graph", "neighbors"})
            and getattr(m, option) != getattr(defaults, option)
        ):
            raise ValueError(
                f"{option} belongs to the removed graph model. PELICAN uses complete pair tensors and pelican_* options"
            )
    if (
        m.pelican_message_width < 0
        or m.pelican_triangle_channels < 1
        or m.pelican_triangle_scale <= 0
        or m.pelican_average_nobj <= 0
        or m.pelican_momentum_scale <= 0
        or m.pelican_activation not in {"leaky_relu", "gelu", "silu", "relu", "tanh", "elu"}
    ):
        raise ValueError("Invalid PELICAN configuration")
    if not m.pelican_triangle and (
        m.pelican_triangle_channels != defaults.pelican_triangle_channels
        or m.pelican_triangle_scale != defaults.pelican_triangle_scale
    ):
        raise ValueError("PELICAN triangle channels and scale require pelican_triangle=true")
    if (
        not m.jet_features
        or len(set(m.jet_features)) != len(m.jet_features)
        or not set(m.jet_features) <= set(defaults.jet_features)
    ):
        raise ValueError("jet_features must be a nonempty unique selection of supported features")
    if "eec" in m.jet_features and not m.jet_eec_betas:
        raise ValueError("EEC features require angular powers")
    if (
        any(beta < 0 for beta in m.jet_eec_betas)
        or m.jet_radius <= 0
        or m.jet_hidden < 1
        or m.jet_clamp < 0
        or m.jet_head_mode not in {"per_head", "shared"}
    ):
        raise ValueError("Invalid jet bias parameters")
    if m.pair_bias == "jet" and m.jet_scale_init == 0 and m.jet_block_scale_init == 0:
        raise ValueError("Jet bias scales cannot both be zero at initialization")
    for option in vars(defaults):
        if option.startswith("jet_") and m.pair_bias != "jet" and getattr(m, option) != getattr(defaults, option):
            raise ValueError(f"{option} requires pair_bias=jet")
    if any(k is not None and k < 1 for k in (m.triangle_kj, m.triangle_kk)):
        raise ValueError("Triangle candidate counts must be positive")
    if m.triangle_shared_candidates and (m.triangle_kj or m.triangle_k) != (m.triangle_kk or m.triangle_k):
        raise ValueError("Shared triangle candidates require equal candidate counts")
    if m.graph_gate_hidden < 0 or m.graph_update_scale <= 0:
        raise ValueError("Invalid graph vector gate configuration")
    if m.graph_edge_hidden < 0 or m.graph_activation not in {"gelu", "silu", "relu", "tanh", "elu"}:
        raise ValueError("Invalid graph MLP configuration")
    if m.graph_input_features not in {"invariant3", "mass_energy"} or m.graph_transform not in {"asinh", "none"}:
        raise ValueError("Invalid graph invariant configuration")
    if m.graph_pooling not in {"mean", "sum"}:
        raise ValueError("graph_pooling must be mean or sum")
    if m.norm not in {"layer", "rms"} or m.head_norm not in {"layer", "rms", "none"}:
        raise ValueError("Invalid model norm or head_norm")
    if m.global_fusion not in {"readout", "tokens"}:
        raise ValueError("global_fusion must be readout or tokens")
    if m.parameter_fusion not in {"readout", "tokens"}:
        raise ValueError("parameter_fusion must be readout or tokens")
    if len(set(m.auxiliary_features)) != len(m.auxiliary_features):
        raise ValueError("auxiliary_features must contain unique feature names")
    if m.auxiliary_features and m.family not in {"twistformer", "pelican", "gatr"}:
        raise ValueError("auxiliary_features belongs to twistformer/pelican/gatr")
    if min(m.twistor_geometry_scale, m.twistor_representation_scale) < 0:
        raise ValueError("Twistor update scales must be nonnegative")
    if m.twistor_residual_scaling not in {"none", "depth"}:
        raise ValueError("twistor_residual_scaling must be none or depth")
    if m.family == "pelican" and not m.final_norm:
        raise ValueError("final_norm belongs to transformer/triformer/twistformer")
    if m.pooling_heads is not None and (m.pooling_heads < 1 or m.width % m.pooling_heads):
        raise ValueError("pooling_heads must divide width")
    if m.pooling_queries < 1 or (m.pooling != "pma" and m.pooling_queries != 1):
        raise ValueError("pooling_queries requires pma and must be positive")
    if m.pooling not in {"pma", "cls"} and m.pooling_heads is not None:
        raise ValueError("pooling_heads belongs to pma/cls pooling")
    if m.embedding_norm not in {"none", "layer", "rms"} or m.ffn_norm not in {"none", "layer", "rms"}:
        raise ValueError("Invalid embedding_norm or ffn_norm")
    activations = {"gelu", "relu", "silu", "tanh", "elu", "sine", "sine_adaptive", "glu", "geglu", "swiglu"}
    if isinstance(m.embedding, list):
        if any(width < 1 for width in m.embedding):
            raise ValueError("embedding widths must be positive")
    elif m.embedding not in {"mlp", "two_stage"}:
        raise ValueError("embedding must be a list of hidden widths")
    if m.ffn_activation not in activations:
        raise ValueError("Invalid ffn_activation")
    if m.head_activation not in activations or m.head_hidden_norm not in {"none", "layer", "rms", "batch"}:
        raise ValueError("Invalid head activation or hidden normalization")
    if m.head_init not in {"default", "xavier_normal"} or any(w < 1 for w in m.head_widths):
        raise ValueError("Invalid head initialization or widths")
    if m.head_widths and m.head_layers:
        raise ValueError("Specify head_widths or head_layers, not both")
    if m.head_activation in {"glu", "geglu", "swiglu"} and not (m.head_widths or m.head_layers):
        raise ValueError("Head gating requires hidden layers")
    if m.pooling_depth < 1 or not 0 <= m.pooling_dropout < 1:
        raise ValueError("Invalid pooling depth or dropout")
    if m.pooling != "cls" and m.pooling_depth != 1:
        raise ValueError("pooling_depth belongs to cls pooling")
    if m.pooling not in {"pma", "cls"} and m.pooling_dropout:
        raise ValueError("pooling_dropout belongs to pma/cls pooling")
    if min(m.pair_channels, m.representation_channels) < 0:
        raise ValueError("Twistor channel counts must be nonnegative")
    if m.twistor_precision not in {"float32", "float64"}:
        raise ValueError("twistor_precision must be float32 or float64")
    if m.twistor_transport not in {"phase", "frame"} or m.reference_token not in {
        "none",
        "event",
        "beam",
        "time",
        "beam_plane",
        "frame",
    }:
        raise ValueError("Invalid twistor transport or reference token")
    if m.family != "twistformer" and m.reference_token != "none":
        if m.family != "transformer" or m.reference_token != "frame":
            raise ValueError("reference_token requires twistformer or transformer with frame")
    if len(m.beam_direction) != 3 or sum(v * v for v in m.beam_direction) <= 0:
        raise ValueError("beam_direction must be a nonzero spatial vector")
    if m.reference_token not in {"beam", "beam_plane"} and m.beam_direction != [0.0, 0.0, 1.0]:
        raise ValueError("beam_direction requires reference_token=beam or beam_plane")
    if m.head_layers < 0 or m.head_expansion < 1 or not 0 <= m.head_dropout < 1:
        raise ValueError("Invalid head dimensions or dropout")
    for probability in (m.embedding_dropout, m.residual_dropout):
        if probability is not None and not 0 <= probability < 1:
            raise ValueError("Invalid embedding_dropout or residual_dropout")
    if m.family not in {"transformer", "triformer"}:
        for option in ("embedding", "embedding_norm", "embedding_dropout", "residual_dropout"):
            if getattr(m, option) != getattr(defaults, option):
                raise ValueError(f"{option} belongs to transformer/triformer")
    if m.family not in {"transformer", "triformer", "twistformer"}:
        if m.ffn_activation != "gelu" or m.ffn_norm != "none":
            raise ValueError("ffn_activation and ffn_norm belong to attention encoders")
    if min(m.width, m.depth, m.heads, m.expansion, m.triangle_k, m.query_chunk, m.neighbors) < 1 or m.width % m.heads:
        raise ValueError("Positive model dimensions required. width must divide evenly into heads")
    if (
        not 0 <= m.dropout < 1
        or not 0 <= m.attention_dropout < 1
        or m.pooling not in {"mean", "sum", "weighted_mean", "attention", "attention_query", "pma", "cls"}
    ):
        raise ValueError("Invalid dropout or pooling")
    if m.family not in {"transformer", "triformer", "twistformer", "gatr"} and m.attention_dropout:
        raise ValueError("attention_dropout belongs to attention encoders")
    if m.family == "pelican" and m.pooling != "mean":
        raise ValueError("PELICAN uses its trace and all-pair readout. pooling must be mean")
    if m.twistor_kind not in {"penrose", "momentum"} or m.graph not in {"full", "knn"}:
        raise ValueError("Invalid twistor_kind or graph")
    # Reject explicitly supplied options that would silently do nothing
    specific = {
        "xgboost": ("xgboost_threads", "xgboost_params", "xgboost_sketch_rows", "xgboost_sketch_seed"),
        "twistformer": (
            "geometry_update pair_channels representation_channels twistor_transport twistor_precision "
            "twistor_separate_norm twistor_qk twistor_geometry_scale twistor_representation_scale "
            "twistor_residual_scaling beam_direction twistor_kind twistor_detector three_point"
        ).split(),
        "triformer": (
            "triangle_k triangle_gamma query_chunk triangle_kj triangle_kk triangle_shared_candidates triangle_distinct"
        ).split(),
        "pelican": (
            "graph neighbors graph_event_features graph_pooling graph_gate_hidden graph_update_scale "
            "graph_activation graph_edge_hidden graph_mass_statistics graph_input_features graph_transform"
        ).split(),
    }
    for family, options in specific.items():
        for option in options:
            if m.family != family and getattr(m, option) != getattr(defaults, option):
                raise ValueError(f"{option} is only meaningful for {family}")
    if t.name not in {"classification", "gaussian_regression", "covariant_regression", "flow_matching", "flow_nll"}:
        raise ValueError("Unknown task.name")
    if t.name.startswith("flow_"):
        f = cfg.flow
        if t.name == "flow_matching" and f.type != "continuous":
            raise ValueError("Flow matching requires a continuous flow")
        if t.name == "flow_nll" and (f.space != "direct" or "morph" in f.mode):
            raise ValueError("Flow NLL requires direct coordinates and a known base density")
        if f.lipschitz_weight and f.network != "lzmlp":
            raise ValueError("Lipschitz regularization requires flow.network=lzmlp")
        if f.type == "discrete" and f.network == "unet":
            raise ValueError("Discrete couplings require a vector network")
        if f.condition == "image" and f.network != "unet":
            raise ValueError("Image conditions require flow.network=unet")
        if f.network == "unet" and f.width % 32:
            raise ValueError("U-Net width must be divisible by 32")
    if t.name == "covariant_regression":
        if m.family not in {"pelican", "gatr"}:
            raise ValueError("covariant_regression requires model.family=pelican or gatr")
        if m.family == "gatr" and (
            m.gatr_algebra != "lorentz" or m.gatr_readout != "invariant" or m.gatr_spurions != "none" or m.gatr_global_token
        ):
            raise ValueError("Covariant GATr requires Lorentz token invariants without fixed references")
        if cfg.preprocessing.normalize_targets:
            raise ValueError("covariant_regression requires preprocessing.normalize_targets=false")
        if t.min_std != TaskConfig().min_std:
            raise ValueError("min_std does not apply to deterministic covariant regression")
    if t.mixture_components < 1 or (t.name != "gaussian_regression" and t.mixture_components != 1):
        raise ValueError("Positive mixture_components require Gaussian regression")
    if t.min_std <= 0 or not 0 <= t.label_smoothing < 1 or any(not math.isfinite(w) or w <= 0 for w in t.class_weights):
        raise ValueError("Invalid task parameters")
    if t.name != "classification" and (t.class_weights or t.label_smoothing or t.balance_classes):
        raise ValueError("Class weights/smoothing require classification")
    if min(tr.epochs, tr.batch_size, tr.eval_batch_size, tr.accumulation) < 1:
        raise ValueError("Training counts must be positive")
    if min(tr.workers, tr.patience, tr.save_every, tr.keep_checkpoints, tr.weight_decay, tr.grad_clip) < 0 or tr.learning_rate <= 0:
        raise ValueError("Invalid training values")
    if tr.optimizer not in {"adamw", "adam", "sgd", "lion", "muon_adamw"} or tr.scheduler not in {
        "none",
        "cosine",
        "cosine_restarts",
        "thirds",
        "plateau",
    }:
        raise ValueError("Unknown optimizer/scheduler")
    if tr.max_steps < 0 or tr.scheduler_interval not in {"epoch", "step"}:
        raise ValueError("Invalid optimizer step budget or scheduler interval")
    if tr.scheduler_interval == "step" and (tr.max_steps < 1 or tr.scheduler not in {"cosine", "none"} or tr.warmup_epochs):
        raise ValueError("Step schedules require max_steps, cosine or none, and no epoch warmup")
    if not 0 <= tr.warmup_epochs < tr.epochs:
        raise ValueError("warmup_epochs must be nonnegative and below epochs")
    if not 0 <= tr.min_learning_rate <= tr.learning_rate:
        raise ValueError("min_learning_rate must lie between zero and learning_rate")
    if tr.scheduler == "cosine_restarts":
        if (
            not isinstance(tr.restart_period, int)
            or not isinstance(tr.restart_mult, int)
            or tr.restart_period < 1
            or tr.restart_mult < 1
        ):
            raise ValueError("restart_period and restart_mult must be positive integers")
        if not isinstance(tr.decay_epochs, int) or not 0 <= tr.decay_epochs < tr.epochs - tr.warmup_epochs:
            raise ValueError("decay_epochs must leave at least one cosine epoch")
        if not 0 < tr.decay_gamma <= 1:
            raise ValueError("decay_gamma must be in (0,1]")
    if not 0 < tr.plateau_factor < 1 or min(tr.plateau_patience, tr.plateau_threshold, tr.plateau_cooldown) < 0:
        raise ValueError("Invalid plateau scheduler parameters")
    if not 0 <= tr.ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    if len(tr.optimizer_betas) != 2 or any(not 0 <= beta < 1 for beta in tr.optimizer_betas) or tr.optimizer_eps <= 0:
        raise ValueError("Optimizer requires two betas in [0,1) and positive epsilon")
    if not 0 <= tr.token_dropout < 1 or min(tr.token_min_keep, tr.prefetch_factor, tr.eval_prefetch_factor) < 1:
        raise ValueError("Invalid token dropout or loader prefetch settings")
    if isinstance(tr.profile_batches, bool) or not isinstance(tr.profile_batches, int) or tr.profile_batches < 0:
        raise ValueError("training.profile_batches must be a nonnegative integer")
    if tr.augmentation not in {"dataset", "online", "none"}:
        raise ValueError("training.augmentation must be dataset, online or none")
    if tr.amp_dtype not in {"auto", "bf16", "fp16"}:
        raise ValueError("amp_dtype must be auto, bf16 or fp16")
    if tr.strategy not in {"single", "ddp"}:
        raise ValueError("training.strategy must be single or ddp")
    if tr.ddp_bucket_cap_mb <= 0:
        raise ValueError("ddp_bucket_cap_mb must be positive")
    if tr.fused_optimizer and tr.optimizer not in {"adam", "adamw", "muon_adamw"}:
        raise ValueError("fused_optimizer requires adam, adamw or muon_adamw")
    if tr.compile_mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise ValueError("Unknown compile_mode")
    if not cfg.evaluation.efficiencies or any(not 0 < value <= 1 for value in cfg.evaluation.efficiencies):
        raise ValueError("Evaluation efficiencies must be nonempty and in (0,1]")
    allowed = {"val_loss", "val_accuracy" if t.name == "classification" else "val_rmse"}
    if t.name == "classification":
        allowed.add("val_auc")
    if t.name.startswith("flow_"):
        allowed = {"val_loss", "val_flow_loss" if t.name == "flow_matching" else "val_nll"}
    if tr.monitor not in allowed or tr.mode not in {"min", "max"}:
        raise ValueError(f"monitor must be one of {sorted(allowed)}. mode must be min or max")
    if (tr.monitor in {"val_accuracy", "val_auc"}) != (tr.mode == "max"):
        raise ValueError("Use mode=max for accuracy or AUC and mode=min for loss/RMSE")
    return cfg

def _merge_yaml(base, updates):
    """
    Merge steering mappings with local values taking priority

    Args:
        base:  Included steering mapping
        updates:  Local steering mapping
    Returns:
        merged: New mapping with lists and scalar values replaced
    """
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_yaml(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

def load_yaml_mapping(path, *, _parents=()):
    """
    Resolve YAML includes relative to each steering file

    Args:
        path:  YAML file with an optional root include path or list of paths
        _parents:  Resolved parent paths for include cycle detection
    Returns:
        data: Merged mapping without include directives
    """
    path = Path(path).resolve()
    if path in _parents:
        chain = " -> ".join(str(item) for item in (*_parents, path))
        raise ValueError(f"YAML include cycle: {chain}")
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML configuration must be a mapping: {path}")
    includes = data.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or any(not isinstance(item, str) or not item.strip() for item in includes):
        raise ValueError(f"YAML include must be a path or list of paths: {path}")
    merged = {}
    for item in includes:
        included = load_yaml_mapping(path.parent / item, _parents=(*_parents, path))
        merged = _merge_yaml(merged, included)
    return _merge_yaml(merged, data)

def load_config(path, overrides=()):
    """
    Load YAML steering and apply dotted CLI overrides

    Args:
        path:  YAML configuration file
        overrides:  Sequence of section.option=value strings
    Returns:
        config: Validated Config instance
    """
    data = load_yaml_mapping(path)
    for item in overrides:
        key, sep, value = item.partition("=")
        parts = key.split(".")
        if not sep or any(not part.strip() for part in parts):
            raise ValueError("Overrides must have the form section.option=value")
        cursor = data
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot override {key}: {part} must be a mapping")
        cursor[parts[-1]] = yaml.safe_load(value)
    return parse_config(data)

def save_config(config, path):
    """
    Write the resolved configuration as YAML

    Args:
        config:  Validated Config instance
        path:  Output YAML file
    Returns:
        None: Configuration is written to disk
    """
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
