# Explicit Supraflow vector-field checkpoint migration
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict
from pathlib import Path
import hashlib
import torch
from twistformer.api.experiment import build_experiment
from twistformer.training.checkpoint import VERSION


def import_checkpoint(checkpoint, output, config, preprocessing, reference):
    """
    Import a direct legacy vector field after numerical parity checks

    Args:
        checkpoint:  Original Supraflow state-dict checkpoint
        output:  New Twistformer checkpoint file
        config:  Explicit matching architecture and dataset configuration
        preprocessing:  Verified original normalization as FlowPreprocessor state
        reference:  Original normalized x, t, optional theta and velocity tensors
    Returns:
        file: Portable inference checkpoint with source provenance
    """
    f = config.flow
    if f.type != "continuous" or f.space != "direct" or f.network == "unet" or f.dct:
        raise ValueError("Legacy migration supports direct vector fields without the old nonorthogonal DCT")
    if f.condition != "parameters" or f.mode not in {"unconditional", "conditional"}:
        raise ValueError("Legacy migration requires explicit vector conditions or unconditional generation")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Migration output exists. Use a new checkpoint file")
    legacy = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state" not in legacy:
        raise ValueError("Expected a Supraflow model_state checkpoint")
    experiment = build_experiment(config)
    model = experiment.model.eval()
    preprocessor = experiment.driver.create_preprocessor(preprocessing, config=config.preprocessing)
    shape = tuple(preprocessing.get("flow_shape", torch.tensor([])).tolist())
    if shape != model.shape or not preprocessing.get("target_std", torch.tensor([])).numel():
        raise ValueError("Migration requires original target shape, transforms and normalization")
    dimensions = {"feature": len(experiment.driver.input_spec.features), "target": len(experiment.driver.task_spec.outputs),
                  "parameter": len(experiment.driver.input_spec.parameters), "global": len(experiment.driver.input_spec.global_features)}
    for name, dimension in dimensions.items():
        if not dimension:
            continue
        for suffix in ("mean", "std"):
            value = preprocessing.get(f"{name}_{suffix}")
            if not isinstance(value, torch.Tensor) or value.shape != (dimension,) or not torch.isfinite(value).all():
                raise ValueError(f"Missing finite original {name}_{suffix}")
            if suffix == "std" and (value <= 0).any():
                raise ValueError("Original normalization scales must be positive")
    weights = {}
    for name, value in legacy["model_state"].items():
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name == "vnet.freqs":
            mapped = "flow.vector_field.frequencies"
        elif name.startswith("vnet.mlp."):
            mapped = "flow.vector_field.network." + name[len("vnet.mlp."):]
        else:
            raise ValueError(f"Unsupported legacy parameter {name}")
        if mapped in weights:
            raise ValueError("Duplicate legacy parameters after wrapper removal")
        weights[mapped] = value
    expected = model.state_dict()
    weights["flow.base.reference"] = expected["flow.base.reference"]
    frequency = "flow.vector_field.frequencies"
    if not len(expected[frequency]):
        weights[frequency] = expected[frequency]
    if set(weights) != set(expected):
        raise ValueError(f"Legacy architecture mismatch: missing {sorted(set(expected) - set(weights))}, "
                         f"extra {sorted(set(weights) - set(expected))}")
    for name, value in weights.items():
        if name.endswith(".c") and value.numel() == expected[name].numel() == 1:
            weights[name] = value.reshape_as(expected[name])
        if weights[name].shape != expected[name].shape or not torch.isfinite(weights[name]).all():
            raise ValueError(f"Invalid legacy tensor {name}")
    model.load_state_dict(weights, strict=True)
    x, time, velocity = (reference[key] for key in ("x", "t", "velocity"))
    if len(x) < 4 or torch.unique(time).numel() < 2 or x.shape != velocity.shape:
        raise ValueError("Migration requires at least four reference events at multiple times")
    with torch.no_grad():
        actual = model.flow.vector_field(x, time, reference.get("theta"))
    torch.testing.assert_close(actual, velocity, rtol=2e-5, atol=2e-6)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": VERSION, "config": config.to_dict(), "input_spec": asdict(experiment.driver.input_spec),
             "task_spec": asdict(experiment.driver.task_spec), "model": model.state_dict(), "ema": None,
             "inference_weights": "model", "preprocessing": preprocessor.state, "inference_only": True,
             "migration": {"source": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                           "reference_events": len(x), "verified": "vector_field", "optimizer_restored": False,
                           "density_solver": "corrected_twistformer"}}
    staged = output.with_name(output.name + ".staged")
    if staged.exists():
        raise FileExistsError("A staged migration checkpoint exists")
    torch.save(state, staged)
    staged.rename(output)
    return output
