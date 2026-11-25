# Saved-model ONNX inference
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.backends import opt_einsum

from .contracts import Batch, Prediction
from .predictor import load_predictor
from twistformer.data.preprocessing import Preprocessor
from twistformer.randomness import torch_generator

class InferenceGraph(nn.Module):
    def __init__(self, predictor):
        """
        Wrap checkpoint inference with fitted input and target normalization

        Args:
            predictor:  Restored CPU predictor
        """
        super().__init__()
        self.model = predictor.model
        self.preprocessor = deepcopy(predictor.preprocessor)
        if type(self.preprocessor) is not Preprocessor:
            raise TypeError("ONNX export requires the standard tensor preprocessor")
        for name, value in self.preprocessor.state.items():
            self.register_buffer(name, value)
        self.preprocessor.state = dict(self.named_buffers(recurse=False))
        self.output_names = []

    def forward(self, features, mask, p4=None, global_features=None, token_weights=None, parameters=None):
        """
        Predict from raw inputs and return named tensors in physical units

        Args:
            features:  Raw particle features (B,N,F), float32
            mask:  Valid particles (B,N), bool
            p4:  Optional physical four-momenta (B,N,4), float64
            global_features:  Optional raw event features (B,G), float32
            token_weights:  Optional pooling weights (B,N), float32
            parameters:  Optional raw hypothesis values (B,P), float32
        Returns:
            outputs: Active Prediction tensors in field order
        """
        features, global_features = self.preprocessor.transform_inputs(features, global_features)
        batch = Batch(
            features,
            mask,
            features.new_ones(features.shape[0]),
            (),
            p4=p4,
            global_features=global_features,
            token_weights=token_weights,
            parameters=self.preprocessor.transform_parameters(parameters),
        )
        prediction = self.preprocessor.inverse_prediction(self.model(batch))
        return tuple(getattr(prediction, name) for name in self.output_names)

def example_inputs(predictor, max_tokens, batch_size=2):
    """
    Construct finite masked inputs without reading driver data

    Args:
        predictor:  Restored CPU predictor
        max_tokens:  Fixed padded particle count
        batch_size:  Number of independent events
    Returns:
        inputs: Raw particle features, mask and optional input tensors
    """
    spec = predictor.experiment.driver.input_spec
    state = predictor.preprocessor.state
    generator = torch_generator(1729 + batch_size)
    features = (
        torch.randn(batch_size, max_tokens, len(spec.features), generator=generator) * state["feature_std"] + state["feature_mean"]
    )
    mask = torch.arange(max_tokens)[None] < (max_tokens - torch.arange(batch_size) % max_tokens)[:, None]
    p4 = None
    if predictor.model.requires_p4:
        spatial = torch.randn(batch_size, max_tokens, 3, generator=generator).double()
        energy = (spatial.square().sum(-1, keepdim=True) + 1).sqrt()
        p4 = torch.cat((energy, spatial), -1)
    globals_ = None
    if spec.global_features:
        globals_ = (
            torch.randn(batch_size, len(spec.global_features), generator=generator) * state["global_std"] + state["global_mean"]
        )
    weights = (
        torch.rand(batch_size, max_tokens, generator=generator) + 0.1
        if predictor.experiment.config.model.pooling == "weighted_mean"
        else None
    )
    parameters = None
    if spec.parameters:
        parameters = (
            torch.randn(batch_size, len(spec.parameters), generator=generator) * state["parameter_std"] + state["parameter_mean"]
        )
    return features, mask, p4, globals_, weights, parameters

def verify_onnx(path, reference, predictor, max_tokens):
    """
    Compare exported predictions across event counts and particle masks

    Args:
        path:  Staged ONNX file
        reference:  Original inference module
        predictor:  Restored predictor with input schema
        max_tokens:  Fixed padded particle count
    Returns:
        None: Runtime predictions agree with checkpoint inference
    """
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    names = {item.name for item in session.get_inputs()}
    for size in (1, 3):
        inputs = example_inputs(predictor, max_tokens, size)
        feed = {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs) if name in names}
        actual = session.run(reference.output_names, feed)
        expected = reference(*inputs)
        for name, value, target in zip(reference.output_names, actual, expected):
            if not np.isfinite(value).all() or not torch.isfinite(target).all():
                raise ValueError(f"Nonfinite ONNX verification output: {name}")
            np.testing.assert_allclose(value, target.numpy(), rtol=3e-4, atol=3e-5, err_msg=name)

INPUT_NAMES = ("features", "mask", "p4", "global_features", "token_weights", "parameters")

@torch.no_grad()
# Contraction planning must not fix symbolic event counts
@opt_einsum.flags(enabled=False)
def export_onnx(checkpoint, output, *, max_tokens=128, opset=18, verify=True):
    """
    Export saved inference weights through the model-owned ONNX preparation

    Args:
        checkpoint:  Versioned checkpoint path
        output:  Destination ONNX path
        max_tokens:  Fixed padded particle count, at least one
        opset:  ONNX operator set, at least 18
        verify:  Check numerical agreement with ONNX Runtime before saving
    Returns:
        path: Completed self-contained ONNX file
    """
    if max_tokens < 1 or opset < 18:
        raise ValueError("ONNX export requires max_tokens >= 1 and opset >= 18")
    output = Path(output)
    if output.resolve() == Path(checkpoint).resolve():
        raise ValueError("ONNX output must differ from the checkpoint")
    try:
        import onnx
        import onnxscript

        if verify:
            import onnxruntime
    except ImportError as error:
        raise ImportError("Install ONNX support with pip install 'twistformer[onnx]'") from error
    predictor = load_predictor(checkpoint, "cpu")
    if predictor.experiment.config.model.family == "xgboost":
        raise ValueError("XGBoost checkpoints use native tree inference outside the Torch ONNX exporter")
    if predictor.experiment.task.spec.name.startswith("flow_"):
        from twistformer.flows.export import export_flow_onnx

        return export_flow_onnx(predictor, output, max_tokens=max_tokens, opset=opset, verify=verify)
    reference = InferenceGraph(predictor).eval()
    inputs = example_inputs(predictor, max_tokens)
    features, mask, p4, globals_, weights, parameters = inputs
    normalized, normalized_globals = predictor.preprocessor.transform_inputs(features, globals_)
    prediction = predictor.model(
        Batch(
            normalized,
            mask,
            features.new_ones(features.shape[0]),
            (),
            p4=p4,
            global_features=normalized_globals,
            token_weights=weights,
            parameters=predictor.preprocessor.transform_parameters(parameters),
        )
    )
    reference.output_names = [field.name for field in fields(Prediction) if getattr(prediction, field.name) is not None]
    module = deepcopy(reference)
    module.model.pool.validate_weights = False
    module = module.model.backbone.prepare_onnx(module, inputs).eval()
    for actual, expected in zip(module(*inputs), reference(*inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    batch = torch.export.Dim("batch", min=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".onnx-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model.onnx"
        torch.onnx.export(
            module,
            inputs,
            str(staged),
            dynamo=True,
            external_data=False,
            # ONNX Runtime optimizes the verified inference graph
            optimize=False,
            opset_version=opset,
            input_names=[name for name, value in zip(INPUT_NAMES, inputs) if value is not None],
            output_names=reference.output_names,
            dynamic_shapes=tuple({0: batch} if value is not None else None for value in inputs),
        )
        graph = onnx.load(staged)
        for item in graph.graph.input:
            if not item.type.tensor_type.shape.dim[0].dim_param:
                raise ValueError(f"ONNX export fixed the batch axis of {item.name}")
        onnx.helper.set_model_props(
            graph,
            {
                "input_spec": json.dumps(asdict(predictor.experiment.driver.input_spec)),
                "task_spec": json.dumps(asdict(predictor.experiment.driver.task_spec)),
                "max_tokens": str(max_tokens),
                "normalization": "raw features in, physical targets out",
                "verification": "onnxruntime" if verify else "unchecked",
            },
        )
        onnx.checker.check_model(graph)
        onnx.save(graph, staged)
        if verify:
            verify_onnx(staged, reference, predictor, max_tokens)
        staged.replace(output)
    return output
