# Deterministic flow sampling export
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from twistformer.api.contracts import Batch
from twistformer.api.export import example_inputs, INPUT_NAMES
from twistformer.randomness import torch_generator
from .discrete import InvertibleLinear
from .transforms import transform_chain


class FrozenLinear(nn.Module):
    def __init__(self, layer):
        """
        Freeze an LU map for portable sampling inference

        Args:
            layer:  Trained invertible linear layer
        """
        super().__init__()
        matrix = layer.permutation @ (layer.lower.tril(-1) + layer.identity) @ (
            layer.upper.triu(1) + torch.diag(layer.sign * layer.log_diagonal.exp()))
        self.register_buffer("matrix", matrix.detach())
        self.register_buffer("inverse_matrix", torch.linalg.inv(matrix).detach())
        self.register_buffer("log_determinant", layer.log_diagonal.sum().detach())

    def forward(self, value, condition=None, *, inverse=False):
        """
        Apply a frozen map or its inverse

        Args:
            value:  Coordinate vectors
            condition:  Unused conditions
            inverse:  Apply the inverse map
        Returns:
            result: Coordinates and log determinant
        """
        return value @ (self.inverse_matrix if inverse else self.matrix), (-1 if inverse else 1) * self.log_determinant


class SamplingGraph(nn.Module):
    def __init__(self, predictor):
        """
        Wrap deterministic sampling with saved physical preprocessing

        Args:
            predictor:  Restored CPU flow predictor
        """
        super().__init__()
        self.model = deepcopy(predictor.model).eval()
        self.state = {}
        for name, value in predictor.preprocessor.state.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(name, value.clone())
                self.state[name] = getattr(self, name)
        self.transforms = predictor.preprocessor.transforms
        if self.model.config.condition in {"image", "graph"} and self.model.conditional:
            raise ValueError("Flow ONNX export currently supports vector and event-encoder conditions")
        if self.model.conditioner is not None:
            self.model.conditioner.pool.validate_weights = False
        if self.model.config.type == "discrete":
            for index, layer in enumerate(self.model.flow.layers):
                if isinstance(layer, InvertibleLinear):
                    self.model.flow.layers[index] = FrozenLinear(layer)

    def forward(self, base_coordinates, features, mask, p4=None, global_features=None, token_weights=None, parameters=None):
        """
        Map explicit base noise or morphing sources to physical events

        Args:
            base_coordinates:  Base noise, or physical sources for morphing
            features:  Raw particle features
            mask:  Valid particle mask
            p4:  Optional physical four-momenta
            global_features:  Optional event features
            token_weights:  Optional pooling weights
            parameters:  Optional physical hypotheses
        Returns:
            output: Generated physical coordinates
        """
        features = (features - self.feature_mean) / self.feature_std
        if global_features is not None:
            global_features = (global_features - self.global_mean) / self.global_std
        if parameters is not None:
            parameters = (parameters - self.parameter_mean) / self.parameter_std
        batch = Batch(features, mask, features.new_ones(features.shape[0]), (), p4=p4,
                      global_features=global_features, token_weights=token_weights, parameters=parameters)
        value = base_coordinates
        shape = self.model.shape
        if "morph" in self.model.config.mode:
            value = transform_chain(value, self.transforms)[0]
            value = (value - self.target_mean.reshape(shape)) / self.target_std.reshape(shape)
            if self.model.codec is not None:
                value = self.model.codec.encode(value)[0]
        value = self.model.flow.transport(value, self.model.condition(batch))
        if self.model.codec is not None:
            value = self.model.codec.decode(value)
        value = value * self.target_std.reshape(shape) + self.target_mean.reshape(shape)
        return transform_chain(value, self.transforms, inverse=True)[0]


def export_flow_onnx(predictor, output, *, max_tokens=128, opset=18, verify=True):
    """
    Export deterministic fixed-batch flow sampling and verify its output

    Args:
        predictor:  Restored CPU flow predictor
        output:  New ONNX file
        max_tokens:  Fixed particle padding count
        opset:  ONNX operator set
        verify:  Compare independent noise inputs with ONNX Runtime
    Returns:
        file: Verified self-contained ONNX file
    """
    import onnx

    if predictor.model.config.mode == "conditional-flip":
        raise ValueError("Export the conditional model and apply the physical class intervention before inference")
    if predictor.model.config.lower:
        raise ValueError("Rejection constraints are a sampling operation and cannot be exported as a deterministic map")
    graph = SamplingGraph(predictor)
    inputs = example_inputs(predictor, max_tokens, batch_size=1)
    shape = predictor.model.shape if "morph" in predictor.model.config.mode else predictor.model.flow.base.shape
    generator = torch_generator(1729)
    base = torch.randn(1, *shape, generator=generator)
    if "morph" in predictor.model.config.mode:
        base = predictor.preprocessor.inverse_target(base)
    arguments = (base, *inputs)
    names = ("base_coordinates", *INPUT_NAMES)
    reference = deepcopy(graph)
    if graph.model.conditioner is not None:
        graph = graph.model.conditioner.backbone.prepare_onnx(graph, arguments)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Flow export output exists. Use a new file")
    staged = output.with_name(output.name + ".staged")
    if staged.exists():
        raise FileExistsError("A staged flow export already exists")
    torch.onnx.export(graph, arguments, str(staged), dynamo=False, external_data=False, opset_version=opset,
                      input_names=[name for name, value in zip(names, arguments) if value is not None], output_names=["samples"])
    model = onnx.load(staged)
    onnx.helper.set_model_props(model, {"task_spec": json.dumps(asdict(predictor.experiment.driver.task_spec)),
                                       "batch_size": "1", "max_tokens": str(max_tokens),
                                       "operation": "deterministic_sampling", "base_input": "explicit_coordinates",
                                       "verification": "onnxruntime" if verify else "unchecked"})
    onnx.checker.check_model(model)
    onnx.save(model, staged)
    if verify:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(staged), options, providers=["CPUExecutionProvider"])
        required = {item.name for item in session.get_inputs()}
        for scale in (0.25, 1.0, 4.0):
            values = (base * scale, *inputs)
            feed = {name: value.detach().numpy() for name, value in zip(names, values) if name in required}
            actual = session.run(None, feed)[0]
            expected = reference(*values).detach().numpy()
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-5)
    staged.rename(output)
    return output
