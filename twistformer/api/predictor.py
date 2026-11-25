# Public experiment API
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import fields
import torch
from twistformer.api.config import parse_config
from twistformer.api.experiment import build_experiment
from twistformer.api.contracts import Prediction
from twistformer.data.preprocessing import PreparedDataset
from twistformer.training.checkpoint import read_checkpoint, restore_model
from twistformer.training.distributed import DistributedRuntime
from twistformer.training.engine import device_for, validate_batch, configure_precision, mixed_precision_dtype, make_loader

class Predictor:
    def __init__(self, experiment, preprocessor, device):
        self.experiment, self.preprocessor = experiment, preprocessor
        if experiment.config.model.family == "xgboost":
            self.device = experiment.model.resolve_device(device)
            if hasattr(experiment.driver, "descriptor_device"):
                experiment.driver.descriptor_device = self.device
        else:
            self.device = device_for(device)
        if self.device.type == "cuda":
            configure_precision(experiment.config.training)
        self.model = experiment.model.to(self.device).eval()

    def condition(self, samples, parameters=None):
        """
        Set explicit hypotheses in the checkpoint parameter order

        Args:
            samples:  Raw event dataset
            parameters:  Named scalars or event arrays in physical units, or None to use stored values
        Returns:
            samples: Dataset with the requested hypotheses and unchanged event inputs
        """
        from twistformer.data.parameters import ParameterDataset

        if parameters is None:
            return samples
        return ParameterDataset(samples, self.experiment.driver.input_spec.parameters, parameters)

    @torch.inference_mode()
    def forward(self, batch):
        """
        Run evaluation-mode inference with CUDA mixed precision when configured

        Args:
            batch:  Validated batch on the predictor device
        Returns:
            prediction: Model outputs in training coordinates
        """
        self.model.eval()
        enabled = self.experiment.config.training.amp and self.device.type == "cuda"
        dtype = mixed_precision_dtype(self.device, self.experiment.config.training.amp_dtype)
        with torch.autocast(self.device.type, dtype=dtype, enabled=enabled):
            prediction = self.model(batch)
        return Prediction(
            **{
                key: getattr(prediction, key).float()
                for key in (field.name for field in fields(Prediction))
                if getattr(prediction, key) is not None
            }
        )

    def loader(self, samples, batch_size=128):
        """
        Prepare raw events with checkpoint normalization

        Args:
            samples:  Dataset or sequence of raw Sample objects
            batch_size:  Maximum events per batch
        Returns:
            loader: Ordered DataLoader with padded batches
        """
        return make_loader(
            PreparedDataset(samples, self.preprocessor),
            self.experiment.config.training,
            runtime=DistributedRuntime(self.device),
            batch_size=batch_size,
            allow_negative_weights=True,
        )

    @torch.inference_mode()
    def predict(self, samples, batch_size=128, *, parameters=None):
        """
        Predict on raw events and restore physical target units

        Args:
            samples:  Dataset or sequence of raw Sample objects
            batch_size:  Maximum events per batch
            parameters:  Optional named hypothesis scalars or one value per event
        Returns:
            prediction: CPU raw logits with a probabilities property, or regression parameters
        """
        parts = {}
        loader = self.loader(self.condition(samples, parameters), batch_size)
        for batch in loader:
            batch = batch.to(self.device, non_blocking=loader.pin_memory)
            validate_batch(batch, self.model.requires_p4)
            prediction = self.preprocessor.inverse_prediction(self.forward(batch))
            values = {
                field.name: value.cpu()
                for field in fields(prediction)
                if field.name not in {"flow_loss", "flow_weights"} and (value := getattr(prediction, field.name)) is not None
            }
            if parts and parts.keys() != values.keys():
                raise ValueError("Prediction fields must agree across batches")
            for name, value in values.items():
                parts.setdefault(name, []).append(value)
        if not parts:
            raise ValueError("No samples to predict")
        return Prediction(**{name: torch.cat(values) for name, values in parts.items()})

    def predict_prob(self, samples, batch_size=128, *, parameters=None):
        """
        Apply sigmoid or softmax to classifier predictions

        Args:
            samples:  Dataset or sequence of raw Sample objects
            batch_size:  Maximum events per batch
            parameters:  Optional named hypothesis scalars or one value per event
        Returns:
            probabilities: CPU class probabilities (B,C), binary columns are (1-p,p)
        """
        if self.experiment.task.spec.name != "classification":
            raise TypeError("predict_prob requires a classification task")
        return self.predict(samples, batch_size, parameters=parameters).probabilities

def load_predictor(path, device="auto"):
    """
    Reconstruct inference from a versioned checkpoint

    Args:
        path:  Checkpoint file
        device:  Torch device or auto
    Returns:
        predictor: Model with restored driver and preprocessing
    """
    state = read_checkpoint(path)
    config = parse_config(state["config"])
    experiment = build_experiment(config)
    restore_model(experiment, state, inference=True)
    preprocessor = experiment.driver.create_preprocessor(state["preprocessing"], config=config.preprocessing)
    if config.task.name.startswith("flow_"):
        from twistformer.flows.predictor import FlowPredictor

        return FlowPredictor(experiment, preprocessor, device)
    predictor = Predictor(experiment, preprocessor, device)
    if state.get("calibration"):
        from twistformer.evaluation.temperature import CalibratedClassifier

        classifier = CalibratedClassifier(predictor.model, state["calibration"]).to(predictor.device).eval()
        predictor.model = experiment.model = classifier
    return predictor
