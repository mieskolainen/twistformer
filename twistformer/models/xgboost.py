# Boosted decision tree inference for shower descriptors
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import torch
from torch import nn
import xgboost as xgb

from twistformer.api.contracts import Prediction


class ShowerBooster(nn.Module):
    requires_p4 = False

    def __init__(self, threads):
        """
        Set the CPU thread budget for tree inference

        Args:
            threads:  Number of XGBoost worker threads
        """
        super().__init__()
        self.threads = threads
        self.booster = None
        self.inference_device = None

    @staticmethod
    def resolve_device(name):
        """
        Select hardware supported by both Torch and XGBoost

        Args:
            name:  cpu, cuda, a CUDA ordinal or auto
        Returns:
            device: Resolved Torch and XGBoost device
        """
        name = str(name)
        if name == "cpu":
            return torch.device("cpu")
        cuda = torch.cuda.is_available() and xgb.build_info().get("USE_CUDA", False)
        if name == "auto":
            return torch.device("cuda" if cuda else "cpu")
        device = torch.device(name)
        if device.type != "cuda":
            raise ValueError("XGBoost device must be cpu, cuda or auto")
        if not cuda or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise RuntimeError("XGBoost CUDA requires a visible GPU and a CUDA-enabled XGBoost build")
        return device

    @staticmethod
    def inputs(batch):
        """
        Collect one descriptor row per shower in checkpoint order

        Args:
            batch:  Prepared event batch
        Returns:
            result: Shower descriptors and configured event scalars
        """
        if batch.features.shape[1] != 1:
            raise ValueError("XGBoost needs one descriptor row per event")
        result = batch.features[:, 0]
        if batch.global_features is not None:
            result = torch.cat((result, batch.global_features), dim=1)
        return result

    def forward(self, batch):
        """
        Predict the raw class-one margin of each shower

        Args:
            batch:  Prepared shower descriptors
        Returns:
            prediction: Binary logits on the batch device
        """
        if self.booster is None:
            raise RuntimeError("Fit or restore the shower booster before prediction")
        device = batch.features.device
        if self.inference_device != device:
            self.booster.set_param({"device": str(device), "nthread": self.threads})
            self.inference_device = device
        features = self.inputs(batch).detach().cpu().numpy()
        if device.type == "cuda":
            matrix = xgb.DMatrix(features, feature_names=self.booster.feature_names, nthread=self.threads)
            margins = self.booster.predict(matrix, output_margin=True)
        else:
            margins = self.booster.inplace_predict(features, predict_type="margin")
        return Prediction(logits=torch.from_numpy(margins[:, None]).to(device))

    def get_extra_state(self):
        """
        Encode native trees in a portable tensor

        Returns:
            state: UBJSON model bytes or None for an unfitted model
        """
        if self.booster is None:
            return None
        return torch.from_numpy(np.frombuffer(self.booster.save_raw(raw_format="ubj"), dtype=np.uint8).copy())

    def set_extra_state(self, state):
        """
        Restore native trees for device selection at inference

        Args:
            state:  UBJSON model tensor from the checkpoint
        """
        self.booster = None
        self.inference_device = None
        if state is not None:
            self.booster = xgb.Booster(model_file=bytearray(state.cpu().numpy()))
