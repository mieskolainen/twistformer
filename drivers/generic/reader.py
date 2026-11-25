# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import torch
from torch.utils.data import Dataset
from twistformer.api.contracts import Sample

class ArrayDataset(Dataset):
    def __init__(self, path, spec, task, *, allow_missing_parameters=False):
        with np.load(path, allow_pickle=False) as f:
            self.arrays = {k: f[k] for k in f.files}
        a = self.arrays
        if "features" not in a or a["features"].ndim != 3 or a["features"].shape[-1] != len(spec.features):
            raise ValueError("NPZ requires features [events,tokens,features] matching InputSpec")
        self.task = task
        self.path = str(path)
        b, n, _ = a["features"].shape
        if spec.parameters:
            if "parameters" not in a:
                if not allow_missing_parameters:
                    raise ValueError("NPZ requires parameters [events,parameters] matching InputSpec")
            elif a["parameters"].shape != (b, len(spec.parameters)) or not np.isfinite(a["parameters"]).all():
                raise ValueError("NPZ parameters must be finite [events,parameters] matching InputSpec")
        elif "parameters" in a:
            raise ValueError("Declare parameter names in the driver configuration")
        if spec.global_features:
            if "global_features" not in a or a["global_features"].shape != (b, len(spec.global_features)):
                raise ValueError("NPZ requires global_features [events,globals] matching InputSpec")
            if not np.isfinite(a["global_features"]).all():
                raise ValueError("Global features must be finite")
        elif "global_features" in a:
            raise ValueError("Declare global feature names in the driver configuration")
        a.setdefault("mask", np.ones((b, n), dtype=bool))
        if a["mask"].shape != (b, n) or a["mask"].dtype != np.bool_:
            raise ValueError("mask must be boolean [events,tokens]")
        if spec.geometry == "p4" and "p4" not in a:
            raise ValueError("p4 geometry requires p4 [events,tokens,4]")
        if "p4" in a:
            momenta = a["p4"]
            if momenta.shape != (b, n, 4) or momenta.dtype.kind not in "fiu":
                raise ValueError("p4 requires real four-momenta [events,tokens,4]")
            a["p4"] = momenta.astype(np.float64 if momenta.dtype.itemsize >= 8 else np.float32, copy=False)
        for key in ("targets", "weights", "ids", "target_mask", "token_weights", "source", "flow_condition"):
            if key in a and len(a[key]) != b:
                raise ValueError(f"{key} event count mismatch")

    def __len__(self):
        return len(self.arrays["features"])

    def __getitem__(self, i):
        a, m = self.arrays, self.arrays["mask"][i]
        y = None
        if "targets" in a:
            y = torch.as_tensor(a["targets"][i])
            if self.task.name == "classification":
                if y.numel() != 1 or y.is_floating_point() and y.item() != int(y.item()):
                    raise ValueError("Classification targets must be integer class indices")
                y = y.long().reshape(())
            elif self.task.name.startswith("flow_"):
                y = y.float().reshape(y.shape or (1,))
            else:
                y = y.float().reshape(-1)
        mask = None
        if "target_mask" in a:
            mask = torch.as_tensor(a["target_mask"][i]).bool()
            mask = mask.reshape(mask.shape or (1,)) if self.task.name.startswith("flow_") else mask.reshape(-1)
        return Sample(
            torch.from_numpy(a["features"][i][m].copy()).float(),
            y,
            torch.from_numpy(a["p4"][i][m]) if "p4" in a else None,
            float(a["weights"][i]) if "weights" in a else 1.0,
            str(a["ids"][i]) if "ids" in a else f"{self.path}:{i}",
            mask,
            torch.as_tensor(a["token_weights"][i][m]).float() if "token_weights" in a else None,
            torch.as_tensor(a["global_features"][i].copy()).float() if "global_features" in a else None,
            parameters=torch.as_tensor(a["parameters"][i].copy()).float() if "parameters" in a else None,
            source=torch.as_tensor(a["source"][i].copy()).float() if "source" in a else None,
            flow_condition=torch.as_tensor(a["flow_condition"][i].copy()).float() if "flow_condition" in a else None,
        )
