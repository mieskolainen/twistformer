# Tensor inference for fitted numerical gradient-boosted decision trees
#
# mikael.mieskolainen@cern.ch (2026)

import json

import torch


def export_forest(booster, output):
    """
    Convert fitted numerical trees into flat node tables

    Args:
        booster:  Trained XGBoost booster with numerical splits
        output:   Destination PyTorch checkpoint file
    """
    learner = json.loads(booster.save_raw(raw_format="json"))["learner"]
    ensemble = learner["gradient_booster"]["model"]
    objective = learner["objective"]["name"]
    if objective not in ("binary:logistic", "multi:softprob", "reg:squarederror"):
        raise ValueError(f"Unsupported tree objective {objective}")
    intercept = torch.tensor(json.loads(learner["learner_model_param"]["base_score"])).reshape(-1).float()
    if objective == "multi:softprob" and intercept.numel() == 1:
        intercept = intercept.expand(int(learner["learner_model_param"]["num_class"])).clone()
    if objective == "binary:logistic":
        intercept = -(1.0 / intercept - 1.0).log()
    tables = {name: [] for name in ("left", "right", "feature", "threshold", "default_left")}
    offset = 0
    roots = []
    max_depth = 0
    for tree in ensemble["trees"]:
        if any(tree["split_type"]):
            raise ValueError("Expected numerical split thresholds")
        roots.append(offset)
        left = torch.tensor(tree["left_children"], dtype=torch.int32)
        right = torch.tensor(tree["right_children"], dtype=torch.int32)
        depth = [0] * len(left)
        for i, (a, b) in enumerate(zip(tree["left_children"], tree["right_children"])):
            if a >= 0:
                depth[a] = depth[b] = depth[i] + 1
        max_depth = max(max_depth, max(depth))
        node = torch.arange(len(left), dtype=torch.int32)
        tables["left"].append(torch.where(left < 0, node, left) + offset)
        tables["right"].append(torch.where(right < 0, node, right) + offset)
        tables["feature"].append(torch.tensor(tree["split_indices"], dtype=torch.int32))
        tables["threshold"].append(torch.tensor(tree["split_conditions"], dtype=torch.float32))
        tables["default_left"].append(torch.tensor(tree["default_left"], dtype=torch.bool))
        offset += len(left)
    weights = {name: torch.cat(columns) for name, columns in tables.items()}
    weights.update(
        roots=torch.tensor(roots),
        rounds=ensemble["iteration_indptr"],
        depth=max_depth,
        groups=torch.tensor(ensemble["tree_info"]),
        intercept=intercept,
        objective=objective,
    )
    torch.save(weights, output)


def _sum_trees(values: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    """Accumulate boosting rounds in checkpoint order and precision"""
    result = initial.clone()
    for contribution in values.unbind(dim=1):
        result += contribution
    return result


class TensorForest:
    """Evaluate a frozen numerical forest on the input device"""

    def __init__(self, file, device="cpu"):
        self.weights = torch.load(file, map_location=device, weights_only=True)
        self.rounds = self.weights["rounds"]

    def leaves(self, x, start=0, stop=0):
        """Traverse trees in parallel and return absolute terminal node indices"""
        w = self.weights
        roots = w["roots"][self.rounds[start] : self.rounds[stop or -1]]
        nodes = roots.expand(len(x), -1)
        event = torch.arange(len(x), device=x.device)[:, None]
        for _ in range(w["depth"]):
            measured = x[event, w["feature"][nodes].long()]
            go_left = torch.where(measured.isnan(), w["default_left"][nodes], measured < w["threshold"][nodes])
            nodes = torch.where(go_left, w["left"][nodes], w["right"][nodes]).long()
        return nodes, roots

    def evaluate(self, x, base=None, margin=False, iteration_range=(0, 0), leaf=False):
        """
        Return fitted margins, probabilities or tree-local leaf identifiers

        Args:
            x:                Dense numerical features in training order
            base:             Optional margin per jet and class
            margin:           Return raw boosting margins
            iteration_range:  Half-open boosting round range, zero end selects all
            leaf:             Return terminal node identifiers
        Returns:
            result:  Tensor on the input device
        """
        w = self.weights
        x = x.float()
        result = []
        groups = len(w["intercept"])
        start, stop = iteration_range
        stop = stop or len(self.rounds) - 1
        if not 0 <= start < stop < len(self.rounds):
            raise ValueError("Invalid boosting round range")
        group = w["groups"][self.rounds[start] : self.rounds[stop]]
        if not len(x):
            if leaf:
                return x.new_empty((0, len(group)), dtype=torch.long)
            return x.new_empty((0,)) if groups == 1 else x.new_empty((0, groups))
        chunk = max(1, 131072 // len(group))
        for i in range(0, len(x), chunk):
            block = x[i : i + chunk]
            nodes, roots = self.leaves(block, *iteration_range)
            if leaf:
                result.append(nodes - roots)
                continue
            initial = (
                w["intercept"].expand(len(block), -1) if base is None else base.reshape(-1, groups)[i : i + chunk]
            )
            values = w["threshold"][nodes]
            if groups == 1:
                score = _sum_trees(values, initial[:, 0].float())[:, None]
            else:
                score = torch.stack(
                    [_sum_trees(values[:, group == g], initial[:, g].float()) for g in range(groups)], 1
                )
            if not margin and w["objective"] == "binary:logistic":
                score = score.sigmoid()
            if not margin and w["objective"] == "multi:softprob":
                score = score.softmax(1)
            result.append(score[:, 0] if groups == 1 else score)
        return torch.cat(result)

    def leaf_table(self, width):
        """Gather fitted leaf weights for pilot tree response coordinates"""
        w = self.weights
        roots = w["roots"]
        ends = torch.cat((roots[1:], roots.new_tensor([len(w["left"])])))
        ids = roots[:, None] + torch.arange(width, device=roots.device)
        return torch.where(ids < ends[:, None], w["threshold"][ids.clamp_max(len(w["left"]) - 1)], 0.0)
