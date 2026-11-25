# Histogram tree training for shower classification
#
# mikael.mieskolainen@cern.ch (2026)

import json
import sys
from inspect import signature

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import Subset

from twistboard.events import EventLogger
from twistboard.outputs import append_epoch_metrics, checkpoint_index, write_json
from twistformer.api.config import save_config
from twistformer.data.augmentation import prepare_training_dataset
from twistformer.data.preprocessing import PreparedDataset, Preprocessor
from twistformer.logging import print
from twistformer.randomness import numpy_generator, seed_all
from twistformer.training.checkpoint import code_version, save_checkpoint
from twistformer.training.engine import make_loader
from twistformer.training.feature_distributions import write_feature_distributions
from twistformer.training.feature_importance import feature_importance, write_feature_importance_pdf
from twistformer.training.schedulers import make_scheduler, step_scheduler


class BoostingHistory(xgb.callback.EarlyStopping):
    def __init__(self, output, training, timer, score_data):
        """
        Select the best round and open the shared Twistboard event stream

        Args:
            output:  Training run directory
            training:  Learning rate and validation monitor
            timer:  Active training attempt
            score_data:  Validation matrix with event weights for accuracy and AUC
        """
        self.round_limit = min(training.epochs, training.max_steps or training.epochs)
        self.monitor = training.monitor
        metric = {"val_loss": "logloss", "val_auc": "auc", "val_accuracy": "error"}[self.monitor]
        super().__init__(rounds=training.patience or self.round_limit + 1, data_name="val" if metric == "logloss" else score_data,
                         metric_name=metric, maximize=metric == "auc")
        self.score_data = score_data
        self.file = output / "metrics.csv"
        self.learning_rate = training.learning_rate
        self.scheduler = None
        if training.scheduler != "none" or training.warmup_epochs:
            # Keep one CPU rate group for the shared Torch scheduler
            self.schedule_optimizer = torch.optim.SGD([torch.zeros((), device="cpu")], lr=self.learning_rate)
            self.scheduler = make_scheduler(self.schedule_optimizer, training)
            self.learning_rate = self.schedule_optimizer.param_groups[0]["lr"]
        self.timer = timer
        self.logger = EventLogger(output, timer.record["attempts"][-1]["started_at"], timer.elapsed)
        self.logger.start_attempt(0)

    def after_iteration(self, model, epoch, evals_log):
        """
        Record weighted classification scalars and set the next boosting rate

        Args:
            model:  Current native booster
            epoch:  Zero-based boosting round
            evals_log:  Native per-split metric history
        Returns:
            stop: True when the monitored metric exhausts its patience
        """
        validation = evals_log[self.score_data]
        values = {"train_loss": evals_log["train"]["logloss"][-1], "val_loss": evals_log["val"]["logloss"][-1],
                  "val_accuracy": 1 - validation["error"][-1], "val_auc": validation["auc"][-1],
                  "lr": self.learning_rate}
        self.timer.persist()
        self.logger.log_scalars(values, step=epoch + 1, epoch=epoch + 1)
        append_epoch_metrics(self.file, {"epoch": epoch, "step": epoch + 1, **values})
        stop = super().after_iteration(model, epoch, evals_log)
        if self.scheduler is not None and not stop and epoch + 1 < self.round_limit:
            self.schedule_optimizer.step()
            step_scheduler(self.scheduler, values[self.monitor])
            learning_rate = self.schedule_optimizer.param_groups[0]["lr"]
            if learning_rate != self.learning_rate:
                model.set_param({"eta": learning_rate})
                self.learning_rate = learning_rate
        return stop


def _packed_reader(dataset, observables=()):
    """
    Bind a batch reader to the physical split or its sampled and reflected views

    Args:
        dataset:  Prepared classifier events in histogram order
        observables:  Physical coordinates for distribution slices
    Returns:
        read:  Callable on a row slice, or None for sample loading
    """
    if not isinstance(dataset, PreparedDataset) or dataset.augment is not None:
        return None
    preprocessor = dataset.preprocessor
    if type(preprocessor) is not Preprocessor or preprocessor.state is None:
        return None
    state = preprocessor.state
    if "parameter_mean" in state or "target_mean" in state:
        return None
    source = dataset.dataset
    indices = None
    if isinstance(source, Subset):
        indices, source = np.asarray(source.indices), source.dataset
    if not hasattr(source, "classification_batch"):
        return None
    if observables and "observables" not in signature(source.classification_batch).parameters:
        return None
    balance = preprocessor.balance_observable
    options = {"observables": observables} if observables else {}
    scale_features = bool((state["feature_mean"] != 0).any() or (state["feature_std"] != 1).any())

    def read(rows):
        """Gather one bounded batch with the fitted weights and detector column order"""
        batch = source.classification_batch(rows if indices is None else indices[rows], balance, **options)
        features, labels, weights = batch.features, batch.labels, batch.weights
        if labels.shape != (len(features),) or weights.shape != labels.shape or ((labels != 0) & (labels != 1)).any():
            raise ValueError("Expected one binary label and physical weight per descriptor row")
        if features.ndim != 2 or features.shape[1:] != state["feature_mean"].shape:
            raise ValueError("Feature width must match checkpoint preprocessing")
        if scale_features:
            features, _ = preprocessor.transform_inputs(features)
        globals_ = batch.global_features
        if "global_mean" in state:
            if globals_ is None or globals_.shape != (len(features), state["global_mean"].numel()):
                raise ValueError("Global features must match checkpoint preprocessing")
            features = torch.cat((features, (globals_ - state["global_mean"]) / state["global_std"]), dim=1)
        elif globals_ is not None:
            raise ValueError("Checkpoint preprocessing does not declare global features")
        if features.numel() and not torch.isfinite(torch.stack(torch.aminmax(features))).all():
            raise ValueError("XGBoost input features must be finite")
        if "class_weights" in state:
            factors = state["class_weights"][labels]
            if "class_bin_weights" in state:
                bins = preprocessor._balance_bins(batch.balance_values.double(), state["class_bin_edges"])
                factors = factors * state["class_bin_weights"][labels, bins]
            weights = weights.double() * factors
        weights = weights.float()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Sample weights must be finite and nonnegative")
        if observables:
            return features, labels, weights, batch.observables
        return features, labels, weights

    return read


class ShowerBatches(xgb.DataIter):
    def __init__(self, experiment, dataset, observables=()):
        """
        Expose prepared shower rows to the histogram sketch in bounded batches

        Args:
            experiment:  Configured shower classifier
            dataset:  Prepared split with the selected augmentation
            observables:  Physical coordinates for distribution slices
        """
        super().__init__()
        self.model = experiment.model
        spec = experiment.driver.input_spec
        self.names = list(spec.features + spec.global_features)
        self.observables = observables
        self.read_rows = _packed_reader(dataset, observables)
        self.count = len(dataset)
        self.batch_size = experiment.config.training.eval_batch_size
        self.loader = make_loader(dataset, experiment.config.training) if self.read_rows is None else None
        self.batches = None

    def reset(self):
        """Start a fresh pass in the same event order"""
        if self.read_rows is None:
            batches = iter(self.loader)
        else:
            batches = (self.read_rows(slice(i, i + self.batch_size)) for i in range(0, self.count, self.batch_size))
        self.batches = self._arrays(batches)

    def next(self, input_data):
        """
        Pass one detector batch to the native quantile matrix

        Args:
            input_data:  XGBoost input callback
        Returns:
            available:  True if there was a batch to consume
        """
        arrays = next(self.batches, None)
        if arrays is None:
            return False
        features, labels, weights, _ = arrays
        input_data(data=features, label=labels, weight=weights, feature_names=self.names)
        return True

    def arrays(self):
        """Start a fresh pass over the model column arrays"""
        self.reset()
        return self.batches

    def _arrays(self, batches):
        """Yield the model columns, event labels, weights and optional slice coordinates"""
        for batch in batches:
            if self.read_rows is None:
                features, labels, weights = self.model.inputs(batch), batch.targets, batch.sample_weights
                values = batch.observables
            else:
                features, labels, weights = batch[:3]
                values = batch[3] if self.observables else {}
            missing = [name for name in self.observables if name not in values]
            if missing:
                raise ValueError(f"Missing distribution observables: {', '.join(missing)}")
            yield features.numpy(), labels.numpy(), weights.numpy(), {name: values[name].numpy() for name in self.observables}


def shower_matrix(experiment, dataset, reference=None):
    """
    Collect weighted and named XGBoost inputs for one labeled split

    Args:
        experiment:  Configured shower classifier
        dataset:  Prepared event split with the requested training views
        reference:  Training quantiles for validation histograms
    Returns:
        matrix: XGBoost descriptor matrix with class labels and loss weights
    """
    params = experiment.config.model.xgboost_params
    if params["tree_method"] == "hist":
        rows = ShowerBatches(experiment, dataset)
        matrix = xgb.QuantileDMatrix(rows, ref=reference, max_bin=params["max_bin"], nthread=experiment.model.threads)
        if np.any(np.bincount(matrix.get_label().astype(int), weights=matrix.get_weight(), minlength=2) <= 0):
            raise ValueError("XGBoost requires positive weight for both classes in each split")
        return matrix
    spec = experiment.driver.input_spec
    names = list(spec.features + spec.global_features)
    features = np.empty((len(dataset), len(names)), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int32)
    weights = np.empty(len(dataset), dtype=np.float32)
    loader = make_loader(dataset, experiment.config.training)
    offset = 0
    for batch in loader:
        rows = slice(offset, offset + len(batch.ids))
        features[rows] = experiment.model.inputs(batch).numpy()
        labels[rows] = batch.targets.numpy()
        weights[rows] = batch.sample_weights.numpy()
        offset = rows.stop
    if np.any(np.bincount(labels, weights=weights, minlength=2) <= 0):
        raise ValueError("XGBoost requires positive weight for both classes in each training and validation split")
    return xgb.DMatrix(features, label=labels, weight=weights, feature_names=names, nthread=experiment.model.threads)


def fit_booster(experiment, output, device, timer):
    """
    Fit binary logistic trees and save the best validation iteration

    Args:
        experiment:  Configured classifier and shower driver
        output:  Training run directory
        device:  XGBoost training device
        timer:  Shared run timer
    Returns:
        checkpoint: File for evaluation and detector-only prediction
    """
    config = experiment.config
    training = config.training
    seed_all(training.seed)
    if hasattr(experiment.driver, "descriptor_device"):
        experiment.driver.descriptor_device = device
    save_config(config, output / "config.yml")
    print("Loading training dataset")
    train = experiment.driver.dataset("train")
    print("Loading validation dataset")
    validation = experiment.driver.dataset("val")
    print(f"Fitting preprocessing on {len(train):,} training events")
    preprocessor = experiment.driver.create_preprocessor(config=config.preprocessing)
    preprocessor.fit(train, class_count=2 if config.task.balance_classes else 0)
    distributions = config.plots.feature_distributions
    if distributions.bins:
        print(f"Computing training feature histograms for {len(train):,} events")
        observables = getattr(experiment.driver, "evaluation_observables", {})
        categories = distributions.resolve_categories(config.driver, observables)
        names = tuple(dict.fromkeys(category["observable"] for category in categories))
        batches = ShowerBatches(experiment, PreparedDataset(train, preprocessor), names)
        write_feature_distributions(output, batches, experiment.driver.task_spec.outputs, distributions.bins,
                                    categories, observables, config.task.class_weights)
        del batches
    reference = None
    sketch_rows = config.model.xgboost_sketch_rows
    if 0 < sketch_rows < len(train):
        # Estimate bin boundaries on physical events before augmentation
        weights = getattr(train, "weights", None)
        if weights is None:
            weights = np.fromiter((sample.weight for sample in train), dtype=float, count=len(train))
        eligible = np.flatnonzero(np.asarray(weights) > 0)
        sketch_seed = config.model.xgboost_sketch_seed
        rng = numpy_generator(training.seed if sketch_seed is None else sketch_seed)
        indices = rng.choice(eligible, min(len(eligible), sketch_rows), replace=False)
        if sketch_seed is not None:
            indices.sort()
        print(f"Estimating histogram boundaries from {len(indices):,} training events")
        sketch = PreparedDataset(Subset(train, indices), preprocessor)
        reference = xgb.QuantileDMatrix(ShowerBatches(experiment, sketch), max_bin=config.model.xgboost_params["max_bin"],
                                        nthread=experiment.model.threads)
    train = prepare_training_dataset(train, preprocessor, experiment.driver.augment, training,
                                     augment_dataset=getattr(experiment.driver, "augment_dataset", None))
    print(f"Building XGBoost training matrix for {len(train):,} rows")
    train_matrix = shower_matrix(experiment, train, reference)
    del reference
    prepared_validation = PreparedDataset(validation, preprocessor)
    print(f"Building XGBoost validation matrix for {len(validation):,} rows")
    matrices = [(train_matrix, "train"), (shower_matrix(experiment, prepared_validation, train_matrix), "val")]
    score_data = "val"
    if config.task.class_weights:
        # Explicit class penalties enter the loss only, as in the supervised task
        scores = shower_matrix(experiment, prepared_validation, train_matrix)
        for matrix, _ in matrices:
            factors = np.take(config.task.class_weights, matrix.get_label().astype(int))
            matrix.set_weight(matrix.get_weight() * factors)
        score_data = "val_scores"
        matrices.append((scores, score_data))
    write_json(output / "manifest.json", {
        "schema_version": 1, "code_version": code_version(), "torch": str(torch.__version__),
        "python_executable": sys.executable, "numpy": np.__version__,
        "xgboost": xgb.__version__, "strategy": training.strategy, "world_size": 1, "device": str(device),
        "train_events": len(train), "val_events": len(validation), "effective_batch_size": len(train),
    })
    history = BoostingHistory(output, training, timer, score_data)
    print(f"Starting XGBoost training on {device} with {train_matrix.num_col():,} input features "
          f"for up to {history.round_limit:,} rounds")
    booster = xgb.train(
        {**config.model.xgboost_params, "device": str(device),
         "eta": history.learning_rate, "max_depth": config.model.depth, "seed": training.seed,
         "nthread": experiment.model.threads, "eval_metric": ["logloss", "error", "auc"]},
        matrices[0][0], num_boost_round=history.round_limit, evals=matrices,
        callbacks=[history], verbose_eval=25,
    )
    best = booster.best_iteration
    score = 1 - booster.best_score if training.monitor == "val_accuracy" else booster.best_score
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    spec = experiment.driver.input_spec
    for name, ensemble in (("last", booster), ("best", booster[:best + 1])):
        experiment.model.booster = ensemble
        steps = ensemble.num_boosted_rounds()
        checkpoint = checkpoints / f"{name}.pt"
        result = feature_importance(ensemble, spec.features + spec.global_features, checkpoint.name)
        save_checkpoint(checkpoint, experiment, preprocessor, epoch=steps - 1, step=steps,
                        best=score, stale=steps - best - 1, optimizer=None, scheduler=None, scaler=None,
                        training_elapsed_seconds=timer.elapsed(),
                        feature_importance_json=json.dumps(result, indent=2, allow_nan=False))
        write_json(checkpoints / f"{name}_feature_importance.json", result)
    checkpoint_index(output, training.monitor, training.mode)
    plots = output / "plots"
    try:
        plots.mkdir(exist_ok=True)
        write_feature_importance_pdf(plots / "best_feature_importance.pdf", result)
    except Exception as error:
        print(f"Feature importance PDF could not be written: {error}", file=sys.stderr)
    return checkpoint
