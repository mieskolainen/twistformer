# Public experiment API
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass
from drivers.base import Driver, build_driver
from twistformer.api.config import Config
from twistformer.models.factory import Model
from twistformer.randomness import seed_all
from twistformer.tasks.supervised import SupervisedTask

@dataclass
class Experiment:
    config: Config
    driver: Driver
    model: Model
    task: SupervisedTask

def parameter_counts(model):
    """
    Count unique model parameters across all registered architectures

    Args:
        model:  Torch module with optional frozen or shared parameters
    Returns:
        counts: Total and trainable parameter counts
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}

def parameter_summary(model):
    """
    Format parameter counts for the training and evaluation logs

    Args:
        model:  Torch module
    Returns:
        summary: Total and trainable parameter counts
    """
    counts = parameter_counts(model)
    return f"Model parameters: {counts['total']:,} total | {counts['trainable']:,} trainable"

def build_experiment(config: Config) -> Experiment:
    """
    Build a seeded driver, model and supervised task

    Args:
        config:  Validated Config instance
    Returns:
        experiment: Components sharing input and task schemas
    """
    seed_all(config.training.seed)
    app = build_driver(config.driver, config.task)
    from twistformer.evaluation.categories import validate_categories

    observables = getattr(app, "evaluation_observables", {})
    validate_categories(config.evaluation.categories, observables)
    if config.task.name.startswith("flow_"):
        from twistformer.flows.model import FlowModel, FlowTask

        task = FlowTask(app.task_spec, config.task)
        model = FlowModel(app.input_spec, app.task_spec, config)
    elif config.model.family == "xgboost":
        if not getattr(app, "event_descriptors", False) or len(app.task_spec.outputs) != 2:
            raise ValueError("XGBoost requires a binary driver with one descriptor row per event")
        if config.plots.feature_distributions.bins:
            config.plots.feature_distributions.resolve_categories(config.driver, observables)
        from twistformer.models.xgboost import ShowerBooster

        task = SupervisedTask(app.task_spec, config.task)
        model = ShowerBooster(config.model.xgboost_threads)
    else:
        task = SupervisedTask(app.task_spec, config.task)
        model = Model(app.input_spec, app.task_spec, config.model, config.task)
    return Experiment(config, app, model, task)
