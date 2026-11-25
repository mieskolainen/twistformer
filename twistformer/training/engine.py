# Training and checkpoint persistence
#
# mikael.mieskolainen@cern.ch (2026)

"""One weighted supervised training loop for every model and driver"""

from twistformer.logging import print
from twistboard.timing import format_duration
from twistboard.events import EventLogger
import csv
import json
import os
import shutil
import time
from contextlib import ExitStack, closing, nullcontext
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from twistformer.api.config import parse_config, save_config
from twistformer.data.batching import collate_samples, drop_tokens
from twistformer.data.augmentation import prepare_training_dataset
from twistformer.data.preprocessing import PreparedDataset
from twistformer.data.loading import EpochDataset, seed_worker, set_loader_epoch, close_loader
from twistformer.geometry.lorentz import validate_p4
from twistformer.metrics import one_vs_rest_auc
from twistformer.randomness import seed_all, torch_generator
from .checkpoint import save_checkpoint, read_checkpoint, restore_model, restore_rng, code_version, rng_state, finish_checkpoint_writes
from .distributed import DistributedRuntime, PaddedTrainingDataset, WeightedDistributedSampler, distributed_session
from .ema import ModelEMA
from .schedulers import make_scheduler, step_scheduler
from .optimizers import Lion, MuonAdamW
from .timing import BatchTiming

def device_for(name):
    """
    Resolve explicit or automatic Torch device selection

    Args:
        name:  Device string or auto
    Returns:
        device: Torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)

def configure_precision(cfg):
    """
    Apply the checkpointed CUDA float32 arithmetic policy

    Args:
        cfg:  TrainingConfig with the TF32 switch
    Returns:
        None: Matmul and cuDNN use the same precision policy
    """
    precision = "tf32" if cfg.tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.conv.fp32_precision = precision
    torch.backends.cudnn.rnn.fp32_precision = precision

def make_grad_scaler(device, enabled):
    """
    Construct the gradient scaler exposed by the installed Torch runtime

    Args:
        device:  Training device
        enabled:  Enable scaling for float16 training
    Returns:
        scaler: Gradient scaler with checkpointable state
    """
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    if enabled and device.type != "cuda":
        raise ValueError("This Torch runtime supports float16 gradient scaling only on CUDA")
    return torch.cuda.amp.GradScaler(enabled=enabled)

def mixed_precision_dtype(device, preference="auto"):
    """
    Resolve the configured autocast dtype

    Args:
        device:  Torch execution device
        preference:  auto, bf16 or fp16
    Returns:
        dtype: Torch floating-point dtype
    """
    if preference != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[preference]
    return torch.bfloat16 if device.type == "cpu" or torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16

def make_optimizer(model, cfg):
    """
    Construct the optimizer with explicit moment and decay policies

    Args:
        model:  Trainable model
        cfg:  TrainingConfig with optimizer parameters
    Returns:
        optimizer: Configured Torch optimizer
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exclude = not cfg.decay_norm_bias and (parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower())
        (no_decay if exclude else decay).append(parameter)
    groups = [
        {"params": parameters, "weight_decay": weight_decay}
        for parameters, weight_decay in ((decay, cfg.weight_decay), (no_decay, 0.0))
        if parameters
    ]
    options = {"lr": cfg.learning_rate}
    if cfg.optimizer != "sgd":
        options.update(betas=tuple(cfg.optimizer_betas))
        if cfg.optimizer != "lion":
            options["eps"] = cfg.optimizer_eps
    if cfg.fused_optimizer:
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("fused_optimizer requires CUDA")
        options["fused"] = True
    if cfg.optimizer == "muon_adamw":
        matrix_parameters = {
            parameter
            for layer in model.modules()
            if isinstance(layer, (torch.nn.Linear, torch.nn.MultiheadAttention))
            for parameter in layer.parameters(recurse=False)
            if parameter.ndim == 2 and not parameter.is_complex()
        }
        return MuonAdamW(groups, matrix_parameters, **options)
    factory = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD, "lion": Lion}[cfg.optimizer]
    return factory(groups, **options)

def normalize_gradients(parameters, normalizer):
    """
    Normalize likelihood gradients in groups with equal device and dtype

    Args:
        parameters:  Model parameters with optional accumulated gradients
        normalizer:  Positive global event weight divided by the rank count
    Returns:
        None: Gradients are divided in place before clipping
    """
    groups = {}
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    for gradients in groups.values():
        torch._foreach_div_(gradients, normalizer)


def clip_gradients(model, max_norm, *, allow_nonfinite=False):
    """
    Clip weighted gradients and distinguish invalid values from norm overflow

    Args:
        model:  Model with normalized gradients
        max_norm:  Positive norm limit or infinity for validation only
        allow_nonfinite:  Let an active GradScaler skip invalid gradients
    Returns:
        norm: Gradient norm before clipping
    """
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None]
    parameters = [parameter for _, parameter in named]
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).tolist()
        invalid = [name for (name, _), valid in zip(named, finite, strict=True) if not valid]
        if invalid:
            if allow_nonfinite:
                return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)
            raise FloatingPointError(
                f"Nonfinite gradients in {len(invalid)} parameters: {', '.join(invalid[:8])}. "
                "No optimizer update was applied"
            ) from error
        # Finite float32 gradients can overflow the sum of squares
        norms = [
            torch.linalg.vector_norm(parameter.grad.to(torch.complex128 if parameter.grad.is_complex() else torch.float64))
            for parameter in parameters
        ]
        norm = torch.linalg.vector_norm(torch.stack(norms))
        if not torch.isfinite(norm):
            raise FloatingPointError("Gradient norm exceeds float64 range. No optimizer update was applied") from error
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        groups = {}
        for parameter in parameters:
            gradient = parameter.grad
            groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
        for gradients in groups.values():
            torch._foreach_mul_(gradients, coefficient.to(device=gradients[0].device, dtype=gradients[0].real.dtype))
        return norm


def compile_model(model, cfg):
    """
    Compile the event model with the configured shape policy

    Args:
        model:  Trainable event model
        cfg:  TrainingConfig with compiler mode and shape policy
    Returns:
        forward_model: Compiled callable or the original eager model
    """
    if not cfg.compile:
        return model
    print(
        f"Starting model compilation: mode={cfg.compile_mode}, dynamic={cfg.compile_dynamic}. "
        "The first forward/backward passes include compilation time.",
        flush=True,
    )
    options = dict(torch._inductor.list_mode_options(cfg.compile_mode))
    options["triton.cudagraph_skip_dynamic_graphs"] = True
    return torch.compile(model, dynamic=cfg.compile_dynamic, options=options)

def make_loader(dataset, cfg, *, train=False, epoch=0, runtime=None, batch_size=None, allow_negative_weights=False):
    """
    Construct deterministic per-epoch event batching

    Args:
        dataset:  Prepared training or validation dataset
        cfg:  TrainingConfig
        train:  Shuffle and use the training batch size
        epoch:  Epoch index for the shuffle seed
        runtime:  Optional distributed rank and device
        batch_size:  Optional standalone inference batch size
        allow_negative_weights:  Permit signed evaluation yields
    Returns:
        loader: Configured DataLoader
    """
    runtime = runtime or DistributedRuntime(device_for(cfg.device))
    options = {}
    sampler = None
    if runtime.enabled:
        if train:
            sampler = WeightedDistributedSampler(
                dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True, seed=cfg.seed
            )
            sampler.set_epoch(epoch)
            dataset = PaddedTrainingDataset(dataset)
        else:
            sampler = range(runtime.rank, len(dataset), runtime.world_size)
    if cfg.workers:
        dataset = EpochDataset(dataset, cfg.seed + runtime.rank)
        # CUDA workers must not inherit compiled kernels from the parent
        if runtime.enabled or runtime.device.type == "cuda":
            options["multiprocessing_context"] = "spawn"
        options["persistent_workers"] = True
        options["prefetch_factor"] = cfg.prefetch_factor if train else cfg.eval_prefetch_factor
    if train and sampler is None:
        sampler = RandomSampler(dataset, generator=torch.Generator())
    loader = DataLoader(
        dataset,
        batch_size=(batch_size if batch_size is not None else cfg.batch_size if train else cfg.eval_batch_size),
        sampler=sampler,
        num_workers=cfg.workers,
        collate_fn=partial(collate_samples, allow_negative_weights=allow_negative_weights),
        worker_init_fn=seed_worker,
        generator=torch_generator(cfg.seed + epoch + runtime.rank),
        pin_memory=cfg.pin_memory and runtime.device.type == "cuda",
        **options,
    )
    loader.epoch_seed = cfg.seed + runtime.rank
    set_loader_epoch(loader, epoch)
    return loader

def validate_batch(batch, physical):
    """
    Check event support and required four-momentum geometry

    Args:
        batch:  Padded particle Batch
        physical:  Require future non-spacelike p4
    Returns:
        None: Invalid inputs raise ValueError
    """
    if not batch.mask.any(1).all():
        raise ValueError("Empty events are unsupported. driver must explicitly filter them")
    if batch.global_features is not None and (
        batch.global_features.ndim != 2
        or batch.global_features.shape[0] != len(batch.mask)
        or not torch.isfinite(batch.global_features).all()
    ):
        raise ValueError("Global features must be finite event vectors")
    if batch.parameters is not None and (
        batch.parameters.ndim != 2
        or batch.parameters.shape[0] != len(batch.mask)
        or not torch.isfinite(batch.parameters).all()
    ):
        raise ValueError("Parameters must be finite event vectors")
    if physical:
        if batch.p4 is None:
            raise ValueError("Missing p4 geometry")
        validate_p4(batch.p4, batch.mask)

def evaluate_validation(model, task, loader, device, amp=False, amp_dtype="auto", runtime=None):
    """
    Evaluate weighted loss and the task selection metric

    Args:
        model:  Shared model instance
        task:  Supervised task
        loader:  Validation DataLoader
        device:  Torch device
        amp:  Use the training mixed-precision policy
        amp_dtype:  Configured autocast dtype preference
        runtime:  Optional global metric reduction
    Returns:
        metrics: Validation loss with classifier accuracy and ROC AUC or task error
    """
    runtime = runtime or DistributedRuntime(device)
    runtime.sync_buffers(model)
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    roc_batches = []
    dtype = mixed_precision_dtype(device, amp_dtype)
    last_report = time.monotonic()
    with (
        torch.no_grad() if task.spec.name.startswith("flow_") else torch.inference_mode(),
        tqdm(
            loader,
            desc="validation",
            unit="batch",
            bar_format="{l_bar}{bar:20}{r_bar}",
            dynamic_ncols=True,
            leave=False,
            disable=not runtime.primary,
        ) as progress,
    ):
        for batch in progress:
            validate_batch(batch, model.requires_p4)
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                prediction = model(batch)
            result = task.compute_loss(prediction, batch)
            if task.spec.name == "classification":
                probabilities = prediction.probabilities
                metric_numerator = ((probabilities.argmax(-1) == batch.targets) * batch.sample_weights).sum()
                metric_denominator = batch.sample_weights.sum()
                roc_batches.append(torch.column_stack((batch.targets, batch.sample_weights, probabilities)).cpu().numpy())
            elif task.spec.name.startswith("flow_"):
                metric_numerator, metric_denominator = result.numerator.detach(), result.normalizer.detach()
            else:
                valid = torch.isfinite(batch.targets)
                if batch.target_mask is not None:
                    valid &= batch.target_mask
                residual = (prediction.mean - batch.targets.masked_fill(~valid, 0)).square().masked_fill(~valid, 0)
                count = valid.sum(-1)
                w = batch.sample_weights * (count > 0)
                metric_numerator = (residual.sum(-1) / count.clamp_min(1) * w).sum()
                metric_denominator = w.sum()
            totals += torch.stack((result.numerator, result.normalizer, metric_numerator, metric_denominator)).double()
            if runtime.primary and time.monotonic() - last_report >= 1.0:
                numerator, denominator = totals[:2].tolist()
                if denominator > 0:
                    progress.set_postfix(val=f"{numerator / denominator:.6g}", refresh=False)
                last_report = time.monotonic()
    numerator, denominator, metric_numerator, metric_denominator = runtime.sum_tensor(totals).tolist()
    if denominator <= 0:
        raise ValueError("Evaluation contains no positively weighted observations")
    value = numerator / denominator
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite validation loss")
    metric = metric_numerator / metric_denominator
    if task.spec.name.startswith("flow_"):
        return {"val_loss": value, "val_flow_loss" if task.spec.name == "flow_matching" else "val_nll": value}
    if task.spec.name == "classification":
        rank_batches = runtime.gather_on_primary(roc_batches)

        def validation_auc():
            """Rank all validation events before computing the weighted ROC area"""
            events = np.concatenate([batch for batches in rank_batches for batch in batches])
            _, area = one_vs_rest_auc(events[:, 0], events[:, 2:], events[:, 1])
            return area if area is not None else float("nan")

        return {"val_loss": value, "val_accuracy": metric, "val_auc": runtime.on_primary(validation_auc)}
    return {"val_loss": value, "val_rmse": metric**0.5}

class Trainer:
    def __init__(self, run_dir, *, resume=None, init_from=None, command=None):
        self.run_dir = Path(run_dir)
        self.resume, self.init_from = resume, init_from
        self.command = command
        if resume and init_from:
            raise ValueError("resume and init_from are mutually exclusive")

    def fit(self, experiment):
        """
        Train one experiment and persist portable checkpoints

        Args:
            experiment:  Driver, model, task and validated configuration
        Returns:
            checkpoint: Path to the best validation checkpoint
        """
        from twistboard.experiments import training_run
        from twistformer.api.experiment import parameter_counts

        boosted = experiment.config.model.family == "xgboost"
        if boosted and (self.resume or self.init_from):
            raise ValueError("XGBoost saves the validation-selected ensemble for inference. Start training with --config")
        with distributed_session(experiment.config.training) as runtime, ExitStack() as stack:
            def begin_run():
                """
                Keep the lifecycle timer on the rank that writes output files
                """
                self.timer = stack.enter_context(
                    training_run(
                        self.run_dir, experiment.config.to_dict(), self.resume, self.init_from,
                        command=self.command, model_parameters=parameter_counts(experiment.model),
                    )
                )

            runtime.on_primary(begin_run)
            if boosted:
                from .xgboost import fit_booster

                device = experiment.model.resolve_device(experiment.config.training.device)
                return fit_booster(experiment, self.run_dir, device, self.timer)
            with ExitStack() as loaders:
                self._loaders = loaders
                return self._fit(experiment, runtime)

    def _fit(self, experiment, runtime):
        """
        Optimize one experiment within its recorded training lifecycle

        Args:
            experiment:  Configured model, driver and task
            runtime:  Device and distributed collective operations
        Returns:
            checkpoint: Best validation checkpoint path
        """
        from twistboard.outputs import append_epoch_metrics, checkpoint_index
        from twistformer.api.experiment import parameter_summary

        cfg = experiment.config.training
        device = runtime.device
        if device.type == "cuda":
            configure_precision(cfg)
        # build_experiment is seeded by CLI/Python factory before parameter creation
        seed_all(cfg.seed + runtime.rank)
        root = self.run_dir
        checkpoints = root / "checkpoints"
        runtime.on_primary(lambda: checkpoints.mkdir(exist_ok=True))
        state = read_checkpoint(self.resume or self.init_from) if (self.resume or self.init_from) else None
        if state and self.resume and state.get("inference_only"):
            raise ValueError("Imported flow weights have no portable optimizer or RNG state. Use init_from")
        if state and self.resume and runtime.primary:
            self.timer.inherit(state.get("training_elapsed_seconds"))
        app, model = experiment.driver, experiment.model.to(device)
        active_device = device
        if device.type == "cuda" and device.index is None:
            active_device = torch.device("cuda", torch.cuda.current_device())
        device_name = f" ({torch.cuda.get_device_name(active_device)})" if active_device.type == "cuda" else ""
        rank_label = f"Rank {runtime.rank}: " if runtime.enabled else ""
        print(f"{rank_label}Active device: {active_device}{device_name}", flush=True)
        if runtime.primary:
            print(parameter_summary(model), flush=True)
            print(
                f"Gradient accumulation: {cfg.accumulation} batches per optimizer update",
                flush=True,
            )
            print(
                f"Effective batch size: {cfg.batch_size * cfg.accumulation * runtime.world_size:,} events "
                f"= {cfg.batch_size} events per rank "
                f"x {cfg.accumulation} accumulation x {runtime.world_size} ranks "
                "(full batches and accumulation windows)",
                flush=True,
            )
        datasets = []
        for split in ("train", "val"):
            if runtime.primary:
                print(f"Loading {split} dataset...", flush=True)
            dataset = app.dataset(split)
            if runtime.primary:
                print(f"Loaded {split} dataset: {len(dataset):,} events", flush=True)
            datasets.append(dataset)
        train_raw, val_raw = datasets
        if not len(train_raw) or not len(val_raw):
            raise ValueError("Training and validation sets must be nonempty")
        reweighting_digest = None
        if experiment.config.reweighting.path:
            from twistformer.api.reweighting import FoldReweighter

            ratio = FoldReweighter(experiment.config.reweighting.path, str(device))
            if state and self.resume and not torch.equal(
                state["preprocessing"].get("reweighting_digest", torch.empty(0)), ratio.digest
            ):
                raise ValueError("Reweighting file changed since the training checkpoint")
            print("Applying classifier density ratios...", flush=True)
            train_raw = ratio.apply(train_raw, app.input_spec, app.task_spec, experiment.config.reweighting.batch_size)
            val_raw = ratio.apply(val_raw, app.input_spec, app.task_spec, experiment.config.reweighting.batch_size)
            reweighting_digest = ratio.digest
            if device.type == "cuda":
                configure_precision(cfg)
            seed_all(cfg.seed + runtime.rank)
        if state:
            restore_model(experiment, state, inference=bool(self.init_from))
            if self.resume and state["config"] != experiment.config.to_dict():
                # The CLI permits runtime device/workers overrides only
                a, b = (parse_config(state["config"]).to_dict(), experiment.config.to_dict())
                import copy

                a = copy.deepcopy(a)
                for key in ("device", "workers"):
                    a["training"][key] = b["training"][key]
                if a != b:
                    raise ValueError("Resume configuration changed. use init_from for a new experiment")
        preprocessor = app.create_preprocessor(
            state["preprocessing"] if state and self.resume else None, config=experiment.config.preprocessing
        )

        def fit_preprocessing():
            """
            Fit shared normalization on the full physical training sample

            Returns:
                state: Rank-zero preprocessing statistics
            """
            if preprocessor.state is not None:
                print("Preprocessing restored from checkpoint", flush=True)
                return preprocessor.state
            print("Preparing training preprocessing...", flush=True)
            with tqdm(
                train_raw,
                total=len(train_raw),
                desc="preprocessing",
                unit="event",
                bar_format="{l_bar}{bar:20}{r_bar}",
                dynamic_ncols=True,
                leave=False,
            ) as progress:
                preprocessor.fit(
                    progress,
                    experiment.task.spec.name != "classification",
                    class_count=(len(experiment.task.spec.outputs) if experiment.config.task.balance_classes else 0),
                )
            print("Training preprocessing ready", flush=True)
            return preprocessor.state

        preprocessor.state = runtime.on_primary(fit_preprocessing)
        if reweighting_digest is not None:
            preprocessor.state["reweighting_digest"] = reweighting_digest
        if runtime.primary:
            print(f"Driver augmentation: {cfg.augmentation}, training token dropout: {cfg.token_dropout:g}", flush=True)
        train = prepare_training_dataset(
            train_raw, preprocessor, app.augment, cfg, augment_dataset=getattr(app, "augment_dataset", None),
            progress=runtime.primary,
        )
        if runtime.primary and len(train) != len(train_raw):
            print(f"Training after augmentation: {len(train):,} examples from {len(train_raw):,} events", flush=True)
        val = PreparedDataset(val_raw, preprocessor)
        distributed_model = runtime.wrap(model, cfg)
        optimizer = make_optimizer(model, cfg)
        scheduler = make_scheduler(optimizer, cfg)
        ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None
        amp_dtype = mixed_precision_dtype(device, cfg.amp_dtype)
        scaler = make_grad_scaler(device, enabled=cfg.amp and amp_dtype == torch.float16)
        initial_best = float("inf") if cfg.mode == "min" else -float("inf")
        start, step, best, stale = 0, 0, initial_best, 0
        if state and self.resume:
            optimizer.load_state_dict(state["optimizer"])
            if scheduler:
                scheduler.load_state_dict(state["scheduler"])
            if ema is not None:
                ema.load_state_dict(state["ema"])
            scaler.load_state_dict(state["scaler"])
            start, step, best, stale = (state["epoch"] + 1, state["step"], state["best"], state["stale"])
            if state.get("world_size", 1) != runtime.world_size:
                raise ValueError("Resume requires the checkpoint world size. Use init_from to change GPU count")
            streams = state.get("rank_rng")
            restore_rng(streams[runtime.rank] if streams is not None else state["rng"])
            metrics_path = root / "metrics.csv"
            if metrics_path.exists():
                with metrics_path.open() as f:
                    rows = list(csv.DictReader(f))
                if rows and int(rows[-1]["epoch"]) > state["epoch"]:
                    raise ValueError("Resuming an older checkpoint requires --output with a new directory")
            elif start < cfg.epochs:
                # A fork retains optimizer/RNG state but selects its own best model
                best, stale = initial_best, 0
            if start >= cfg.epochs or (cfg.max_steps and step >= cfg.max_steps) or (cfg.patience and stale >= cfg.patience):
                if (checkpoints / "best.pt").exists():
                    return checkpoints / "best.pt"
                raise ValueError("Checkpoint completed its training budget. use --init-from for a new run")

        def write_metadata():
            """
            Record the event counts and distributed batch convention

            Returns:
                None: Run metadata are written by rank zero
            """
            save_config(experiment.config, root / "config.yml")
            self.logger = EventLogger(root, self.timer.record["attempts"][-1]["started_at"], self.timer.elapsed)
            self.logger.start_attempt(step)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "code_version": code_version(),
                        "torch": str(torch.__version__),
                        "train_events": len(train),
                        "val_events": len(val),
                        "schema_version": 1,
                        "strategy": cfg.strategy,
                        "world_size": runtime.world_size,
                        "effective_batch_size": cfg.batch_size * cfg.accumulation * runtime.world_size,
                    },
                    indent=2,
                )
            )

        runtime.on_primary(write_metadata)
        forward_model = compile_model(distributed_model, cfg)
        val_loader = make_loader(val, cfg, runtime=runtime)
        self._loaders.callback(close_loader, val_loader)
        loader = make_loader(train, cfg, train=True, epoch=start, runtime=runtime)
        self._loaders.callback(close_loader, loader)
        if runtime.primary:
            print(
                f"Data loaders: train batch {cfg.batch_size}, "
                f"validation batch {cfg.eval_batch_size}, "
                f"workers {cfg.workers} per rank",
                flush=True,
            )
        timing = BatchTiming(cfg.profile_batches, device, root / f"batch_timing_rank{runtime.rank}.csv")
        self._loaders.callback(timing.save)
        if cfg.profile_batches and runtime.primary:
            print(f"Timing {cfg.profile_batches} batches with device synchronization: {timing.path}", flush=True)
        for epoch in range(start, cfg.epochs):
            forward_model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            set_loader_epoch(loader, epoch)
            set_loader_epoch(val_loader, epoch)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            window = torch.zeros(2, dtype=torch.float64, device=device)
            last_report = time.monotonic()
            with closing(timing.batches(loader, epoch)) as batches, tqdm(
                batches,
                total=len(loader),
                desc=f"epoch {epoch + 1}/{cfg.epochs}",
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}",
                dynamic_ncols=True,
                leave=False,
                disable=not runtime.primary,
                postfix={"lr": f"{epoch_lr:.2e}"},
            ) as progress:
                for index, batch in enumerate(progress):
                    if cfg.scheduler_interval == "step":
                        epoch_lr = optimizer.param_groups[0]["lr"]
                    validate_batch(batch, model.requires_p4)
                    batch = batch.to(device, non_blocking=cfg.pin_memory)
                    batch = drop_tokens(batch, cfg.token_dropout, cfg.token_min_keep)
                    boundary = (index + 1) % cfg.accumulation == 0 or index + 1 == len(loader)
                    # Static DDP establishes its graph on the first backward pass
                    skip_sync = (
                        runtime.enabled
                        and not boundary
                        and not (cfg.ddp_static_graph and epoch == start and index < cfg.accumulation)
                    )
                    sync = distributed_model.no_sync() if skip_sync else nullcontext()
                    with sync:
                        with torch.autocast(device.type, dtype=amp_dtype, enabled=cfg.amp):
                            prediction = forward_model(batch)
                        result = experiment.task.compute_loss(prediction, batch)
                        loss_numerator = result.numerator.detach()
                        loss_weight = result.normalizer.detach()
                        window += torch.stack((loss_weight, (~torch.isfinite(loss_numerator)).to(loss_weight.dtype))).double()
                        scaler.scale(result.numerator).backward()
                    totals += torch.stack((loss_numerator, loss_weight)).double()
                    if runtime.primary:
                        timing.persist(self.timer)
                        if time.monotonic() - last_report >= 1.0:
                            numerator, denominator = totals.tolist()
                            if denominator > 0:
                                progress.set_postfix(
                                    train=f"{numerator / denominator:.6g}", lr=f"{epoch_lr:.2e}",
                                    total=format_duration(self.timer.elapsed()), refresh=False,
                                )
                            last_report = time.monotonic()
                    if boundary:
                        # Check all ranks before updating, including partial windows
                        # DDP averages gradients, the likelihood uses global event weight
                        window_weight, nonfinite = runtime.sum_tensor(window).tolist()
                        if nonfinite:
                            raise FloatingPointError(f"Nonfinite loss in accumulation window at epoch {epoch}, batch {index}")
                        window_weight /= runtime.world_size
                        scaler.unscale_(optimizer)
                        attempted_update = window_weight > 0
                        previous_scale = scaler.get_scale()
                        if attempted_update:
                            normalize_gradients(model.parameters(), window_weight)
                            clip_gradients(
                                model,
                                cfg.grad_clip if cfg.grad_clip else float("inf"),
                                allow_nonfinite=scaler.is_enabled(),
                            )
                            scaler.step(optimizer)
                        scaler.update()
                        if attempted_update and scaler.get_scale() >= previous_scale:
                            step += 1
                            if cfg.scheduler_interval == "step":
                                step_scheduler(scheduler, None)
                            if ema is not None:
                                runtime.sync_buffers(model)
                                ema.update(model)
                        optimizer.zero_grad(set_to_none=True)
                        window.zero_()
                        if cfg.max_steps and step >= cfg.max_steps:
                            break
            numerator, denominator = runtime.sum_tensor(totals).tolist()
            runtime.sync_buffers(model)
            if denominator <= 0:
                raise ValueError("Training contains no positively weighted observations")
            validation = evaluate_validation(
                ema.module if ema is not None and cfg.ema_eval else model,
                experiment.task,
                val_loader,
                device,
                cfg.amp,
                cfg.amp_dtype,
                runtime=runtime,
            )
            metric = validation[cfg.monitor]
            if not np.isfinite(metric):
                raise ValueError(f"Cannot monitor {cfg.monitor} without a defined validation value")
            improved = metric < best if cfg.mode == "min" else metric > best
            best = metric if improved else best
            stale = 0 if improved else stale + 1
            if cfg.scheduler_interval == "epoch":
                step_scheduler(scheduler, metric)
            row = {"epoch": epoch, "step": step, "train_loss": numerator / denominator, **validation, "lr": epoch_lr}
            rank_rng = runtime.gather_rng(rng_state(device if runtime.enabled else None))

            @finish_checkpoint_writes()
            def write_epoch():
                """
                Persist one globally reduced epoch and its rank RNG streams

                Returns:
                    None: Metrics and portable checkpoints are written by rank zero
                """
                path = root / "metrics.csv"
                self.timer.persist(force=True)
                self.logger.log_scalars({key: value for key, value in row.items() if key not in {"epoch", "step"}}, step=step, epoch=epoch + 1)
                append_epoch_metrics(path, row)
                save_checkpoint(
                    checkpoints / "last.pt",
                    experiment,
                    preprocessor,
                    epoch=epoch,
                    step=step,
                    best=best,
                    stale=stale,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    rank_rng=rank_rng,
                    training_elapsed_seconds=self.timer.elapsed(),
                )
                if improved:
                    shutil.copy2(checkpoints / "last.pt", checkpoints / "best.tmp")
                    os.replace(checkpoints / "best.tmp", checkpoints / "best.pt")
                if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
                    shutil.copy2(checkpoints / "last.pt", checkpoints / f"epoch_{epoch:05d}.pt")
                    if cfg.keep_checkpoints:
                        for path in sorted(checkpoints.glob("epoch_*.pt"))[: -cfg.keep_checkpoints]:
                            path.unlink()
                checkpoint_index(root, cfg.monitor, cfg.mode)
                auc_text = ""
                if "val_auc" in validation:
                    area = validation["val_auc"]
                    auc_text = f"val_auc={area:.6g} " if np.isfinite(area) else "val_auc=undefined "
                print(
                    f"epoch {epoch + 1}/{cfg.epochs} train={row['train_loss']:.6g} "
                    f"val={validation['val_loss']:.6g} {auc_text}lr={row['lr']:.2e} "
                    f"total={format_duration(self.timer.elapsed())}",
                    flush=True,
                )

            runtime.on_primary(write_epoch)
            if cfg.max_steps and step >= cfg.max_steps:
                break
            if cfg.patience and stale >= cfg.patience:
                break
        return checkpoints / "best.pt"
