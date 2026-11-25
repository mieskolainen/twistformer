# Particle encoder throughput and peak device memory
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from drivers.tau3_classification.geometry import hit_momenta
from twistformer import build_experiment, load_config
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.training.engine import compile_model, configure_precision, mixed_precision_dtype

def benchmark(path, args):
    """
    Measure a tau or ECAL encoder after warmup with fixed synthetic inputs

    Args:
        path:  Tau or ECAL model configuration
        args:  Device, batch and timing options
    Returns:
        result: Median step time and peak CUDA memory
    """
    torch.manual_seed(17)
    cfg = load_config(path)
    cfg.training.compile = args.compile
    cfg.training.compile_mode = args.compile_mode
    if args.tf32 is not None:
        cfg.training.tf32 = args.tf32
    if args.checkpoint is not None:
        cfg.model.checkpoint_blocks = args.checkpoint
        if cfg.model.family == "gatr":
            cfg.model.gatr_checkpoint = args.checkpoint
    configure_precision(cfg.training)
    experiment = build_experiment(cfg)
    model = experiment.model.backbone.to(args.device)
    model.train(args.mode == "train")
    if cfg.driver["name"] == "pi0gamma":
        hits = torch.randn(args.batch_size, args.tokens, len(experiment.driver.input_spec.features))
        # ECAL local lengths use 50 mm units
        hits[..., :2] *= 0.3
        hits[..., 2] = hits[..., 2].abs() + 1
        hits[..., 3] -= 5
        if hits.shape[-1] > 4:
            hits[..., 4] = (hits[..., 4] > 0).to(hits.dtype)
            hits[..., 5] = torch.softmax(hits[..., 3], dim=-1)
            hits[..., 6:] = hits[..., 6:].abs() + 0.1
        samples = [Sample(event) for event in hits]
    else:
        hits = torch.randn(args.batch_size, args.tokens, 4)
        hits[..., :3] *= 1000
        hits[..., 3] = hits[..., 3].abs() + 0.1
        samples = [Sample(event, p4=hit_momenta(event)) for event in hits]
    batch = collate_samples(samples).to(args.device)
    forward = compile_model(model, cfg.training)
    device = torch.device(args.device)
    amp = device.type == "cuda" and cfg.training.amp
    amp_dtype = mixed_precision_dtype(device, args.amp_dtype)

    def step():
        """
        Run one encoder forward and optional backward pass
        """
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(args.mode == "train"):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                encoded = forward(batch).features
                loss = encoded.float().square().mean()
            if args.mode == "train":
                loss.backward()
        return loss

    for _ in range(args.warmup):
        loss = step()
    if args.warmup and not torch.isfinite(loss):
        raise RuntimeError("Benchmark warmup produced a nonfinite loss")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)
    return {
        "config": str(path),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "torch_version": torch.__version__,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "compile_dynamic": cfg.training.compile_dynamic,
        "amp_dtype": str(amp_dtype) if amp else None,
        "tf32": cfg.training.tf32,
        "checkpoint": cfg.model.gatr_checkpoint or cfg.model.checkpoint_blocks,
        "median_ms": 1000 * statistics.median(elapsed),
        "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None),
    }

def main():
    """
    Benchmark tau or ECAL encoder cards and write optional JSON measurements
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"), default="default"
    )
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.tokens, args.steps, args.threads) < 1 or args.warmup < 0:
        parser.error("Batch size, tokens, steps and threads must be positive")
    torch.set_num_threads(args.threads)
    results = []
    for path in args.configs:
        result = benchmark(path, args)
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
