# Experiment workflow and dashboard steering
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
import json
from pathlib import Path
import subprocess
import sys
from .experiments import unique_directory
from .commands import run_command

def main(argv=None):
    """
    Run training plus evaluation or serve the experiment dashboard

    Args:
        argv:  Command-line arguments, or None for process arguments
    Returns:
        None: Workflow results or the local dashboard URL are printed
    """
    parser = argparse.ArgumentParser(description="Twistboard experiment management")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train and evaluate the best validation checkpoint")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume")
    run.add_argument("--root", default="runs")
    run.add_argument("--name", default="experiment")
    run.add_argument("--set", action="append", default=[])
    run.add_argument("--init-from")
    run.add_argument("--device")
    run.add_argument("--workers", type=int)
    run.add_argument("--split", choices=["train", "val", "test"], default="test")
    run.add_argument("--no-evaluate", action="store_true")
    run.add_argument("--validate", action="store_true", help="Check configuration without opening data")
    serve = commands.add_parser("serve", help="Open the experiment dashboard")
    serve.add_argument("--root", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true")
    serve.add_argument("--force", action="store_true", help="Stop the previous same-user Twistboard on this port (Linux)")
    listing = commands.add_parser("list", help="List recorded experiments")
    listing.add_argument("--root", default="runs")
    scan = commands.add_parser("scan", help="Expand a grid scan and optionally train its trials")
    scan.add_argument("--config", required=True)
    scan.add_argument("--root", default="runs")
    scan.add_argument("--execute", action="store_true", help="Train trials sequentially")
    synthetic = commands.add_parser("synthetic", help="Generate synthetic scans for dashboard testing")
    synthetic.add_argument("--root", default="runs")
    synthetic.add_argument("--trials", type=int, default=576)
    synthetic.add_argument("--epochs", type=int, default=40)
    synthetic.add_argument("--seed", type=int, default=19)
    args = parser.parse_args(argv)
    if args.command in {"scan", "synthetic"}:
        from .scans import run_scan, synthetic_scans

        try:
            manifest = run_scan(args.config, args.root, args.execute) if args.command == "scan" else synthetic_scans(args.root, args.trials, args.epochs, args.seed)
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        print(f"Scan manifest: {manifest}")
        return
    if args.command == "serve":
        from .server import serve_dashboard

        try:
            serve_dashboard(args.root, args.host, args.port, args.open, force=args.force)
        except (OSError, RuntimeError) as error:
            parser.exit(1, f"Twistboard: {error}\n")
        return
    if args.command == "list":
        from .server import list_runs

        print(json.dumps(list_runs(Path(args.root).resolve()), indent=2))
        return
    if args.resume and (args.set or args.init_from):
        parser.error("Resume accepts only runtime device and worker overrides")
    root = Path(args.resume).resolve().parent.parent if args.resume else unique_directory(args.root, args.name)
    training = [sys.executable, "-u", "-m", "twistformer.cli.train", "--output", str(root)]
    training += ["--resume", args.resume] if args.resume else ["--config", args.config]
    for item in args.set:
        training += ["--set", item]
    if args.init_from:
        training += ["--init-from", args.init_from]
    if args.device:
        training += ["--device", args.device]
    if args.workers is not None:
        training += ["--workers", str(args.workers)]
    if args.validate:
        training.append("--validate")
    print(f"Experiment: {root}", flush=True)
    try:
        run_command(training)
        if not args.validate and not args.no_evaluate:
            evaluation = [
                sys.executable,
                "-u",
                "-m",
                "twistformer.cli.evaluate",
                "--checkpoint",
                str(root / "checkpoints/best.pt"),
                "--split",
                args.split,
            ]
            if args.device:
                evaluation += ["--device", args.device]
            run_command(evaluation)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
