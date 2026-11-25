# Flow sampling, transport and density commands
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
from pathlib import Path
import numpy as np
from twistformer.api.predictor import load_predictor
from twistformer.flows.predictor import FlowPredictor


def main(argv=None):
    """
    Run a checkpoint flow operation in physical coordinates

    Args:
        argv:  Optional command-line arguments
    Returns:
        None: Portable output arrays are written
    """
    parser = argparse.ArgumentParser(description="Sample, transport or evaluate a flow density")
    parser.add_argument("operation", choices=["sample", "transport", "log-prob"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--events", type=int)
    parser.add_argument("--count", type=int, default=1, help="Samples per conditioning event")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, help="Override the checkpoint training seed for sampling")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    args = parser.parse_args(argv)
    if min(args.count, args.batch_size) < 1 or (args.events is not None and args.events < 1):
        parser.error("Event and batch counts must be positive")
    predictor = load_predictor(args.checkpoint, args.device)
    if not isinstance(predictor, FlowPredictor):
        parser.error("The checkpoint does not contain a flow task")
    if args.input and args.operation == "sample":
        samples = predictor.experiment.driver.prediction_dataset(args.input)
    else:
        samples = None if args.events is not None and args.input is None else predictor.experiment.driver.dataset(args.split, args.input)
    if args.operation == "sample":
        result = {"samples": predictor.sample(samples, args.count, events=args.events, batch_size=args.batch_size,
                                               seed=args.seed, temperature=args.temperature).numpy()}
    else:
        if samples is None:
            parser.error("Density and transport operations require input events")
        if args.operation == "log-prob":
            result = {"log_prob": predictor.log_prob(samples, batch_size=args.batch_size).numpy()}
        else:
            result = {"trajectory" if args.trajectory else "transport": predictor.transport(
                samples, batch_size=args.batch_size, reverse=args.reverse, trajectory=args.trajectory).numpy()}
    if samples is not None:
        result["ids"] = np.asarray([sample.id for sample in samples])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Flow output exists. Use a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.savez_compressed(file, **result)
    print(f"Flow output: {output}")


if __name__ == "__main__":
    main()
