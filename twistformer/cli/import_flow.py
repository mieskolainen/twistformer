# Verified Supraflow checkpoint conversion
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
import torch
from twistformer.api.config import load_config
from twistformer.flows.legacy import import_checkpoint


def main(argv=None):
    """
    Convert explicitly configured legacy flow weights

    Args:
        argv:  Optional command arguments
    Returns:
        None: Verified checkpoint is written
    """
    parser = argparse.ArgumentParser(description="Import a verified Supraflow vector-field checkpoint")
    for name in ("checkpoint", "output", "config", "preprocessing", "reference"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    output = import_checkpoint(args.checkpoint, args.output, load_config(args.config),
                               torch.load(args.preprocessing, map_location="cpu", weights_only=True),
                               torch.load(args.reference, map_location="cpu", weights_only=True))
    print(f"Imported flow checkpoint: {output}")


if __name__ == "__main__":
    main()
