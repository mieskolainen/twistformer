# Command-line model comparison
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
import argparse
from twistformer.evaluation.comparison import compare, comparison_config

def main(argv=None):
    """
    Evaluate models from a shared comparison card

    Args:
        argv: Optional command-line arguments
    Returns:
        None: Comparison files are written to disk
    """
    parser = argparse.ArgumentParser(description="Compare models on common events")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Empty comparison output directory")
    args = parser.parse_args(argv)
    print(f"Comparison config: {args.config}", flush=True)
    output = compare(comparison_config(args.config), args.output)
    print(f"Comparison: {output}", flush=True)

if __name__ == "__main__":
    main()
