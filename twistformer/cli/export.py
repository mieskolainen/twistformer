# Saved-model ONNX export steering
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
import argparse
from twistformer.api.export import export_onnx

def main(argv=None):
    """
    Export a checkpoint with fitted normalization and physical task outputs

    Args:
        argv:  Command-line arguments, or None for process arguments
    Returns:
        None: Exported model path is printed
    """
    parser = argparse.ArgumentParser(description="Export saved inference to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    print(export_onnx(args.checkpoint, args.output, max_tokens=args.max_tokens, opset=args.opset, verify=not args.no_verify))

if __name__ == "__main__":
    main()
