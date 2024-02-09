import argparse
from pathlib import Path

from oncojit.io import get_logger
from oncojit.jitter import jit_model

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="JIT PyTorch Models CLI")
    parser.add_argument(
        "--model_def", type=str, required=True, help="Path to the Python file containing the model definition."
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Optional: Path to the .pth file containing the model's weights."
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Optional: Name of the model which needs to be JIT compiled."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the JIT-compiled model.")
    parser.add_argument(
        "--input_dims",
        type=int,
        nargs="+",
        help="Input dimensions for the model, e.g., --input_dims 1 3 224 224 for a single image.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="Method to convert the model ('trace' or 'script').",
    )
    args = parser.parse_args()
    jit_model(
        model_definition_path=Path(args.model_def),
        model_weights_path=Path(args.weights),
        model_identifier=args.model_name,
        output_path=Path(args.output_path),
        method=args.method,
        input_dims=args.input_dims if args.input_dims else None,
    )


if __name__ == "__main__":
    main()
