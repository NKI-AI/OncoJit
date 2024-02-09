import argparse
from .jitter import jit_model
from .io import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="JIT PyTorch Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the JITed model")
    parser.add_argument(
        "--input_dims", type=int, nargs="+", help="Input dimensions for the model, e.g., --input_dims 1 3 224 224"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="Method to convert the model ('trace' or 'script')",
    )
    args = parser.parse_args()

    logger.info(f"JIT-ing model at {args.model_path} with the {args.method} method...")
    jit_model(args.model_path, args.output_path, args.method, args.input_dims)


if __name__ == "__main__":
    main()
