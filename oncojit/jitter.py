import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from oncojit.io import get_logger

logger = get_logger(__name__)


def load_model_with_weights(
    model_def_path: Path,
    model_name: str,
    weights_path: Optional[Path] = None,
    input_dims: Optional[List[int]] = None,
) -> Tuple[torch.nn.Module, Optional[torch.Tensor]]:
    spec = importlib.util.spec_from_file_location("model_module", str(model_def_path))
    if spec is None:
        raise ImportError(f"Could not create a module spec for {model_def_path}")

    model_module = importlib.util.module_from_spec(spec)
    # Since spec is checked to be not None above, loader must be present.
    # However, MyPy or other static analysis tools might not infer this correctly,
    # leading to potential false positives without an explicit None check.
    if spec.loader is not None:
        spec.loader.exec_module(model_module)  # Load the module
    else:
        raise ImportError(f"Could not load the module {model_def_path}")

    model = getattr(model_module, model_name)(weights_path=str(weights_path) if weights_path else None)
    model.eval()  # Set the model to evaluation mode

    example_input = torch.rand(*input_dims) if input_dims else None

    return model, example_input


def jit_model(
    model_definition_path: Path,
    model_identifier: str,
    output_path: Path,
    model_weights_path: Optional[Path] = None,
    method: str = "trace",
    input_dims: Optional[List[int]] = None,
) -> None:
    if method == "trace" and input_dims is not None:
        model, example_input = load_model_with_weights(
            model_definition_path, model_identifier, model_weights_path, input_dims
        )

        scripted_model = torch.jit.trace(model, example_input)  # type: ignore
    elif method == "script":
        model, _ = load_model_with_weights(model_definition_path, model_identifier, model_weights_path)
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError("Invalid method or input dimensions not provided for tracing.")

    scripted_model.save(output_path / f"{model_identifier}_jit_.pt")
    logger.info(f"Model JITed and saved to {output_path}")
