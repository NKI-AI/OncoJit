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
    """
    Loads a PyTorch model from a specified definition file and optionally initializes it with weights.

    This function dynamically imports a Python module that defines a PyTorch model from the given file path,
    instantiates the model by name, and optionally loads it with weights from a specified path. If input
    dimensions are provided, it also generates a random tensor matching these dimensions, which can be used
    for testing or inference purposes.

    Parameters
    ----------
        model_def_path (Path): The file system path to the Python file defining the model class.
        model_name (str): The name of the model class to be instantiated.
        weights_path (Optional[Path]): The file system path to the model's weights. If provided, the model
            will be loaded with these weights. Defaults to None.
        input_dims (Optional[List[int]]): The dimensions of the input tensor to be generated for testing or
            inference. If provided, a tensor with these dimensions will be returned. Defaults to None.

    Returns
    -------
        Tuple[torch.nn.Module, Optional[torch.Tensor]]: A tuple containing the loaded model and an example
        input tensor. The input tensor is None if `input_dims` is not provided.

    Raises
    ------
        ImportError: If the module defined by `model_def_path` cannot be imported or if the specified
            model class cannot be found within the module.
    """
    spec = importlib.util.spec_from_file_location("model_module", str(model_def_path))
    if spec is None:
        raise ImportError(f"Could not create a module spec for {model_def_path}")

    model_module = importlib.util.module_from_spec(spec)
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
    """
    Compiles a PyTorch model using JIT (Just-In-Time) compilation for optimization.

    This function supports either tracing or scripting methods for JIT compilation. For tracing, input dimensions
    must be provided to generate an example input tensor. Scripting does not require example inputs and works
    directly with the model's code.

    Parameters
    ----------
    model_definition_path : Path
        The file system path to the Python file defining the model class.
    model_identifier : str
        The name of the model class to be instantiated and JIT compiled.
    output_path : Path
        The directory path where the JIT compiled model will be saved.
    model_weights_path : Optional[Path], optional
        The file system path to the model's weights, by default None. If provided, the model will be loaded
        with these weights before JIT compilation.
    method : str, optional
        The method of JIT compilation to use. Can be either "trace" or "script", by default "trace".
    input_dims : Optional[List[int]], optional
        The dimensions of the input tensor for tracing, by default None. Required if the method is "trace".

    Raises
    ------
    ValueError
        If an invalid method is provided or if input dimensions are not provided when required for tracing.

    Notes
    -----
    The compiled model is saved with the filename pattern `<model_identifier>_jit_.pt` in the specified `output_path`.
    Logging is used to inform the user about the success of the operation.

    Examples
    --------
    >>> jit_model(Path("/path/to/model.py"), "MyModel", Path("/path/to/output"),
    ...           Path("/path/to/weights.pth"), method="trace", input_dims=[1, 3, 224, 224])
    Model JITed and saved to /path/to/output
    """
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
