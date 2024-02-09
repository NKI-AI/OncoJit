import torch
import importlib.util


def load_model(model_path, input_dims=None):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.model  # Assuming the model instance is named 'model' in the .py file
    if input_dims:
        example_input = torch.rand(*input_dims)
        return model, example_input
    return model, None


def jit_model(model, output_path, method="trace", input_dims=None):
    if method == "trace" and input_dims is not None:
        model, example_input = load_model(model, input_dims)
        scripted_model = torch.jit.trace(model, example_input)
    elif method == "script":
        model, _ = load_model(model)
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError("Invalid method or input dimensions not provided for tracing.")

    scripted_model.save(output_path)
    print(f"Model JITed and saved to {output_path}")
