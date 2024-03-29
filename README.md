# OncoJit
**OncoJit - Accelerating Oncology Models with JIT Compilation**

OncoJIT is an innovative deep learning library designed to empower researchers and clinicians in the field of oncology with the speed and efficiency of Just-In-Time (JIT) compilation. By optimizing neural network models specifically for cancer research and clinical applications, OncoJIT enables rapid, real-time inference, facilitating faster diagnostics, personalized treatment planning, and cutting-edge cancer research.

**Installation**:

```
git clone https://github.com/NKI-AI/OncoJit.git

cd OncoJit

pip install -e .
```

**Usage**:

1. Download and install (as an editable package) the repository containing the model which needs to be JIT compiled preferably in a clean conda environment.
2. You can use OncoJit to generate a jit compiled version of the model. Optionally, you can also compile it with the pretrained weights.

Run the following to obtain jit compiled model:

```
oncojit --model_def <path_to_model_definition_in_original_repo> --weights <path to weights file> --model_name <Name of the model> --output_path <Path where you need the jit compiled model to be stored> --input_dims <The input dimensions to the model to generate dummy input> --method <One of "trace" or "script">
```

**Note**:
1. Please note that oncojit assumes the nn.Module which needs jit compilation, loads its weight in the constructor.
2. Since different open source models are written differently and often without regard for types, it may be possible that you end up fixing the original model definition. 

**Key Features**:

**Optimized Performance**: Leverage the power of JIT compilation to enhance the performance of deep learning models, ensuring quicker load times and faster inference with minimal latency.

**Oncology-Specific Models**: Access a curated collection of pre-built and JIT-compiled models tailored for a wide range of oncology applications, from tumor detection to genetic mutation analysis.

 **Flexible Integration**: OncoJIT is designed to seamlessly integrate with existing deep learning and medical imaging pipelines, offering a plug-and-play solution for researchers and developers.

**Open Source Collaboration**: Contribute to and benefit from a growing ecosystem of oncology-focused models and tools, developed in collaboration with leading researchers and institutions worldwide.

**Ease of Use**: Simplify the deployment of complex models with a user-friendly interface and comprehensive documentation, making advanced oncology modeling accessible to a broader audience.

Whether you're conducting advanced cancer research, developing clinical diagnostic tools, or exploring novel therapeutic strategies, OncoJIT provides the technological backbone to accelerate your work and unlock new possibilities in the fight against cancer.
