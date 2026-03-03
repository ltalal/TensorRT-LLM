# Python-Only Development in Devcontainer

This guide describes how to work with TensorRT-LLM when you only need to modify Python code, without building the C++ components. This significantly speeds up development by skipping the full compilation (which can take 30+ minutes).

## Prerequisites

- Devcontainer environment (see `.devcontainer/`)
- No C++ changes required

## Setup: Use Precompiled Binaries

### 1. Set the Version to Match an Existing Wheel

**Important:** The precompiled workflow downloads a wheel whose version must match what is specified in `tensorrt_llm/version.py`. You must set the version to one that has a published wheel available.

Edit `tensorrt_llm/version.py` and set `__version__` to an existing wheel version, for example:

```python
__version__ = "1.3.0rc3"
```

Check [PyPI](https://pypi.org/project/tensorrt-llm/#history) or your internal wheel registry for available versions. The version in `version.py` must exactly match a wheel that exists (e.g. `tensorrt_llm-1.3.0rc3-cp312-cp312-linux_x86_64.whl`).

### 2. Install Matching PyTorch Version

The precompiled wheel is built with PyTorch 2.9.1. The devcontainer may use a different PyTorch (e.g. NVIDIA 2.10.0a0), which causes ABI incompatibility. Install the matching version:

```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. Install with Precompiled Binaries

```bash
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

This will:

- Download the prebuilt wheel for the version in `version.py`
- Extract compiled libraries (bindings, plugins, etc.) into your workspace
- Install the package in editable mode so your Python changes take effect immediately

### 4. Install Dev Dependencies (if not already done)

```bash
pip install -r requirements-dev.txt
```

## Alternative: Specify Wheel Explicitly

If the default download fails or you need a specific wheel, use `TRTLLM_PRECOMPILED_LOCATION`:

```bash
TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.3.0rc3-cp312-cp312-linux_x86_64.whl pip install -e .
```

Or with a local wheel file:

```bash
TRTLLM_PRECOMPILED_LOCATION=/path/to/tensorrt_llm-1.3.0rc3-cp312-cp312-linux_x86_64.whl pip install -e .
```

When using `TRTLLM_PRECOMPILED_LOCATION`, the version in `version.py` is ignored for the download; the wheel you specify is used directly.

## Running Tests

Tests require model files. Download minimal models first:

```bash
python nebius/download_minimal_models.py  # this requires HF_TOKEN to be set
export LLM_MODELS_ROOT=$(pwd)/llm-models  # or the path printed by the script
```

Then run the OpenAI server tests:

```bash
cd tests
pytest unittest/llmapi/apps/ -k openai -v
```

## When You Already Have a Build

If you have previously built TensorRT-LLM (C++ bindings exist), you can simply reinstall to pick up Python changes:

```bash
pip install -e .
```

No version change or precompiled setup is needed.

## Known Limitations

1. **Version mismatch:** The precompiled wheel must be built from C++ code compatible with your branch. Using a wheel from a different commit may cause compatibility issues.

2. **C++ changes:** If you modify C++ code, you must do a full build with `scripts/build_wheel.py`; precompiled binaries will not reflect your changes.

3. **Wheel availability:** Not all versions have published wheels. Ensure the version in `version.py` corresponds to an existing wheel before running `TRTLLM_USE_PRECOMPILED=1 pip install -e .`.
