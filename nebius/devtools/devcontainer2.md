# TensorRT-LLM Devcontainer Guide (Nebius)

This guide describes how to run the TensorRT-LLM development container and set up an editable wheel build for development.

## Prerequisites

- **Docker** with GPU support (NVIDIA Container Toolkit)
- **VS Code** or **Cursor** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- **NVIDIA GPU** for building and running models

## Running the Devcontainer

### 1. Prepare the environment

Before opening the devcontainer, the project runs an initialization step:

```bash
cd /path/to/tensorrt_llm
./.devcontainer/make_env.py
```

This script:
- Selects a suitable Docker image (NGC, Jenkins, or builds locally)
- Generates `.devcontainer/.env` for Docker Compose
- Ensures `docker-compose.override.yml` exists (copied from example if missing)

### 2. Open in devcontainer

1. Open the project folder in VS Code or Cursor
2. Run **Dev Containers: Reopen in Container** (Cmd/Ctrl+Shift+P)
3. Or use the prompt to reopen when opening the folder

The container will:
- Start with your workspace at `/workspaces/tensorrt_llm`
- Run `postCreateCommand`: install `requirements-dev.txt`, set up pre-commit
- Mount the host `/mnt` and `~/.cache/huggingface` (see `docker-compose.override.yml`)

### 3. Volume mounts (optional)

Edit `.devcontainer/docker-compose.override.yml` to add or adjust mounts:

```yaml
volumes:
  - /mnt:/mnt                                    # Host storage
  - ${HOME}/.cache/huggingface:/home/ubuntu/.cache/huggingface  # HF cache
  # - /path/to/models:/home/scratch.trt_llm_data_ci/:ro       # Models (uncomment if needed)
```

## Build with Editable Wheel

An editable install lets Python source changes take effect without rebuilding. The C++ bindings must be built once.

### Step 1: Build the C++ bindings (one time)

From the project root inside the container:

```bash
python3 scripts/build_wheel.py \
  --cuda_architectures native \
  --no-venv \
  --fast_build \
  -G Ninja \
  --use_ccache
```

**Flags:**
- `--cuda_architectures native` — build for current GPU
- `--no-venv` — use system Python
- `--fast_build` — faster, dev-oriented build
- `-G Ninja` — use Ninja
- `--use_ccache` — reuse compilation output

Build time: typically 30–60+ minutes for a full build.

### Step 2: Install in editable mode

After the build, C++ artifacts (including `bindings*.so`) are in the source tree. Install the package in editable mode:

```bash
./.devcontainer/install_wheel.sh
```

Or directly:

```bash
pip install -q -e ".[devel]"
```

### Step 3: Verify

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__file__)"
```

The path should point at your workspace (e.g. `/workspaces/tensorrt_llm/tensorrt_llm/__init__.py`).

### Workflow

| Change type   | Action                                      |
|---------------|----------------------------------------------|
| Python only   | None — edits apply immediately               |
| C++ code      | Rebuild with `build_wheel.py`, then reinstall |

## Running Tests

Ensure `tensorrt_llm` is installed (editable or wheel) and add `trtllm-build` to PATH:

```bash
export PATH=$HOME/.local/bin:$PATH
```

### Unit tests

From the project root:

```bash
cd tests

# Run all unit tests (can take a long time)
pytest ./

# Run a specific test file
pytest unittest/test_builder.py

# Run tests matching a keyword
pytest -k test_basic_builder_flow -v

# Run tests in a directory
pytest unittest/functional/
```

### OpenAPI / openai_server tests

These tests spawn a server and need a model. Set up once:

```bash
export LLM_MODELS_ROOT=/workspaces/tensorrt_llm/.llm-models
```

Download TinyLlama (if `.llm-models` is empty):

```bash
cd /workspaces/tensorrt_llm
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                  local_dir='.llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0')
"
```

Run openai_server tests:

```bash
cd tests
pytest unittest/llmapi/apps/ -v

# Single test file
pytest unittest/llmapi/apps/_test_openai_chat.py -v

# Filter by test name
pytest unittest/llmapi/apps/ -k "test_single_chat_session" -v
```

### Integration tests

Integration tests require models in `LLM_MODELS_ROOT`. See [tests/README.md](../tests/README.md) for the expected directory layout.

```bash
export LLM_MODELS_ROOT=/path/to/your/models
cd tests/integration/defs

# Run a single test
pytest "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_auto_dtype"

# List available tests
pytest --co -q

# Run tests matching a keyword
pytest -k llmapi --co -q
```

## Troubleshooting

### pip cache warning

If you see:

```
WARNING: The directory '/home/ubuntu/.cache/pip' or its parent directory is not owned or is not writable
```

Create the cache directory:

```bash
mkdir -p ~/.cache/pip
```

### "No module named 'tensorrt_llm'"

The package is not installed. Either:

1. Complete Step 1 and Step 2 above (build + editable install), or
2. Build and install the wheel: `pip install -q build/tensorrt_llm*.whl` (non-editable)

### "No tensorrt_llm bindings found"

Run `build_wheel.py` before `install_wheel.sh`. Editable install needs the compiled bindings in the source tree.

### trtllm-build not found

Add user binaries to PATH:

```bash
export PATH=$HOME/.local/bin:$PATH
```
