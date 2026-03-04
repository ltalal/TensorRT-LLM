# Run this script to install TensorRT-LLM with precompiled binaries as editable package
# and install PyTorch 2.9.1 with CUDA 13.0 support

TRTLLM_USE_PRECOMPILED=1 pip install -e .
# pip install -r requirements-dev.txt
# pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
