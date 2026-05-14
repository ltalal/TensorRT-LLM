#!/usr/bin/env bash
# Force imports from the git checkout: a wheel under /usr/local/lib/python3.12/dist-packages
# can shadow an editable install in ~/.local when user site is disabled or ordering differs.
set -euo pipefail
export PYTHONPATH=/workspaces/tensorrt_llm:${PYTHONPATH:-}
export TLLM_NUMA_AWARE_WORKER_AFFINITY=1
export MODEL_PATH=/mnt/model-storage/Cursor/Kimi-K2.5-NVFP4-TRTLLM-MTPFP4-amaxfix-kvscale-1276sample-20260429
exec trtllm-serve serve $MODEL_PATH --host 0.0.0.0 --port 8000 \
  --chat_template $MODEL_PATH/chat_template_for_safe_tokenization_v2.jinja \
  --tool_parser multiline \
  --reasoning_parser deepseek-r1 \
  --served_model_name Kimi-K2.5  \
  --extra_llm_api_options cfg_kimi_cursor.yaml
