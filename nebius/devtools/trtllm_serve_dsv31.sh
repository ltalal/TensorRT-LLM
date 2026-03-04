exec trtllm-serve serve /mnt/model-storage/nvidia/DeepSeek-V3.1-NVFP4 --host 0.0.0.0 --port 8000 --tp_size 4 \
  --served_model_name nvidia/DeepSeek-V3.1-NVFP4  \
  --extra_llm_api_options cfg_dsv31.yaml
