exec trtllm-serve serve /home/ubuntu/scratch.trt_llm_data_ci/llama-3.1-model/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --tp_size 1 \
  --served_model_name meta-llama/Meta-Llama-3.1-8B-Instruct  \
  --kv_cache_free_gpu_memory_fraction 0.8 --extra_llm_api_options cfg_tests.yaml
