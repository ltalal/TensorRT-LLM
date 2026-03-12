# Eigen Infer — Entrypoint & Frontend Analysis

**Image:** `cr.eu-north1.nebius.cloud/e00x1bqngdhpsezt4g/eigenai/eigeninfer-deepseekv3.1:v6.1.6`

---

## 1. Container Entrypoint

```
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD        ["python3", "/opt/eigen/launcher.py"]
```

The NVIDIA entrypoint script is standard NGC boilerplate (EULA check, environment setup). The actual engine launcher is `/opt/eigen/launcher.py`.

---

## 2. Launcher (`/opt/eigen/launcher.py`)

A custom startup script — not `trtllm-serve`. `trtllm-serve` uses a Click CLI entrypoint; this launcher has no TRT-LLM equivalent.

### Hardcoded Configuration
```python
VERSION = "6.1.6"
MODEL   = "/root/DeepSeekV3.1-Eigen"   # hardcoded model path
```
Model path is baked in; no CLI flag accepted.

### Engine Arguments

Built via `_build_llm_args()` using `DEFAULT_*` constants compiled into `eigen_infer/llmapi/_defaults.cpython-312-x86_64-linux-gnu.so` (Cython, COMPRESS_STRINGS). All parameters are standard TRT-LLM `LlmArgs` fields; only the baked-in defaults are Eigen-specific:

| Parameter | DEFAULT value (recovered from binary) |
|-----------|---------------------------------------|
| `backend` | `"pytorch"` |
| `tensor_parallel_size` | runtime (GPU count) |
| `moe_expert_parallel_size` | runtime |
| `max_batch_size` | 1024 |
| `max_num_tokens` | 81920 |
| `max_seq_len` | 8192 |
| `trust_remote_code` | `True` |
| `enable_chunked_prefill` | `True` |
| `kv_cache_config.dtype` | `"auto"` |
| `kv_cache_config.free_gpu_memory_fraction` | `0.9` |
| `moe_config.backend` | `"EIGINF"` (custom fused MoE op) |
| `moe_config.use_low_precision_moe_combine` | `True` |
| `nvfp4_gemm_config.allowed_backends` | `["cublaslt", "cuda_core", "cutlass", "fp8"]` |
| `speculative_config.decoding_type` | `"MTP"` |
| `speculative_config.num_nextn_predict_layers` | N (model-specific) |
| `speculative_config.use_relaxed_acceptance_for_thinking` | `True` |
| `speculative_config.relaxed_topk` | N |
| `speculative_config.relaxed_delta` | `0.4` |

**Note:** `cutedsl` (CUTE DSL SM100 kernel) is NOT in `allowed_backends` by default.

### Network Binding
```python
host = os.environ.get("EIGEN_HOST", "0.0.0.0")
port = int(os.environ.get("EIGEN_PORT", "8003"))
```

### Startup Sequence
1. Suppress all logging (C++ fd-level redirect + Python logger muting)
2. Import `eigen_infer.LLM` and `OpenAIServer` (suppressed output)
3. Call `PyTorchLLM(**llm_args)` with animated spinner (background thread writes to duplicated fd)
4. Create `OpenAIServer(generator=llm, model=model)`
5. Bind socket, **disable Python GC** (`gc.disable()`), run `asyncio.run(server(...))`

GC disable after model load is Eigen-specific; `trtllm-serve` does not do this.

---

## 3. OpenAI Server (`eigen_infer/serve/openai_server.py`)

Built on **FastAPI + uvicorn**. The implementation is a near-verbatim copy of `tensorrt_llm/serve/openai_server.py` with namespace substitution (`tensorrt_llm` → `eigen_infer`, `trtllm/` → `eiginf/` ETCD key prefix).

### Routes

All routes below exist identically in TRT-LLM 1.3.x:

| Path | Method | Description |
|------|--------|-------------|
| `/v1/completions` | POST | Text completions |
| `/v1/chat/completions` | POST | Chat completions (Harmony if model is `gpt_oss` type) |
| `/v1/responses` | POST | Responses API |
| `/v1/responses/{id}` | GET/DELETE | Responses API CRUD |
| `/v1/models` | GET | List available models |
| `/health`, `/health_generate` | GET | Liveness probes |
| `/version` | GET | Server version |
| `/metrics` | GET | Prometheus metrics |
| `/steady_clock_offset` | GET/POST | NTP-style clock sync for disagg serving |
| `/kv_cache_events` | POST | KV cache block table export |
| `/server_info` | GET | Server metadata |
| `/release_memory`, `/resume_memory`, `/update_weights` | POST | Runtime control |
| `/v1/images/generations`, `/v1/images/edits` | POST | Visual gen (VisualGen mode) |
| `/v1/videos`, `/v1/videos/{id}`, `/v1/videos/{id}/content` | GET/POST/DELETE | Video gen |
| `/v1/chat/completions` (multimodal mode) | POST | Multimodal encoder path |

### Service Discovery (ETCD)
On startup, registers `eiginf/{llm_id}` (vs `trtllm/{llm_id}` in upstream). Logic identical.

### Harmony Adapter (`eigen_infer/serve/harmony_adapter.py`)
**Identical to TRT-LLM.** Diff shows only 2 import lines differ. Integrates `openai_harmony` for structured reasoning, maintaining per-request `HarmonyStreamState` with `analysis`/`final` channels. Activated when `model_config.model_type == "gpt_oss"`.

### Tool Parsers (`eigen_infer/serve/tool_parser/`)
**Identical to TRT-LLM.** Zero substantive diff lines across all 5 parsers:

| File | Model |
|------|-------|
| `deepseekv31_parser.py` | DeepSeek-V3.1 |
| `deepseekv32_parser.py` | DeepSeek-V3.2 |
| `kimi_k2_tool_parser.py` | Kimi-K2 |
| `qwen3_tool_parser.py` | Qwen3 |
| `qwen3_coder_parser.py` | Qwen3-Coder |

---

## 4. Router (`eigen_infer/serve/router.py`)

**Identical to TRT-LLM.** The full diff shows only 7 changed lines — all import namespaces and the ETCD key prefix (`trtllm/` → `eiginf/`). All three router classes, all algorithms, and `create_router` factory are verbatim copies:

- `RoundRobinRouter` — cyclic index
- `LoadBalancingRouter` — min-heap on active request/token count
- `KvCacheAwareRouter` — scores servers by `matched_tokens/total - active_requests/max_batch_size`, polling `/kv_cache_events` on request completion

Dynamic server monitoring (ETCD poll loop, double health-check before marking dead, `_filter_servers_by_role`) is identical.

---

## 5. Disaggregated Serving (`eigen_infer/serve/openai_disagg_server.py`)

**Identical to TRT-LLM.** Diff shows one non-import difference: tracer name `"trt.llm"` is unchanged (not rebranded). `OpenAIDisaggServer`, `DisaggPerfMetricsCollector`, NTP clock sync (`_sync_server_clock`), ctx/gen router separation — all verbatim.

---

## 6. Actual Differences vs TRT-LLM

Only the following substantive code differences exist in the serve layer:

| Item | TRT-LLM | Eigen Infer |
|------|---------|-------------|
| **Launcher** | Click CLI (`trtllm-serve`) | Hardcoded `launcher.py`, no CLI args |
| **GC disable** | Not present | `gc.disable()` after model load |
| **`_is_visual_gen`** | Set from `isinstance(generator, VisualGen)` | Hardcoded `False`; visual gen imports wrapped in `try/except` |
| **Stuck request tracker** | `stuck_requests_tracker` dict + 60s threshold | Removed |
| **Prometheus JSON file** | Writes `/dev/shm/prom_metrics.json` | Removed |
| **Video format support** | `.mp4` only | `.mp4` + `.avi` fallback via `resolve_video_format` |
| **Media storage env var** | `TRTLLM_MEDIA_STORAGE_PATH` | `EIGINF_MEDIA_STORAGE_PATH` |
| **KV cache timing env var** | `TRTLLM_KVCACHE_TIME_OUTPUT_PATH` | `EIGINF_KVCACHE_TIME_OUTPUT_PATH` |
| **ETCD key prefix** | `trtllm/{llm_id}` | `eiginf/{llm_id}` |
| **Default port** | `8000` (CLI `--port`) | `8003` (`EIGEN_PORT` env var) |
| **Config source** | CLI flags / YAML | Compiled defaults in `_defaults.so` |
