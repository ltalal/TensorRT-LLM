# Eigen Infer vs TensorRT-LLM 1.3.x — DeepSeek-V3.1 Technical Analysis

**Image:** `cr.eu-north1.nebius.cloud/e00x1bqngdhpsezt4g/eigenai/eigeninfer-deepseekv3.1:v6.1.6`
**Package:** `eigen_infer` v2.1.1 by MagicByte AI, Inc. DBA Eigen AI Labs
**Base:** TensorRT-LLM ~1.3.x (C++ namespace `tensorrt_llm::_v1`, TensorRT 10.14.1.48, NGC PyTorch 25.12)

---

> **Note on methodology:** All claims below were verified by direct `diff` of files extracted from the Docker
> container filesystem against this TRT-LLM 1.3.0rc6 source tree. "Identical" means the diff shows only
> namespace substitution (`tensorrt_llm` → `eigen_infer`, `trtllm` → `eiginf`) and copyright header changes.

---

## 1. DeepSeek-V3.1 NVFP4 Support on Blackwell (SM100/SM103)

**All present in stock TRT-LLM 1.3.x.** Eigen changes are namespace-only.

### CUTE DSL Grouped GEMM Kernel
`_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py`

Exists identically in TRT-LLM. Diff: one env-var rename `TRTLLM_ENABLE_PDL` → `EIGINF_ENABLE_PDL`.

### FP4 Quantization Types
`quantization/utils/fp4_utils.py`

Exists identically in TRT-LLM. `FP4GemmType`, `reorder_rows_for_gated_act_gemm()`, `block_scale_interleave()`, Triton dequantization kernels — all present. Diff: `torch.ops.trtllm.*` → `torch.ops.eiginf.*`.

### CUTE DSL Auto-Tuner
`_torch/custom_ops/cute_dsl_custom_ops.py`

Exists identically in TRT-LLM (`CuteDSLNVFP4BlackwellRunner`). Diff: namespace only.

### Actual Differences in This Area

- `modeling_deepseekv3.py`: Eigen changes `moe_backend == 'TRTLLM'` check to `'EIGINF'`; replaces `max_total_draft_tokens + 1` with `spec_config.tokens_per_gen_step` (API change); adds `cp_allgather` import; uses in-place `shared_output.add_(routed_output)` instead of `shared_output + routed_output` to reduce peak memory; fixes mixed quantization for shared experts (adds `_get_shared_experts_quant_config`).

---

## 2. KV Cache

**All present in stock TRT-LLM 1.3.x.** Eigen changes are namespace/comment-only.

### Block Radix Tree with SHA-256 Hashing
`runtime/kv_cache_manager_v2/_block_radix_tree.py`

Exists identically in TRT-LLM. `Hasher` (SHA-256), `sequence_to_blockchain_keys`, hierarchical block keys, LoRA-aware hashing — all present. Diff: TODO ticket numbers renamed (`TRTLLM-7784` → `EIGINF-7784`).

### Committed/Uncommitted Page Lifecycle
`runtime/kv_cache_manager_v2/_core/_kv_cache.py`

Exists in TRT-LLM. No substantive diff.

### KV Cache Types
`llmapi/kv_cache_type.py`

Identical. `KVCacheType` enum (CONTINUOUS, PAGED, DISABLED) unchanged.

---

## 3. Speculative Decoding

**TRT-LLM 1.3.x already has 10 modes.** Eigen adds 3.

### Modes in TRT-LLM 1.3.x (already present)

```python
class SpeculativeDecodingMode(IntEnum):
    MTP = auto()
    MTP_EAGLE = auto()
    MTP_EAGLE_ONE_MODEL = auto()
    EAGLE3 = auto()
    EAGLE3_ONE_MODEL = auto()
    NGRAM = auto()
    DRAFT_TARGET = auto()
    USER_PROVIDED = auto()
    SAVE_HIDDEN_STATES = auto()
    AUTO = auto()
    NONE = auto()
```

`NGramPoolManager` (`_torch/speculative/ngram.py`) and EAGLE3 resource manager (`_torch/speculative/eagle3.py`) also exist in TRT-LLM.

### Modes Added by Eigen (genuinely new)

```python
    SA = auto()                     # Suffix Automaton — hybrid MTP+SA drafting
    DRAFT_TARGET_ONE_MODEL = auto() # Single-engine draft/target
    PARD = auto()                   # Parallel Auto-Regressive Decoding
```

- `_torch/speculative/suffix_automaton.py` — `SuffixAutomatonManager`, not present in TRT-LLM
- `_torch/speculative/pard.py` — `PARDWorker`, `PARDSpecMetadata`, not present in TRT-LLM
- SA mode is a hybrid: MTP drafts candidates, Suffix Automaton replaces or supplements them; integrated into `mtp.py` via `sa_manager` field

### Other Speculative Decoding Differences

- `eagle3.py`: `pin_memory=True` → `pin_memory=prefer_pinned()` (6 sites); `config.max_total_draft_tokens` → `config.tokens_per_gen_step - 1`
- `mtp.py`: namespace (`trtllm` → `eiginf`); SA hybrid path; `prefer_pinned()`; `tokens_per_gen_step` API

---

## 4. Prefix Caching

**Present in stock TRT-LLM 1.3.x.** Covered by the Block Radix Tree (see §2). No differences.

---

## 5. CUDA Kernels

**All three `.so` libraries and the fused MoE op are present in stock TRT-LLM 1.3.x.**

### DeepGEMM (`deep_gemm_cpp_tllm.so`)
Present in TRT-LLM at `tensorrt_llm/deep_gemm_cpp_tllm.cpython-312-x86_64-linux-gnu.so`.
The Python wrapper module `tensorrt_llm/deep_gemm/` is also present.

### DeepEP (`deep_ep_cpp_tllm.so`)
Present in TRT-LLM at `tensorrt_llm/deep_ep_cpp_tllm.cpython-312-x86_64-linux-gnu.so`.
Python modules: `_torch/modules/fused_moe/communication/deep_ep.py`, `deep_ep_low_latency.py`.

### FlashMLA (`flash_mla_cpp_tllm.so`)
Present in TRT-LLM at `tensorrt_llm/flash_mla_cpp_tllm.cpython-312-x86_64-linux-gnu.so`.
Python interface: `tensorrt_llm/flash_mla/flash_mla_interface.py`.

### Fused MoE Custom Op
`_torch/modules/fused_moe/interface.py` — `moe_custom_op` exists in TRT-LLM as `trtllm::moe_custom_op`.
Eigen renames it to `eiginf::moe_custom_op`. Logic is identical.

---

## 6. Model Sparsity

**Both backends present in stock TRT-LLM 1.3.x.** Differences are minor.

### DSA (Deep Sparse Attention)
`_torch/attention_backend/sparse/dsa.py`

Present in TRT-LLM. Eigen differences (53 lines):
- Class renamed `DSATrtllmAttention` → `DSAEigInfAttention`, base class `TrtllmAttention` → `EigInfAttention`
- `pin_memory=True` → `pin_memory=prefer_pinned()` (11 sites)
- **New guard:** null-check for `kv_cache_manager` before accessing `index_head_dim`/`tokens_per_block`/`quant_block_size`, to handle MTP speculative decoding where DSA metadata is used with a regular (non-DSA) draft KV cache manager

### ROCKET (Window + Top-K Hybrid)
`_torch/attention_backend/sparse/rocket.py`

Present in TRT-LLM. Eigen differences (18 lines):
- Class renamed `RocketTrtllmAttentionMetadata` → `RocketEigInfAttentionMetadata`, `RocketTrtllmAttention` → `RocketEigInfAttention`
- `pin_memory=True` → `pin_memory=prefer_pinned()` (2 sites)
- No algorithmic changes

### Triton Kernels
`_torch/attention_backend/sparse/kernel.py` — **zero substantive diff lines**. Identical.

---

## 7. Parallelization Strategies

**All present in stock TRT-LLM 1.3.x.** `mapping.py` diff is copyright header only.

### Mapping Dimensions
`MappingBase` with `cp_size`, `moe_ep_size`, `moe_cluster_size`, `attn_tp_size`, `attn_cp_size` — identical in TRT-LLM.

### Context Parallel Types
`CpType` enum (ULYSSES, STAR, RING, HELIX) — identical in TRT-LLM.

### STAR Attention
`_torch/attention_backend/star_flashinfer.py` — present in TRT-LLM; diff is namespace only (6 lines).

### Expert Parallel Communication Modes
`AlltoallMethodType` (NVLinkOneSided, NVLinkTwoSided, DeepEP, DeepEPLowLatency) — identical in TRT-LLM.

---

## Summary

| Dimension | Claimed as Eigen-only | Reality |
|---|---|---|
| DeepSeek NVFP4 Blackwell CUTE DSL kernels | Eigen addition | **Present in TRT-LLM; identical** |
| KV cache Block Radix Tree + SHA-256 | Eigen addition | **Present in TRT-LLM; identical** |
| Speculative decoding (11 modes) | "EAGLE only" in stock | **TRT-LLM has 10 modes; Eigen adds 3 (SA, DRAFT_TARGET_ONE_MODEL, PARD)** |
| Prefix caching | Eigen addition | **Present in TRT-LLM; identical** |
| DeepGEMM / DeepEP / FlashMLA `.so` | "Absent from stock" | **Present in TRT-LLM** |
| Fused MoE custom op | Eigen addition | **Present in TRT-LLM as `trtllm::moe_custom_op`** |
| DSA + ROCKET sparse attention | "Stock has none" | **Present in TRT-LLM; Eigen renames classes + `prefer_pinned()` + one null-guard** |
| TP/PP + CP + EP parallelism | Eigen extension | **Present in TRT-LLM; `mapping.py` identical** |

### What Eigen Actually Adds

| Item | Description |
|------|-------------|
| `SA` mode | Suffix Automaton speculative decoding (`suffix_automaton.py`), MTP+SA hybrid |
| `PARD` mode | Parallel Auto-Regressive Decoding (`pard.py`, `PARDWorker`) |
| `DRAFT_TARGET_ONE_MODEL` mode | Single-engine draft/target variant |
| `prefer_pinned()` | Conditional `pin_memory` helper; replaces hardcoded `True` across speculative/sparse code |
| DSA null-guard | Handles DSA metadata with non-DSA draft KV cache manager (MTP + DSA combination) |
| `modeling_deepseekv3.py` fixes | In-place MoE output add; `tokens_per_gen_step` API; mixed quant for shared experts; `cp_allgather` |
