from enum import Enum


class MetricNames(str, Enum):
    TTFT = "ttft"
    TPOT = "tpot"
    E2E = "e2e"
    REQUEST_QUEUE_TIME = "request_queue_time"
    GPU_PREFIX_CACHE_HIT_RATE = "gpu_prefix_cache_hit_rate"
    KV_CACHE_TRANSFER_TIME = "kv_cache_transfer_time"
    SPEC_DECODE_DRAFT_ACCEPTANCE_RATE = "spec_decode_draft_acceptance_rate"
    SPEC_DECODE_ACCEPTED_TOKENS = "spec_decode_num_accepted_tokens_total"
    SPEC_DECODE_DRAFT_TOKENS = "spec_decode_num_draft_tokens_total"


class RequestEventTiming(Enum):
    ARRIVAL_TIME = "arrival_time"
    FIRST_TOKEN_TIME = "first_token_time"  # nosec: B105
    FIRST_SCHEDULED_TIME = "first_scheduled_time"
    LAST_TOKEN_TIME = "last_token_time"  # nosec: B105
    KV_CACHE_TRANSFER_START = "kv_cache_transfer_start"
    KV_CACHE_TRANSFER_END = "kv_cache_transfer_end"
