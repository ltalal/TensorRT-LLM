# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for Prometheus Metrics Collection."""

import time
from typing import Dict, Optional, Union

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    """
    Collects and logs metrics from TensorRT-LLM engine stats and request performance metrics to Prometheus.

    Used by OpenAIServer in tensorrt_llm/serve/openai_server.py.

    Args:
        labels: A key-value dictionary of labels to add as metadata to all created Prometheus metrics. Useful for
        distinguishing between multiple series of the same metric name. Example:
        {"model_name": "nemotron-nano-3", "engine_type": "trtllm"}

    Created Prometheus metrics:
        trtllm_request_success_total
        trtllm_e2e_request_latency_seconds
        trtllm_time_to_first_token_seconds
        trtllm_time_per_output_token_seconds
        trtllm_request_queue_time_seconds
        trtllm_kv_cache_hit_rate
        trtllm_kv_cache_utilization
    """
    labelname_finish_reason = "finished_reason"

    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Counter, Gauge, Histogram
        self.last_log_time = time.time()
        self.labels = labels
        self.metric_prefix = "trtllm_"

        self.finish_reason_label = {
            MetricsCollector.labelname_finish_reason: "unknown"
        }
        self.labels_with_finished_reason = {
            **self.labels,
            **self.finish_reason_label
        }

        self.num_requests_running = Gauge(
            name="num_requests_running",
            documentation="Number of requests currently being processed.",
            labelnames=self.labels.keys()).labels(**self.labels)

        self.num_requests_waiting = Gauge(
            name="num_requests_waiting",
            documentation="Number of requests currently being processed.",
            labelnames=self.labels.keys()).labels(**self.labels)

        self.counter_request_success = Counter(
            name=self.metric_prefix + "request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=self.labels_with_finished_reason.keys())

        self.histogram_e2e_time_request = Histogram(
            name=self.metric_prefix + "e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.histogram_time_to_first_token = Histogram(
            name=self.metric_prefix + "time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.histogram_time_per_output_token = Histogram(
            name=self.metric_prefix + "time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.histogram_queue_time_request = Histogram(
            name=self.metric_prefix + "request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.histogram_gpu_prefix_cache_hit_rate = Histogram(
            name="gpu_prefix_cache_hit_rate",
            documentation=
            "Histogram of GPU prefix cache hit rate as a ratio (0.0 to 1.0).",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.histogram_kv_cache_transfer_time = Histogram(
            name="kv_cache_transfer_time_seconds",
            documentation="Histogram of KV cache transfer time in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.spec_decode_draft_acceptance_rate = Histogram(
            name="spec_decode_draft_acceptance_rate",
            documentation="Speculative decoding draft acceptance rate.",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labelnames=self.labels.keys()).labels(**self.labels)

        self.spec_decode_num_accepted_tokens = Counter(
            name="spec_decode_num_accepted_tokens",
            documentation=
            "Total number of accepted tokens in speculative decoding.",
            labelnames=self.labels.keys()).labels(**self.labels)

        self.spec_decode_num_draft_tokens = Counter(
            name="spec_decode_num_draft_tokens",
            documentation=
            "Total number of draft tokens in speculative decoding.",
            labelnames=self.labels.keys()).labels(**self.labels)

        self.kv_cache_hit_rate = Gauge(name=self.metric_prefix +
                                       "kv_cache_hit_rate",
                                       documentation="KV cache hit rate",
                                       labelnames=self.labels.keys())
        self.kv_cache_utilization = Gauge(name=self.metric_prefix +
                                          "kv_cache_utilization",
                                          documentation="KV cache utilization",
                                          labelnames=self.labels.keys())

        # Metrics used by read_metrics_file in openai_server
        self.generation_tokens_total = Gauge(
            name="generation_tokens_total",
            documentation="Total number of generated tokens.",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.prompt_tokens_total = Gauge(
            name="prompt_tokens_total",
            documentation="Total number of prompt tokens.",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.cpu_mem_usage = Gauge(
            name="cpu_mem_usage",
            documentation="CPU memory usage in bytes",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_mem_usage = Gauge(
            name="gpu_mem_usage",
            documentation="GPU memory usage in bytes",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.num_iterations_total = Gauge(
            name="num_iterations_total",
            documentation="Number of iterations executed",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.num_active_requests = Gauge(
            name="num_active_requests",
            documentation="Number of active requests",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.num_queued_requests = Gauge(
            name="num_queued_requests",
            documentation="Number of queued requests",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_usage_perc = Gauge(
            name="gpu_cache_usage_perc",
            documentation="Percentage of used cache blocks",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_free = Gauge(
            name="gpu_cache_blocks_free",
            documentation="Number of free blocks in KV cache",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_used = Gauge(
            name="gpu_cache_blocks_used",
            documentation="Number of used blocks in KV cache",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_reused_total = Gauge(
            name="gpu_cache_blocks_reused_total",
            documentation="Total number of reused blocks in KV cache",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_missed_total = Gauge(
            name="gpu_cache_blocks_missed_total",
            documentation="Total number of missed blocks in KV cache",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_max = Gauge(
            name="gpu_cache_blocks_max",
            documentation="Total number of blocks in KV cache",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_alloc_total = Gauge(
            name="gpu_cache_blocks_alloc_total",
            documentation="Total number of blocks allocated",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.gpu_cache_blocks_alloc_new_total = Gauge(
            name="gpu_cache_blocks_alloc_new_total",
            documentation="Total number of new blocks allocated",
            labelnames=self.labels.keys()).labels(**self.labels)
        self.conf_kv_tokens_per_block = Gauge(
            name="conf_kv_tokens_per_block",
            documentation="Size of block in KV cache in tokens",
            labelnames=self.labels.keys()).labels(**self.labels)

    def _label_merge(self, labels: Dict[str, str]) -> Dict[str, str]:
        if labels is None or len(labels) == 0:
            return self.labels
        return {**self.labels, **labels}

    def _log_counter(self, counter, labels: Dict[str, str],
                     data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self._label_merge(labels)).inc(data)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        # Convenience function for logging to histogram.
        histogram.observe(data)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge (labeled gauges).
        gauge.labels(**self.labels).set(data)

    def log_request_success(self, data: Union[int, float],
                            labels: Dict[str, str]) -> None:
        self._log_counter(self.counter_request_success, labels, data)
        self.last_log_time = time.time()

    def log_request_metrics_dict(
            self, metrics_dict: Optional[dict[str, float]]) -> None:
        """
        Log per-request metrics from TRTLLM engine responses.

        Merges log_metrics_dict and log_histogram: handles finish_reason + counter,
        all histograms (e2e, ttft, tpot, request_queue_time, gpu_prefix_cache_hit_rate,
        kv_cache_transfer_time), and spec_decode metrics.
        """
        if metrics_dict is None:
            return
        if finish_reason := metrics_dict.get(
                MetricsCollector.labelname_finish_reason):
            self._log_counter(
                self.counter_request_success,
                {MetricsCollector.labelname_finish_reason: finish_reason}, 1)
        if e2e := metrics_dict.get(MetricNames.E2E, 0):
            self._log_histogram(self.histogram_e2e_time_request, e2e)
        if ttft := metrics_dict.get(MetricNames.TTFT, 0):
            self._log_histogram(self.histogram_time_to_first_token, ttft)
        if tpot := metrics_dict.get(MetricNames.TPOT, 0):
            self._log_histogram(self.histogram_time_per_output_token, tpot)
        if request_queue_time := metrics_dict.get(
                MetricNames.REQUEST_QUEUE_TIME, 0):
            self._log_histogram(self.histogram_queue_time_request,
                                request_queue_time)
        if gpu_prefix_cache_hit_rate := metrics_dict.get(
                MetricNames.GPU_PREFIX_CACHE_HIT_RATE):
            self._log_histogram(self.histogram_gpu_prefix_cache_hit_rate,
                                gpu_prefix_cache_hit_rate)
        if kv_cache_transfer_time := metrics_dict.get(
                MetricNames.KV_CACHE_TRANSFER_TIME, 0):
            self._log_histogram(self.histogram_kv_cache_transfer_time,
                                kv_cache_transfer_time)
        if spec_decode_draft_acceptance_rate := metrics_dict.get(
                MetricNames.SPEC_DECODE_DRAFT_ACCEPTANCE_RATE):
            self.spec_decode_draft_acceptance_rate.observe(
                spec_decode_draft_acceptance_rate)
        if spec_decode_accepted_tokens := metrics_dict.get(
                MetricNames.SPEC_DECODE_ACCEPTED_TOKENS, 0):
            self.spec_decode_num_accepted_tokens.inc(
                spec_decode_accepted_tokens)
        if spec_decode_draft_tokens := metrics_dict.get(
                MetricNames.SPEC_DECODE_DRAFT_TOKENS, 0):
            self.spec_decode_num_draft_tokens.inc(spec_decode_draft_tokens)
        self.last_log_time = time.time()

    def log_iteration_stats(self, iteration_stats: dict) -> None:
        """
        Log iteration-level statistics from TRTLLM engine.

        This method updates Prometheus metrics including:
        - kv_cache_hit_rate
        - kv_cache_utilization

        Args:
            iteration_stats: A JSON dict returned from `BaseLLM.get_stats()` containing iteration-level statistics
                with the following expected structure:
                - "kvCacheStats" (dict): KV cache statistics containing:
                    - "cacheHitRate" (float): Cache hit rate (0.0 to 1.0). If present (including zero),
                      the kv_cache_hit_rate gauge is updated.
                    - "usedNumBlocks" (int): Number of KV cache blocks currently in use.
                    - "maxNumBlocks" (int): Maximum number of KV cache blocks available. Should always be
                      non-zero.

        Returns:
            None: Metrics are logged to Prometheus; nothing is returned.

        Note:
            - Needs to include `enable_iter_perf_stats: true` in LLM args to collect iteration-level stats.
            - KV cache utilization is only calculated and logged when both "usedNumBlocks" and
              "maxNumBlocks" are present in kvCacheStats and "maxNumBlocks" is non-zero.
        """
        if kv_stats := iteration_stats.get("kvCacheStats"):
            cache_hit_rate = kv_stats.get("cacheHitRate")
            if cache_hit_rate is not None:
                self._log_gauge(self.kv_cache_hit_rate, cache_hit_rate)
            if "usedNumBlocks" in kv_stats and "maxNumBlocks" in kv_stats:
                max_num_blocks = kv_stats["maxNumBlocks"]
                if max_num_blocks:
                    utilization = kv_stats["usedNumBlocks"] / max_num_blocks
                    self._log_gauge(self.kv_cache_utilization, utilization)
