#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download minimal models required to run OpenAI server unit tests.
# Run from repo root: python nebius/devtools/download_minimal_models.py
#
# Usage:
#   python nebius/devtools/download_minimal_models.py [--output-dir DIR]
#
# After running, set LLM_MODELS_ROOT and run tests:
#   export LLM_MODELS_ROOT=/path/to/output/dir
#   cd tests && pytest unittest/llmapi/apps/ -k openai -v

import argparse
import os
import sys
from pathlib import Path

# Models needed for unittest/llmapi/apps/ -k openai tests
# Path under LLM_MODELS_ROOT : HuggingFace repo ID
#
# GPT-OSS (HuggingFace): see docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md
#   - gpt_oss/gpt-oss-20b: _test_openai_responses.py, _test_openai_chat_harmony.py, etc.
#   - gpt_oss/gpt-oss-120b + gpt_oss/gpt-oss-120b-Eagle3: _test_openai_chat_guided_decoding.py
#     when running the openai/gpt-oss-120b parametrization (large weights; needs HF auth if gated).
MINIMAL_MODELS = {
    "llama-models-v2/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-3.1-model/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt_oss/gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt_oss/gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt_oss/gpt-oss-120b-Eagle3": "nvidia/gpt-oss-120b-Eagle3",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download minimal models for OpenAI server unit tests"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.environ.get(
            "LLM_MODELS_ROOT",
            str(Path(__file__).resolve().parent.parent / "/home/scratch.trt_llm_data_ci"),
        ),
        help="Output directory (default: /home/scratch.trt_llm_data_ci/ or LLM_MODELS_ROOT)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only create directory structure and print instructions",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        print(f"Output directory: {output_dir}")
        print("Run without --no-download to fetch models.")
        print_usage(output_dir)
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading models to {output_dir}")
    print()

    for subdir, repo_id in MINIMAL_MODELS.items():
        dest = output_dir / subdir
        if dest.exists() and any(dest.iterdir()):
            print(f"  [skip] {subdir} (already exists)")
            continue
        print(f"  [downloading] {repo_id} -> {subdir}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            print(f"  [done] {subdir}")
        except Exception as e:
            print(f"  [error] {subdir}: {e}")
            sys.exit(1)

    print()
    print_usage(output_dir)
    return 0


def print_usage(output_dir: Path):
    print("Next steps:")
    print()
    print(f"  export LLM_MODELS_ROOT={output_dir}")
    print("  cd tests && pytest unittest/llmapi/apps/ -k openai -v")
    print()
    print("Guided decoding without GPT-OSS 120B (Llama-3.1-8B only):")
    print(
        "  cd tests && pytest unittest/llmapi/apps/_test_openai_chat_guided_decoding.py -k meta-llama -v"
    )
    print()
    print("Full guided decoding module (Llama + GPT-OSS 120B + Eagle3):")
    print(
        "  cd tests && pytest unittest/llmapi/apps/_test_openai_chat_guided_decoding.py -v"
    )
    print()
    print("Or add to your shell profile / .env:")
    print(f"  LLM_MODELS_ROOT={output_dir}")


if __name__ == "__main__":
    sys.exit(main())
