#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download minimal models required to run OpenAI server unit tests.
# Run from repo root: python nebius/download_minimal_models.py
#
# Usage:
#   python nebius/download_minimal_models.py [--output-dir DIR]
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
# Guided decoding (_test_openai_chat_guided_decoding.py) needs:
#   - meta-llama/Llama-3.1-8B-Instruct (below) - run with -k "meta-llama"
#   - openai/gpt-oss-120b: gpt_oss/gpt-oss-120b + gpt_oss/gpt-oss-120b-Eagle3
#     (not on public HF; skip with -k "meta-llama" for minimal setup)
MINIMAL_MODELS = {
    "llama-models-v2/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-3.1-model/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
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
    print("For guided decoding tests only (Llama-3.1-8B):")
    print(
        "  cd tests && pytest unittest/llmapi/apps/_test_openai_chat_guided_decoding.py -k meta-llama -v"
    )
    print()
    print("Or add to your shell profile / .env:")
    print(f"  LLM_MODELS_ROOT={output_dir}")


if __name__ == "__main__":
    sys.exit(main())
