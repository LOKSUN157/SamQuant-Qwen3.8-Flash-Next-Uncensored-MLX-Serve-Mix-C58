#!/usr/bin/env python3
"""Minimal benchmark template for mlx-serve.

This script is intentionally simple: it launches mlx-serve in non-interactive mode,
records time and tokens, and prints a summary. Edit the prompt and parameters to match
your use case.

Usage:
    python scripts/benchmark.py \
        --model-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
        --prompt "Explain quantum computing in simple terms." \
        --max-tokens 512 \
        --temp 0.7
"""
import argparse
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark mlx-serve generation.")
    parser.add_argument("--model-dir", required=True, help="Path to the local model directory.")
    parser.add_argument("--prompt", required=True, help="Input prompt.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--mtp", action="store_true", help="Enable MTP speculative decoding.")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        "mlx-serve",
        "--model", args.model_dir,
        "--prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
        "--temp", str(args.temp),
    ]
    if args.mtp:
        cmd.append("--mtp")

    print(f"Running: {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Requested max tokens: {args.max_tokens}")
    print("Note: pipe the output through a tokenizer or use mlx-serve's built-in metrics for exact tok/s.")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
