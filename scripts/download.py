#!/usr/bin/env python3
"""Download the SamQuant C58 model from Hugging Face and verify file integrity.

Usage:
    python scripts/download.py --repo-id LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
        --local-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree


def parse_args():
    parser = argparse.ArgumentParser(description="Download and verify the SamQuant C58 model.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository ID.")
    parser.add_argument("--local-dir", required=True, help="Local directory to save the model.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume partial downloads.")
    return parser.parse_args()


def sha256_file(path: str, block_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 of a file with a progress-friendly block size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_repo(repo_id: str, local_dir: str) -> dict:
    """Download all files from repo and return {relative_path: local_absolute_path}."""
    os.makedirs(local_dir, exist_ok=True)
    api = HfApi()
    files = [f for f in list_repo_tree(repo_id, repo_type="model", expand=True) if f.__class__.__name__ == "RepoFile"]

    downloaded = {}
    for i, repo_file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Downloading {repo_file.path} ...")
        # 注：local_dir_use_symlinks / resume_download 已在 huggingface_hub 1.x 移除
        # （1.x 默认即拷贝到 local_dir 并支持断点续传）；保留会抛 TypeError。
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file.path,
            repo_type="model",
            local_dir=local_dir,
        )
        downloaded[repo_file.path] = local_path
    return downloaded


def verify_download(repo_id: str, local_dir: str, downloaded: dict) -> bool:
    """Verify downloaded files match remote sizes and sampled SHA-256 hashes."""
    api = HfApi()
    files = {f.path: f for f in list_repo_tree(repo_id, repo_type="model", expand=True) if f.__class__.__name__ == "RepoFile"}

    all_ok = True
    for rel_path, repo_file in files.items():
        local_path = os.path.join(local_dir, rel_path)
        if not os.path.exists(local_path):
            print(f"MISSING: {rel_path}")
            all_ok = False
            continue

        local_size = os.path.getsize(local_path)
        if local_size != repo_file.size:
            print(f"SIZE MISMATCH: {rel_path} (local {local_size} != remote {repo_file.size})")
            all_ok = False
            continue

        # SHA-256 for the largest file and a few sampled shards; size-only for the rest.
        # huggingface_hub 1.x: 远端 SHA-256 直接来自 list_repo_tree 对象的 lfs.sha256（file_metadata 已移除）。
        lfs = getattr(repo_file, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None) if lfs else None
        if rel_path in ("ngram_table.bin", "model-00001.safetensors", "model-00050.safetensors"):
            if not remote_sha:
                print(f"WARN: 远端无 lfs.sha256，跳过 SHA 校验 {rel_path}（仅按大小核对）")
                continue
            local_sha = sha256_file(local_path)
            if local_sha != remote_sha:
                print(f"SHA-256 MISMATCH: {rel_path}")
                all_ok = False
            else:
                print(f"SHA-256 OK: {rel_path}")
        else:
            print(f"SIZE OK: {rel_path}")

    return all_ok


def main():
    args = parse_args()
    print(f"Repo:    {args.repo_id}")
    print(f"Target:  {os.path.abspath(args.local_dir)}")

    downloaded = download_repo(args.repo_id, args.local_dir)
    print("\nDownload complete. Verifying ...")

    if verify_download(args.repo_id, args.local_dir, downloaded):
        print("\nAll files downloaded and verified successfully.")
        return 0
    else:
        print("\nVerification failed. Please re-run the script.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
