#!/usr/bin/env python3
"""Verify a local copy of the model against the Hugging Face repository.

Usage:
    python scripts/verify.py --repo-id LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
        --local-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import HfApi, list_repo_tree


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a local model copy against Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository ID.")
    parser.add_argument("--local-dir", required=True, help="Local directory containing the model.")
    parser.add_argument(
        "--sha-samples",
        nargs="+",
        default=["ngram_table.bin", "model-00001.safetensors", "model-00050.safetensors"],
        help="Files to fully SHA-256 verify.",
    )
    return parser.parse_args()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    api = HfApi()

    print(f"Fetching remote file list for {args.repo_id} ...")
    remote_files = {
        f.path: f
        for f in list_repo_tree(args.repo_id, repo_type="model", expand=True)
        if f.__class__.__name__ == "RepoFile"
    }

    missing = []
    size_mismatches = []
    sha_mismatches = []
    sha_ok = []

    for rel_path, repo_file in sorted(remote_files.items()):
        local_path = os.path.join(args.local_dir, rel_path)
        if not os.path.exists(local_path):
            missing.append(rel_path)
            continue

        local_size = os.path.getsize(local_path)
        if local_size != repo_file.size:
            size_mismatches.append((rel_path, local_size, repo_file.size))
            continue

        if rel_path in args.sha_samples:
            # huggingface_hub 1.x: lfs.sha256 直接挂在 list_repo_tree 对象上（file_metadata 已移除）。
            lfs = getattr(repo_file, "lfs", None)
            remote_sha = getattr(lfs, "sha256", None) if lfs else None
            if not remote_sha:
                print(f"  WARN: 远端无 lfs.sha256，跳过 SHA 校验: {rel_path}（仅按大小核对）")
                continue
            local_sha = sha256_file(local_path)
            if local_sha != remote_sha:
                sha_mismatches.append(rel_path)
            else:
                sha_ok.append(rel_path)

    print(f"\nRemote files: {len(remote_files)}")
    print(f"Missing:      {len(missing)}")
    print(f"Size mismatches: {len(size_mismatches)}")
    print(f"SHA-256 OK:   {len(sha_ok)}")
    print(f"SHA-256 mismatches: {len(sha_mismatches)}")

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")
    if size_mismatches:
        print("\nSize mismatches:")
        for f, local, remote in size_mismatches:
            print(f"  - {f}: local={local}, remote={remote}")
    if sha_mismatches:
        print("\nSHA-256 mismatches:")
        for f in sha_mismatches:
            print(f"  - {f}")

    if not missing and not size_mismatches and not sha_mismatches:
        print("\nVerification PASSED.")
        return 0
    else:
        print("\nVerification FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
