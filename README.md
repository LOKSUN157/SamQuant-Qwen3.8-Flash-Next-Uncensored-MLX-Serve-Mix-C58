# SamQuant C58 — Qwen3.8-Flash-Next Uncensored MLX-Serve Mix

This repository is the project homepage for the `SamQuant C58` mlx-serve model pack.

> **Model weights are hosted on Hugging Face:**  
> https://huggingface.co/LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
>
> Note: Hugging Face forbids double hyphens in repo names, so the online repo id uses a single `-` (local pack directories may still be named `SamQuant--...` with `--`; the files are identical). GitHub account here is `LOKSUN157`, Hugging Face account is `LOKSUN-158`.

The Hugging Face repository contains the full quantized weights (~90 GB). This GitHub repository provides:

- One-click download and verification scripts
- Local benchmarking helpers
- Usage examples for `mlx-serve`
- Project documentation and issue tracking

---

## What is this?

A mixed-precision `mlx-serve` pack of the uncensored Qwen3.8-Flash-Next model (180B-class MoE: 125B backbone + 51B n-gram table + 4B MTP head), built for Apple Silicon with 128 GB unified memory. The uncensored property is inherited from the `orcarouter` source weights; this pack makes no behavior modifications.

| Attribute | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next` |
| Uncensored source weights | `orcarouter/Qwen3.8-Flash-Next-Uncensored` (BF16, 335 GiB / 131 shards, sha256-verified 132/132) |
| Format | `mlx-serve` (`.safetensors` + `ngram_table.bin`) |
| Quantization | Mixed width: experts 3-bit/4-bit (layer-wise greedy, 70 of 96 groups 3-bit), layers 46-47 + MTP-head experts 4-bit g64, attention/GDN/shared-experts 6-bit g64, lm_head 8-bit g64, embed 4-bit g64, n-gram table 4-bit g32, routers/norms/vision tower bf16 |
| Weight payload | ~59.8 GB (incl. vision tower & MTP head) |
| Total size | ~90 GB (adds the 32 GB n-gram table) |
| N-gram table | 32,000,153,976 bytes (`ngram_table.bin`, memory-mapped) |
| Recommended RAM | 128 GB Apple Silicon |
| Context length | Native 262,144; production uses YaRN factor 2.0 → 524,288 (a 421,473-token single turn ran in production) |

---

## Quick start

### 1. Install dependencies

```bash
pip install -U huggingface_hub mlx-serve
```

> **Version requirement:** this pack contains 3-bit and 6-bit weights and requires **mlx-serve ≥ 26.9.1** (older versions dequantize these widths into noise). Verified working on mlx-serve 26.9.6 / MLX 0.32.2. `ngram_table.bin` must stay in the model directory with its `.bin` extension — do not rename or convert it (it is mmap'd, not a safetensors shard).

### 2. Download the model

```bash
python scripts/download.py \
  --repo-id LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
  --local-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
```

The script automatically verifies every file with SHA-256 after download.

### 3. Run with mlx-serve

```bash
mlx-serve \
  --model ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
  --prompt "Hello!" \
  --max-tokens 512 \
  --temp 0.7
```

For OpenAI-compatible server mode:

```bash
mlx-serve serve \
  --model ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
  --host 127.0.0.1 \
  --port 11234
```

---

## Repository structure

```
.
├── LICENSE                 # Qwen Community License 1.0
├── README.md               # This file
├── scripts/
│   ├── download.py         # Download weights from Hugging Face + verify SHA-256
│   ├── verify.py           # Verify an already-downloaded local copy
│   └── benchmark.py        # Run a simple local benchmark (template)
```

No model weights are stored in this repository.

---

## Hardware requirements

- **Apple Silicon Mac** (M-series)
- **128 GB unified memory** recommended for long contexts
- **Internal SSD** with ~100 GB free; keep the model folder on the internal drive because `ngram_table.bin` is memory-mapped and random reads from an external drive will severely hurt performance
- `mlx-serve` with `qwen4_exp` support

---

## Verification

To confirm a local download is byte-for-byte identical to the Hugging Face version:

```bash
python scripts/verify.py \
  --repo-id LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
  --local-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
```

The verifier compares every file's size and samples SHA-256 hashes.

---

## Benchmark

A minimal benchmark template is included in [`scripts/benchmark.py`](scripts/benchmark.py). Adapt it to your prompt and context length, then run:

```bash
python scripts/benchmark.py \
  --model-dir ~/.mlx-serve/models/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58 \
  --prompt "Explain quantum computing in simple terms." \
  --max-tokens 512
```

Contributions with real-world numbers (tokens/sec, resident memory) are welcome.

---

## Measured on the author's machine (Apple M3 Max, 128 GB)

All numbers below are real production/verification readings from a single Apple M3 Max 128 GB machine running mlx-serve 26.9.6 / MLX 0.32.2. They are **not** representative of every device, and `token ≠ Chinese character`.

**Production run** (daily agent sessions, 51 hours from 2026-10-06 17:24 to 10-08 evening, aggregated from the engine's per-round completion lines):
- 947 completed rounds, **0 failures** (the only 6 interruptions were manual aborts by the author)
- 182M prompt tokens read; **98.19% prefix-cache hit**
- decode p50 **36.9 tok/s** (p10 26.2 / p90 52.6); largest single turn reached **421,473 tokens** of context
- engine-resident footprint after load (phys_footprint): **~61 GB**

**Dedicated long-context window** (2026-09-30, 524K ctx, 57 rounds with >30K-token prompts): decode p50 **46.0 tok/s**. On the same night the author's lighter same-series pack B55 (52.4 GB weights, 3-bit experts) ran p50 **37.0 tok/s** — C58 is ~8 GB heavier yet faster, because its 4-bit experts land in the engine's fused-kernel fast path while 3-bit experts do not.

**Offline quality** (600 held-out scoring positions, top-1 agreement vs the BF16 reference; scoring prefixes 65–2,297 tokens): C58 **85.40%**; community 4-bit pack 87.82% (−2.42); same-budget all-4-bit pack 85.76% (+0.36).

**Business acceptance** (24 tasks × 7 packs: tool-orchestration / code / long-doc, 2026-10-04): C58 — 0 long-doc failures, code throughput 50.4 tok/s (highest of the 7 tested), tool-orchestration 3/8. Multi-round tool orchestration is a shared weakness across all quantized packs tested; set expectations accordingly.

**Recommended production launch flags** (verbatim from the running engine process):
```bash
mlx-serve --model <this-pack-dir> --serve --host 127.0.0.1 --port 8093 --metrics \
  --ctx-size 524288 \
  --config-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144},"max_position_embeddings":524288}}' \
  --prefix-cache-mem 32GB --prefix-cache-entries 128 \
  --mtp --kv-quant 4 --pld-draft-len 8 --mtp-head-kv-quant \
  --prefill-chunk 8192 --reasoning-budget 32768 \
  --idle-evict-secs 3600 --tokenize-cache-entries 16 \
  --os-reserve-gib 24 --wired-margin-gib 8 --timeout 0
```

---

## Status and exposure (as of 2026-10-10)

| Platform | Metric | Value |
|---|---|---|
| Hugging Face | Downloads | 71 |
| Hugging Face | Likes | 1 |
| Hugging Face | Discussions | 0 |

The model was published on 2026-10-08. For a 90 GB niche deployment pack, early organic downloads are healthy, but wider exposure will depend on community posts, benchmarks, and tutorials.

---

## License

This project is released under the [Qwen Community License 1.0](LICENSE), matching the upstream model license.

The uncensored base weights come from `orcarouter/Qwen3.8-Flash-Next-Uncensored` (uncensoring done upstream; this pack adds no behavior modification). The original architecture and training are by Qwen. The quantization toolchain (imatrix calibration → bit allocation → conversion → scoring) is the stock pipeline shipped with [mlx-serve](https://github.com/ddalcu/mlx-serve).

---

## Links

- **Hugging Face model:** https://huggingface.co/LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
- **mlx-serve:** https://github.com/ddalcu/mlx-serve
- **Base uncensored weights:** https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored
- **Original Qwen model:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next
