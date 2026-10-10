# SamQuant C58 — Qwen3.8-Flash-Next Uncensored MLX-Serve Mix

This repository is the project homepage for the `SamQuant C58` mlx-serve model pack.

> **Model weights are hosted on Hugging Face:**  
> https://huggingface.co/LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58

The Hugging Face repository contains the full quantized weights (~90 GB). This GitHub repository provides:

- One-click download and verification scripts
- Local benchmarking helpers
- Usage examples for `mlx-serve`
- Project documentation and issue tracking

---

## What is this?

A mixed-precision, 4-bit `mlx-serve` pack of the abliterated (refusal-removed) Qwen3.8-Flash-Next model, built for Apple Silicon with 128 GB unified memory.

| Attribute | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next` |
| Abliterated source | `orcarouter/Qwen3.8-Flash-Next-Uncensored` |
| Format | `mlx-serve` (`.safetensors` + `ngram_table.bin`) |
| Quantization | Mixed 4-bit / 8-bit / bf16 |
| Total size | ~90 GB |
| N-gram table | 32 GB (`ngram_table.bin`, memory-mapped) |
| Recommended RAM | 128 GB Apple Silicon |
| Context length | Up to 262,144 tokens |

---

## Quick start

### 1. Install dependencies

```bash
pip install -U huggingface_hub hf_transfer
```

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

## Status and exposure

| Platform | Metric | Value |
|---|---|---|
| Hugging Face | Downloads | 71 |
| Hugging Face | Likes | 1 |
| Hugging Face | Discussions | 0 |

The model was published on 2026-10-08. For a 90 GB niche deployment pack, early organic downloads are healthy, but wider exposure will depend on community posts, benchmarks, and tutorials.

---

## License

This project is released under the [Qwen Community License 1.0](LICENSE), matching the upstream model license.

The abliterated base weights come from `orcarouter/Qwen3.8-Flash-Next-Uncensored`. The original architecture and training are by Qwen.

---

## Links

- **Hugging Face model:** https://huggingface.co/LOKSUN-158/SamQuant-Qwen3.8-Flash-Next-Uncensored-MLX-Serve-Mix-C58
- **mlx-serve:** https://github.com/ddalcu/mlx-serve
- **Base uncensored weights:** https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored
- **Original Qwen model:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next
