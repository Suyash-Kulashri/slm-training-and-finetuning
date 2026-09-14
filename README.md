# SLM Training & Finetuning — COVID-19 Medical Assistant

A reproducible pipeline for domain-adapting **Meta Llama-3.1-8B** into a COVID-19 medical assistant "small language model" (SLM), built for **multi-GPU training with `torchrun`** on **4× NVIDIA A100-SXM4-40GB**.

The pipeline runs two training stages — **Continued Pre-Training (CPT)** for domain adaptation on COVID-19 literature, followed by **Supervised Fine-Tuning (SFT)** for instruction-following medical Q&A — then converts the result to **GGUF** for fast local inference via `llama.cpp`.

It is a rewrite of an original Unsloth/single-GPU notebook workflow into plain **HuggingFace `Trainer` + PEFT + DDP**, so it can scale across multiple GPUs without Unsloth's single-GPU limitation.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
- [Hardware & software requirements](#hardware--software-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
  - [1. Data preprocessing](#1-data-preprocessing-cpu-only)
  - [2. CPT training](#2-cpt-continued-pre-training)
  - [3. SFT training](#3-sft-supervised-fine-tuning)
  - [4. GGUF conversion](#4-gguf-conversion)
  - [5. Inference](#5-inference)
- [Data transfer helpers (S3)](#data-transfer-helpers-s3)
- [Troubleshooting](#troubleshooting)
- [Known issues in this repo](#known-issues-in-this-repo)
- [License](#license)

---

## Pipeline overview

```
                 ┌────────────────┐
 raw COVID       │ Data Pipeline  │  chunk → filter/dedup → build dataset → tokenize+cache
 papers (.jsonl) │ (CPU, 40 cores)│
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐   LoRA r=32 (incl. embed_tokens + lm_head)
                 │  CPT training  │   Llama-3.1-8B → domain-adapted base model
                 │ (4×A100, DDP)  │
                 └───────┬────────┘
                         ▼
                merged CPT model (bf16, ~16GB)
                         │
                         ▼
                 ┌────────────────┐   LoRA r=16 (attention + FFN only)
                 │  SFT training  │   chat-formatted Q&A, NEFTune noise
                 │ (4×A100, DDP)  │
                 └───────┬────────┘
                         ▼
        SFT LoRA adapter (~150MB) + merged SFT model (bf16, ~16GB)
                         │
                         ▼
                 ┌────────────────┐
                 │ GGUF conversion│   via llama.cpp convert_hf_to_gguf.py
                 └───────┬────────┘
                         ▼
              llama3.1-8b-covid-sft-f16.gguf  →  inference (HF or llama.cpp)
```

The base model is domain-adapted on COVID-19 research text during CPT, then taught to answer medical questions conversationally during SFT (using the ChatDoctor/HealthCareMagic-style instruction dataset referenced in `config.py`).

---

## Repository structure

| File | Purpose |
|---|---|
| `train_model_QAT16_final.py` | **Main entry point.** CLI that dispatches to data-only, CPT-only, SFT-only, full-pipeline, or inference modes. |
| `config.py` | Central configuration: hardware settings, model name, LoRA hyperparameters, training hyperparameters, data pipeline settings, and output paths. |
| `model_loader.py` | Loads the base model/tokenizer and applies PEFT LoRA adapters for CPT and SFT respectively. |
| `data_chunking.py` | Splits raw COVID-19 papers into overlapping token-window chunks in parallel (`ProcessPoolExecutor`). |
| `data_filtering.py` | Filters and deduplicates chunks: exact/MD5 dedup, length filter, alpha-ratio filter, language detection, boilerplate removal, and MinHash-LSH near-duplicate removal. |
| `data_dataset.py` | Builds the final HuggingFace dataset from filtered chunks: appends EOS tokens, applies a token budget cap, splits train/val, and counts tokens. |
| `data_tokenize_cache.py` | Pre-tokenizes the CPT and SFT datasets and persists them to disk so training can resume instantly without re-tokenizing. |
| `cpt_trainer.py` | Builds and runs the CPT `Trainer` (HuggingFace `Trainer` + `TrainingArguments`), with checkpoint-resume and rank-0-only saving for DDP. |
| `sft_trainer.py` | Builds and runs the SFT `Trainer`: applies the Llama-3 chat template, formats instruction/input/output records, profiles token-length distribution, applies NEFTune noise, and trains. |
| `merge_cpt.py` | Standalone script to manually merge the best CPT LoRA checkpoint into the base model (fallback if the automatic merge step fails). |
| `eval_utils.py` | Evaluation: perplexity on held-out data, plus free-form COVID-19 completion spot-checks (CPT) and medical Q&A spot-checks (SFT). |
| `gguf_utils.py` | Converts merged HF models to GGUF via `llama.cpp`, and runs GGUF inference through `llama-cpp-python`. |
| `inference.py` | Core HF inference functions: streaming/non-streaming completion, and chat-style Q&A using the Llama-3 template. |
| `run_cpt.py` | Standalone CPT pipeline runner (distributed setup + data pipeline + training), launchable directly with `torchrun`. |
| `run_sft.py` | Standalone SFT pipeline runner (distributed setup + data formatting + training), launchable directly with `torchrun`. |
| `run_inference.py` | CLI for running inference — HF or GGUF, streaming or not, with custom questions/model paths. |
| `download_from_s3.sh` / `upload_to_s3.sh` | Helper scripts to pull the SFT dataset from / push the final GGUF model to an S3 bucket. |
| `requirements.txt` | Python dependencies, with an install-order note (torch → flash-attn → the rest). |
| `LAUNCH.md` | Step-by-step operational runbook for launching training on 4×A100 using `screen` sessions (see [Running the pipeline](#running-the-pipeline) below, which condenses it). |

---

## Hardware & software requirements

The pipeline is tuned specifically for:

- **4× NVIDIA A100-SXM4-40GB** GPUs
- **~48 CPU cores** (40 are used for data preprocessing)
- CUDA 12.1
- Python environment with PyTorch, HuggingFace `transformers`/`datasets`/`peft`/`accelerate`, and (optionally) Flash Attention 2

It will also run on fewer/smaller GPUs or a single GPU with reduced batch sizes (see `config.py`), and gracefully falls back to `sdpa` attention if Flash Attention 2 isn't installed.

---

## Installation

```bash
# 1. Persist environment variables (survive reboots)
echo 'export PYTORCH_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
source ~/.bashrc

# 2. Install PyTorch for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Flash Attention 2 — try the prebuilt wheel first (no nvcc build needed)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# If this 404s, skip it and use the "sdpa" fallback (see Troubleshooting)

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Log in to HuggingFace (required once, for gated Llama-3.1 weights)
huggingface-cli login
# Token: https://huggingface.co/settings/tokens (Read permission)
# Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B
```

Every new terminal session, before training:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd ~/slm_training_no_unsloth      # or wherever you cloned this repo
source .venv/bin/activate
```

---

## Configuration

All tunables live in `config.py`, grouped as:

- **Hardware** — `NUM_WORKERS` (CPU cores for data preprocessing), `NPROC_PER_NODE` (GPU count), `MASTER_PORT`.
- **Model** — `BASE_MODEL_NAME` (`meta-llama/Llama-3.1-8B`), `MAX_SEQ_LENGTH` (2048).
- **CPT LoRA** — `r=32`, `alpha=64`, targets include `embed_tokens`/`lm_head` (needed so the model can adapt to the new domain's token distribution), rank-stabilized LoRA (`rsLoRA`) enabled.
- **SFT LoRA** — `r=16`, `alpha=32`, attention/FFN projections only (no embedding layers — SFT teaches instruction-following, not new vocabulary).
- **CPT hyperparameters** — per-device batch, gradient accumulation, epochs, LR (`2e-5`), cosine schedule, `adamw_bnb_8bit` optimizer, eval/save every 500 steps.
- **SFT hyperparameters** — per-device batch, gradient accumulation, 3 epochs, LR (`1e-5`), NEFTune alpha `5.0`, eval/save every 200 steps.
- **Data pipeline** — raw input path, cache file names, chunk size (1900 tokens with 200 overlap), MinHash/Jaccard dedup thresholds, and a **500M-token budget cap** for CPT (since CPT saturates well before 2B tokens on a dataset this size).
- **Output paths** — checkpoint/merged-model directories for both CPT and SFT, and the final GGUF file paths.

Effective batch size is `per_device_batch × num_gpus × grad_accum` — both CPT and SFT are configured to reach an effective batch of **64** on 4 GPUs.

---

## Running the pipeline

### 1. Data preprocessing (CPU only)

No GPUs or `torchrun` needed — uses all available CPU cores.

```bash
python train_model_QAT16_final.py --data-only
```

This chunks the raw papers, filters/deduplicates them, builds the HF dataset, and pre-tokenizes + caches everything:

- `_covid_cache_01_chunks_raw.jsonl`
- `_covid_cache_02_chunks_final.jsonl`
- `covid19_data_chunked_deduped.jsonl`
- `_covid_tokenized_cache/`

### 2. CPT (Continued Pre-Training)

Run inside a `screen` session so it survives disconnects:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

screen -S cpt_training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --cpt-only --skip-data 2>&1 | tee cpt_training.log
# Detach: Ctrl+A then D
```

Reattach any time with `screen -r cpt_training` or `tail -f cpt_training.log`. If it crashes, re-run the same command with `tee -a` (append) — it resumes automatically from the last checkpoint.

**Outputs:** `llama3.1-8b-covid-cpt-checkpoints/` (rolling checkpoints) and `llama3.1-8b-covid-cpt-bf16/` (final merged model). If the automatic merge step is skipped or fails, run `python merge_cpt.py` manually.

### 3. SFT (Supervised Fine-Tuning)

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

screen -S sft_training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --sft-only 2>&1 | tee sft_training.log
```

**Outputs:** `llama3.1-8b-covid-sft-checkpoints/`, `llama3.1-8b-covid-sft-lora/` (adapter only, ~150 MB), `llama3.1-8b-covid-sft-merged-bf16/` (final merged model, ~16 GB).

Before converting to GGUF, the tokenizer must be copied from the base model into the merged directory:

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tok.save_pretrained('llama3.1-8b-covid-sft-merged-bf16')
"
```

### 4. GGUF conversion

```bash
python3 llama.cpp/convert_hf_to_gguf.py llama3.1-8b-covid-sft-merged-bf16 \
    --outfile llama3.1-8b-covid-sft-f16.gguf --outtype f16
```

(Requires `git clone https://github.com/ggerganov/llama.cpp` and its own `requirements.txt` installed. You can also do this from the code via `gguf_utils.convert_sft_to_gguf()` or `run_inference.py --convert-gguf sft`.)

### 5. Inference

**Via HuggingFace (Python):**

```bash
python run_inference.py
python run_inference.py --question "What are COVID-19 symptoms?"
python run_inference.py --model-path llama3.1-8b-covid-sft-merged-bf16
```

**Via GGUF (llama.cpp / llama-cpp-python):**

```bash
python run_inference.py --gguf --model-path llama3.1-8b-covid-sft-f16.gguf

# or the raw llama.cpp CLI:
~/llama.cpp/llama-cli \
    -m llama3.1-8b-covid-sft-f16.gguf \
    -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful COVID-19 medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat are the symptoms of COVID-19?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
    -n 512 --temp 0.7 --repeat-penalty 1.15 --no-penalize-nl -ngl 99
```

**Evaluation** (perplexity + spot-check prompts) runs automatically after each training stage unless `--no-eval` is passed, and can also be triggered directly via `eval_utils.run_cpt_eval()` / `run_sft_eval()`.

---

## Data transfer helpers (S3)

Two convenience scripts wrap the AWS CLI for moving data in and out of S3:

- `download_from_s3.sh` — pulls the SFT training dataset (`SFT_covid.jsonl`) from S3 to the local working directory.
- `upload_to_s3.sh` — pushes the final GGUF model (`llama3.1-8b-covid-sft-f16.gguf`) up to S3.

Both check that the AWS CLI is installed and that credentials are configured (`aws sts get-caller-identity`) before running. **The S3 bucket name and file paths are hardcoded** in each script — edit them to point at your own bucket before use.

---

## Expected training times (4× A100-SXM4-40GB)

| Stage | Steps | Est. time |
|---|---|---|
| Data pipeline | — | ~30–60 min |
| CPT training | ~4,500 | ~15–20 hours |
| SFT training | ~3,300 | ~3–5 hours |
| GGUF conversion | — | ~10–15 min |

(For comparison, CPT alone takes ~50–80 hours on a single A100-80GB — roughly a 4× speedup from multi-GPU DDP.)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NCCL timeout` or hang at first step | Confirm `ddp_find_unused_parameters=False` in `TrainingArguments` and `use_reentrant=False` in `gradient_checkpointing_kwargs`. |
| `CUDA out of memory` during eval | Set `per_device_eval_batch_size=1` in `cpt_trainer.py`; reduce `CPT_PER_DEVICE_BATCH` to 1 in `config.py` and increase `CPT_GRAD_ACCUM` to compensate. |
| `ImportError: cannot import name get_chat_template` | Remove the Unsloth import from `sft_trainer.py` and use the hardcoded Llama-3 chat template instead. |
| `ValueError: Tokenizer class TokenizersBackend does not exist` | Run the tokenizer-fix command (Step 2 above) before GGUF conversion. |
| `ModuleNotFoundError: flash_attn` | Change `attn_implementation="sdpa"` in `model_loader.py` and `inference.py` — `sdpa` ships with PyTorch 2.0+, no install needed. |
| Port already in use | Change `--master_port=29500` to any free port. |
| Only 1 GPU being used | Make sure you launched with `torchrun`, not `python`; check `CUDA_VISIBLE_DEVICES` isn't restricted to one device. |
| Lost `screen` session | `screen -ls` to list sessions; `tail -f *.log` still works even if the session died — checkpoints are safe regardless. |

---

## Known issues in this repo

A few things worth fixing before you rely on this code as-is:

- **Unresolved Git merge conflicts.** Several files still contain literal `<<<<<<< HEAD` / `=======` / `>>>>>>>` conflict markers left over from a merge (`config.py`, `data_dataset.py`, `data_tokenize_cache.py`, `cpt_trainer.py`, `model_loader.py`, `sft_trainer.py`, `inference.py`, `run_inference.py`). **These will cause a `SyntaxError` if run as-is** — each conflict needs to be resolved by picking one side (generally the `07d00f6 (Final code of CPT and SFT)` side looks like the intended final version, e.g. `attn_implementation="sdpa"` over `flash_attention_2`, and `SFT_DATASET_PATH = "SFT_covid.jsonl"` over the ChatDoctor path) before training will run.
- **Hardcoded S3 bucket** (`s3://hsihsak`) in both shell scripts — replace with your own bucket.
- **`unsloth` listed as a dependency** solely for its chat-template helper, but `sft_trainer.py`'s conflicting code already includes a hardcoded Llama-3 template as a fallback — once the merge conflict above is resolved in favor of the hardcoded template, the `unsloth` dependency in `requirements.txt` can likely be dropped entirely.
- **No `LICENSE` file** is present in the repository.
- **Gated base model** — `meta-llama/Llama-3.1-8B` requires accepting Meta's license on HuggingFace and authenticating via `huggingface-cli login` before the pipeline can download it.

---

## License

No license file was found in this repository. Confirm usage terms with the repository owner before reusing this code, and note that the base model (Llama-3.1-8B) carries its own separate license from Meta.
