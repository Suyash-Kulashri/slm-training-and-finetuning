# =============================================================================
# config.py — Central configuration for COVID SLM training pipeline
# Updated for: 4x NVIDIA A100-SXM4-40GB | 48 CPU cores | torchrun multi-GPU
# =============================================================================

import os

# ── Hardware ──────────────────────────────────────────────────────────────────
# CHANGED: 48 cores → use 40 for preprocessing (leave 8 for OS + GPU comms)
# OLD machine: 24 cores → NUM_WORKERS=20
NUM_WORKERS      = 40    # used by all data preprocessing stages
NPROC_PER_NODE   = 4     # number of GPUs for torchrun
MASTER_PORT      = 29500 # torchrun master port

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME  = "meta-llama/Llama-3.1-8B"
# CHANGED: max_seq_length stays 2048 — fits on 40GB with flash_attention_2 + batch=4
MAX_SEQ_LENGTH   = 2048

# ── CPT LoRA ──────────────────────────────────────────────────────────────────
# Unchanged — r=32 is correct, embed_tokens+lm_head needed for CPT
CPT_LORA_R       = 32
CPT_LORA_ALPHA   = 64    # 2× r rule
CPT_LORA_DROPOUT = 0.0
CPT_USE_RSLORA   = True
CPT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",             # required for CPT domain adaptation
]

# ── SFT LoRA ──────────────────────────────────────────────────────────────────
SFT_LORA_R       = 16
SFT_LORA_ALPHA   = 32
SFT_LORA_DROPOUT = 0.0
SFT_USE_RSLORA   = False
SFT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── CPT Training hyperparameters ──────────────────────────────────────────────
# CHANGED for 4x A100-40GB + torchrun DDP:
#   per_device=4 × 4 GPUs × grad_accum=4 = 64 effective batch (same as before)
#   OLD: per_device=16, grad_accum=4, 1 GPU → effective=64
CPT_PER_DEVICE_BATCH  = 4     # per GPU (was 16 on single 80GB)
CPT_GRAD_ACCUM        = 4     # per_device × num_gpus × grad_accum = 4×4×4 = 64
CPT_NUM_EPOCHS        = 1
CPT_LR                = 2e-5
CPT_WARMUP_RATIO      = 0.05
CPT_LR_SCHEDULER      = "cosine"
CPT_WEIGHT_DECAY      = 0.01
CPT_MAX_GRAD_NORM     = 1.0
# CHANGED: adamw_8bit → adamw_bnb_8bit (bitsandbytes naming in HF TrainingArguments)
CPT_OPTIM             = "adamw_bnb_8bit"
CPT_EVAL_STEPS        = 500
CPT_SAVE_STEPS        = 500
CPT_SAVE_TOTAL_LIMIT  = 5
# CHANGED: 8 → 10 workers per GPU (48 cores / 4 GPUs = 12, use 10 safely)
CPT_DATALOADER_WORKERS = 10

# ── SFT Training hyperparameters ──────────────────────────────────────────────
# CHANGED for 4x A100-40GB + torchrun DDP:
#   per_device=2 × 4 GPUs × grad_accum=8 = 64 effective batch
#   OLD: per_device=2, grad_accum=8, 1 GPU → effective=16
SFT_PER_DEVICE_BATCH  = 2
SFT_GRAD_ACCUM        = 8     # per_device × num_gpus × grad_accum = 2×4×8 = 64
SFT_NUM_EPOCHS        = 3
SFT_LR                = 1e-5
SFT_WARMUP_RATIO      = 0.05
SFT_LR_SCHEDULER      = "cosine"
SFT_WEIGHT_DECAY      = 0.01
SFT_MAX_GRAD_NORM     = 1.0
SFT_OPTIM             = "adamw_bnb_8bit"
SFT_NEFTUNE_ALPHA     = 5.0   # applied via model hook, not SFTConfig
SFT_EVAL_STEPS        = 200
SFT_SAVE_STEPS        = 200
SFT_SAVE_TOTAL_LIMIT  = 3
SFT_DATALOADER_WORKERS = 10

# ── Data pipeline ─────────────────────────────────────────────────────────────
RAW_PAPERS_JSONL    = "new_covid.jsonl"
CACHE_RAW_CHUNKS    = "_covid_cache_01_chunks_raw.jsonl"
CACHE_FINAL_CHUNKS  = "_covid_cache_02_chunks_final.jsonl"
OUTPUT_JSONL        = "covid19_data_chunked_deduped.jsonl"
CACHE_CPT_TOKENIZED = "_covid_tokenized_cache"
LOCAL_TOK_DIR       = "/tmp/local_tokenizer"

# Chunking
CHUNK_SIZE          = 1900   # leave ~148 token headroom for EOS + specials
OVERLAP             = 200
CHUNK_BATCH_SIZE    = 500

# Filtering
MIN_TEXT_LEN        = 50
ALPHA_RATIO_THR     = 0.5
JACCARD_THRESHOLD   = 0.8
MINHASH_NUM_PERM    = 64     # 64 vs 128: ~2x faster, ~97% accuracy retained
MINHASH_BATCH       = 5_000

# Token budget cap
TOKEN_BUDGET_M      = 500    # million tokens — CPT saturates well before 2B

# ── SFT data ──────────────────────────────────────────────────────────────────
SFT_DATASET_PATH    = "chatdoctor_healthcaremagic_train.jsonl"
SFT_CACHE_TOKENIZED = "_sft_tokenized_cache"

# ── Output paths ──────────────────────────────────────────────────────────────
CPT_CHECKPOINT_DIR  = "llama3.1-8b-covid-cpt-checkpoints"
CPT_MERGED_DIR      = "llama3.1-8b-covid-cpt-bf16"
SFT_CHECKPOINT_DIR  = "llama3.1-8b-covid-sft-checkpoints"
SFT_LORA_DIR        = "llama3.1-8b-covid-sft-lora"
SFT_MERGED_DIR      = "llama3.1-8b-covid-sft-merged-bf16"

# ── GGUF ──────────────────────────────────────────────────────────────────────
LLAMA_CPP_SCRIPT    = "llama.cpp/convert_hf_to_gguf.py"
CPT_GGUF_PATH       = "llama3.1-8b-covid-cpt-f16.gguf"
SFT_GGUF_PATH       = "llama3.1-8b-covid-sft-f16.gguf"

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED         = 3407
TOKEN_BATCH         = 1_000  # for batched token counting
