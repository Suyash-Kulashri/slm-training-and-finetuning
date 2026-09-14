#!/usr/bin/env python3
# =============================================================================
# merge_cpt.py — Manually merge the best CPT checkpoint into base model
# Run: python merge_cpt.py
# =============================================================================

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "meta-llama/Llama-3.1-8B"
CHECKPOINT_DIR = "llama3.1-8b-covid-cpt-checkpoints"
OUTPUT_DIR     = "llama3.1-8b-covid-cpt-bf16"

# ── Find best checkpoint ──────────────────────────────────────────────────────
import re
checkpoints = []
for d in os.listdir(CHECKPOINT_DIR):
    m = re.match(r"checkpoint-(\d+)$", d)
    if m:
        full = os.path.join(CHECKPOINT_DIR, d)
        if os.path.exists(os.path.join(full, "adapter_config.json")) or \
           os.path.exists(os.path.join(full, "config.json")):
            checkpoints.append((int(m.group(1)), full))

checkpoints = sorted(checkpoints, key=lambda x: x[0])
best_step, best_ckpt = checkpoints[-1]
print(f"Using checkpoint: {best_ckpt}  (step {best_step})")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
# Try checkpoint first, fall back to base model
try:
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
    print(f"  Tokenizer loaded from checkpoint")
except:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"  Tokenizer loaded from base model")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load base model ───────────────────────────────────────────────────────────
print("Loading base model in BF16 (this takes ~2 min)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.tie_word_embeddings = False
print(f"  Base model loaded")

# ── Load LoRA adapter from checkpoint ─────────────────────────────────────────
print(f"Loading LoRA adapter from {best_ckpt}...")
try:
    model = PeftModel.from_pretrained(model, best_ckpt)
    print(f"  LoRA adapter loaded")
except Exception as e:
    print(f"  ERROR loading as PeftModel: {e}")
    print("  Checkpoint may already be a merged model — trying direct load...")
    model = AutoModelForCausalLM.from_pretrained(
        best_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("  Loaded as full model")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✓ Saved directly → {OUTPUT_DIR}/")
    exit(0)

# ── Merge LoRA into base weights ──────────────────────────────────────────────
print("Merging LoRA into base weights...")
model = model.merge_and_unload()
model.config.tie_word_embeddings = False
print("  Merge complete")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving merged model → {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Verify
files = os.listdir(OUTPUT_DIR)
total_size = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in files
) / 1e9

print(f"\n✓ Saved → {OUTPUT_DIR}/")
print(f"  Files   : {len(files)}")
print(f"  Size    : {total_size:.1f} GB")
print(f"\nCPT merge complete. Ready to run SFT.")