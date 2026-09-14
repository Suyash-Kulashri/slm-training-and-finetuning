# =============================================================================
# data_tokenize_cache.py — Pre-tokenize and persist datasets to disk
#
# CHANGE from original:
#   - num_proc=NUM_WORKERS=40 for both CPT and SFT tokenization (was 20)
#   - set_format("torch") re-applied after load_from_disk (not persisted by HF)
# =============================================================================

import os
import time
from datasets import DatasetDict, load_from_disk
from config import (
    CACHE_CPT_TOKENIZED, SFT_CACHE_TOKENIZED,
    NUM_WORKERS,   # CHANGED: now 40
)


def tokenize_and_cache_cpt(dataset, tokenizer, max_seq_length: int) -> DatasetDict:
    """
    Pre-tokenize CPT dataset with 40 workers and persist to disk.
    On cache hit, loads instantly and re-applies set_format("torch")
    (set_format is NOT persisted by save_to_disk/load_from_disk).
    """
    if os.path.exists(CACHE_CPT_TOKENIZED):
        print("✓ Loading pre-tokenized CPT dataset from cache...")
        tokenized = load_from_disk(CACHE_CPT_TOKENIZED)
        # IMPORTANT: set_format is not saved to disk — must re-apply on load
        tokenized["train"].set_format("torch")
        tokenized["test"].set_format("torch")
        print(f"✓ Ready — {len(tokenized['train']):,} train  |  {len(tokenized['test']):,} val")
        return tokenized

    print(f"Pre-tokenizing CPT dataset with {NUM_WORKERS} workers (runs once, then cached)...")
    t0 = time.time()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = {}
    for split in ["train", "test"]:
        tokenized[split] = dataset[split].map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=NUM_WORKERS,       # CHANGED: 40 (was 20)
            remove_columns=["text"],    # removes text col — trainer must NOT use dataset_text_field
            desc=f"Tokenizing {split}",
        )
        tokenized[split].set_format("torch")

    tokenized = DatasetDict(tokenized)
    tokenized.save_to_disk(CACHE_CPT_TOKENIZED)
    print(f"✓ CPT tokenized cache → {CACHE_CPT_TOKENIZED}  ({(time.time()-t0)/60:.1f} min)")
    print(f"✓ Ready — {len(tokenized['train']):,} train  |  {len(tokenized['test']):,} val")
    return tokenized


def tokenize_and_cache_sft(sft_dataset, tokenizer, sft_max_seq_length: int) -> DatasetDict:
    """
    Pre-tokenize SFT dataset with 40 workers and persist to disk.
    Same pattern as CPT. set_format("torch") re-applied on cache hit.
    """
    if os.path.exists(SFT_CACHE_TOKENIZED):
        print("✓ Loading pre-tokenized SFT dataset from cache...")
        sft_tok = load_from_disk(SFT_CACHE_TOKENIZED)
        # IMPORTANT: set_format is not saved to disk — must re-apply on load
        sft_tok["train"].set_format("torch")
        sft_tok["test"].set_format("torch")
        print(f"✓ Ready — {len(sft_tok['train']):,} train  |  {len(sft_tok['test']):,} val")
        return sft_tok

    print(f"Pre-tokenizing SFT dataset with {NUM_WORKERS} workers (runs once, then cached)...")

    def sft_tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=sft_max_seq_length,
            padding=False,
        )

    sft_tok = {}
    for split in ["train", "test"]:
        sft_tok[split] = sft_dataset[split].map(
            sft_tokenize,
            batched=True,
            batch_size=1000,
            num_proc=NUM_WORKERS,       # CHANGED: 40 (was 20)
            remove_columns=["text"],    # removes text col — trainer must NOT use dataset_text_field
            desc=f"SFT tokenizing {split}",
        )
        sft_tok[split].set_format("torch")

    sft_tok = DatasetDict(sft_tok)
    sft_tok.save_to_disk(SFT_CACHE_TOKENIZED)
    print(f"✓ SFT tokenized cache → {SFT_CACHE_TOKENIZED}")
    print(f"✓ Ready — {len(sft_tok['train']):,} train  |  {len(sft_tok['test']):,} val")
    return sft_tok
