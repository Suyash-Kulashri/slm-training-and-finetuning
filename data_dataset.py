# =============================================================================
# data_dataset.py — Build HF dataset from filtered chunks
#
# CHANGE from original:
#   - _add_eos map() now uses num_proc=NUM_WORKERS=40 (was single-core)
# =============================================================================

import time
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm
from config import (
    CACHE_FINAL_CHUNKS, OUTPUT_JSONL,
    TOKEN_BUDGET_M, TOKEN_BATCH,
    NUM_WORKERS,   # CHANGED: now 40
    RANDOM_SEED,
)


def _add_eos(examples, eos_token):
    return {"text": [
        (t + eos_token) if t and t.strip() else eos_token
        for t in examples["text"]
    ]}


def build_dataset(tokenizer):
    """
    Load final filtered chunks, append EOS, apply token budget cap,
    split 90/10 train/val, count tokens, save OUTPUT_JSONL.
    Returns HuggingFace DatasetDict with 'train' and 'test' splits.
    """
    t0 = time.time()
    print("Loading final filtered chunks ...")
    dataset = load_dataset("json", data_files={"train": CACHE_FINAL_CHUNKS}, split="train")
    print(f"Loaded {len(dataset):,} chunks from {CACHE_FINAL_CHUNKS}")

    # ── Append EOS ────────────────────────────────────────────────────────────
    EOS_TOKEN = tokenizer.eos_token
    if EOS_TOKEN is None:
        raise ValueError("Tokenizer has no EOS token. Set tokenizer.eos_token explicitly.")

    # CHANGED: num_proc=NUM_WORKERS=40 (was missing — single-core)
    dataset = dataset.map(
        lambda ex: _add_eos(ex, EOS_TOKEN),
        batched=True,
        num_proc=NUM_WORKERS,
        desc="Appending EOS",
    )

    # ── Token budget cap ──────────────────────────────────────────────────────
    sample_texts = dataset.select(range(min(1000, len(dataset))))["text"]
    avg_tokens   = sum(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in sample_texts
    ) / len(sample_texts)
    total_est_M  = len(dataset) * avg_tokens / 1e6

    print(f"Dataset size         : {len(dataset):,} chunks")
    print(f"Avg tokens/chunk     : {avg_tokens:.0f}")
    print(f"Estimated total      : {total_est_M:.0f}M tokens")

    if total_est_M > TOKEN_BUDGET_M:
        keep_n = int(len(dataset) * TOKEN_BUDGET_M / total_est_M)
        print(f"⚡ Capping to {keep_n:,} chunks (~{TOKEN_BUDGET_M}M tokens) "
              f"— {total_est_M/TOKEN_BUDGET_M:.1f}× fewer steps")
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(keep_n))
    else:
        print(f"✓ Under budget — using all {len(dataset):,} chunks")

    # ── Train / val split ─────────────────────────────────────────────────────
    dataset = dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    print(f"\n✓ Train chunks : {len(dataset['train']):,}")
    print(f"✓ Val   chunks : {len(dataset['test']):,}")

    # ── Save combined output ──────────────────────────────────────────────────
    concatenate_datasets([dataset["train"], dataset["test"]]).to_json(OUTPUT_JSONL)
    print(f"✓ Saved → {OUTPUT_JSONL}")

    # ── Token count (batched) ─────────────────────────────────────────────────
    print("\nCounting tokens (batched) ...")

    def _count_tokens(texts_list, label):
        total   = 0
        batches = [texts_list[i: i + TOKEN_BATCH] for i in range(0, len(texts_list), TOKEN_BATCH)]
        for batch in tqdm(batches, desc=label, unit="batch"):
            enc    = tokenizer.batch_encode_plus(batch, add_special_tokens=False)
            total += sum(len(ids) for ids in enc["input_ids"])
        return total

    train_tokens = _count_tokens(dataset["train"]["text"], "Train token count")
    val_tokens   = _count_tokens(dataset["test"]["text"],  "Val token count  ")

    print(f"\n✓ Total train tokens : {train_tokens:,}")
    print(f"✓ Total val tokens   : {val_tokens:,}")
    print(f"✓ Total tokens       : {train_tokens + val_tokens:,}")
    print(f"✓ Cell time          : {(time.time()-t0)/60:.1f} min")

    print("\n--- Sample training chunks ---")
    for row in dataset["train"][:3]["text"]:
        print("=" * 60)
        print(row[:300])

    return dataset
