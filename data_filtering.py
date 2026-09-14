# =============================================================================
# data_filtering.py — Filter and deduplicate chunks
#
# CHANGES from original (all three parallel stages updated):
#   - pandarallel.initialize(nb_workers=NUM_WORKERS)  → 40 (was 20)
#   - mp.Pool(processes=NUM_WORKERS)                  → 40 (was 20)
#   - ProcessPoolExecutor(max_workers=NUM_WORKERS)    → 40 (was 20)
# =============================================================================

import os
import re
import time
import hashlib
import multiprocessing as mp
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from pandarallel import pandarallel
from datasketch import MinHash, MinHashLSH
from config import (
    CACHE_RAW_CHUNKS, CACHE_FINAL_CHUNKS,
    NUM_WORKERS,           # CHANGED: now 40
    JACCARD_THRESHOLD, MINHASH_NUM_PERM, MINHASH_BATCH,
    MIN_TEXT_LEN, ALPHA_RATIO_THR,
)

# CHANGED: nb_workers=NUM_WORKERS = 40 (was hardcoded 16/20)
pandarallel.initialize(nb_workers=NUM_WORKERS, progress_bar=False, verbose=0)

# ── Filter functions ──────────────────────────────────────────────────────────

def _is_mostly_alpha(text, thr=ALPHA_RATIO_THR):
    return sum(c.isalpha() for c in text) / max(len(text), 1) > thr


_BPAT = re.compile(
    r"^\s*references\s*$|^\s*acknowledgment|^\s*funding\b|"
    r"^\s*conflict of interest|^\s*author contributions|"
    r"^\s*supplementary materials?|^\s*data availability|^\s*ethics statement",
    re.IGNORECASE | re.MULTILINE,
)


def _is_boilerplate(text):
    lines = text.strip().split("\n")
    if lines and _BPAT.search(lines[0]):
        return True
    return sum(1 for ln in lines if _BPAT.search(ln)) / max(len(lines), 1) > 0.3


def _is_english(text):
    from langdetect import detect, LangDetectException, DetectorFactory
    DetectorFactory.seed = 0
    try:
        return detect(str(text)[:500]) == "en"
    except LangDetectException:
        return False


def _compute_minhash_batch(args):
    """Worker: compute MinHash hashvalues for a batch of texts."""
    batch_texts, n_perm = args
    from datasketch import MinHash
    results = []
    for text in batch_texts:
        m = MinHash(num_perm=n_perm)
        for word in text.lower().split():
            m.update(word.encode("utf-8"))
        results.append(m.hashvalues.copy())
    return results


def run_filtering() -> pd.DataFrame:
    """
    Run all filter and dedup stages on the raw chunks cache.
    Checks cache first — skips all computation if final cache exists.
    Returns filtered DataFrame with column 'text'.
    """
    if os.path.exists(CACHE_FINAL_CHUNKS):
        print(f"✓ Cache hit: {CACHE_FINAL_CHUNKS}")
        chunk_df = pd.read_json(CACHE_FINAL_CHUNKS, lines=True)
        print(f"  Loaded {len(chunk_df):,} chunks — skipping filtering.")
        return chunk_df

    print(f"Loading {CACHE_RAW_CHUNKS} ...")
    chunk_df = pd.read_json(CACHE_RAW_CHUNKS, lines=True)
    print(f"Loaded {len(chunk_df):,} chunks\n")
    t0 = time.time()

    # ── Exact dedup ───────────────────────────────────────────────────────────
    before   = len(chunk_df)
    chunk_df = chunk_df.drop_duplicates(subset=["text"])
    print(f"After exact dedup    : {len(chunk_df):,}  (removed {before - len(chunk_df):,})")

    # ── MD5 hash dedup ────────────────────────────────────────────────────────
    chunk_df["_h"] = chunk_df["text"].apply(
        lambda t: hashlib.md5(" ".join(t.lower().split()).encode()).hexdigest()
    )
    chunk_df = chunk_df.drop_duplicates(subset=["_h"]).drop(columns=["_h"])
    print(f"After MD5 dedup      : {len(chunk_df):,}")

    # ── Length filter ─────────────────────────────────────────────────────────
    chunk_df = chunk_df[chunk_df["text"].str.strip().str.len() > MIN_TEXT_LEN].reset_index(drop=True)
    print(f"After length filter  : {len(chunk_df):,}\n")

    # ── Alpha filter — pandarallel (nb_workers=40) ────────────────────────────
    print("Alpha filter (parallel) ...")
    before   = len(chunk_df)
    chunk_df = chunk_df[chunk_df["text"].parallel_apply(_is_mostly_alpha)].reset_index(drop=True)
    print(f"After alpha filter   : {len(chunk_df):,}  (removed {before - len(chunk_df):,})\n")

    # ── Language detection — mp.Pool(processes=40) ────────────────────────────
    print("Language detection (parallel) ...")
    before = len(chunk_df)
    # CHANGED: processes=NUM_WORKERS = 40 (was 16/20)
    with mp.Pool(processes=NUM_WORKERS) as pool:
        mask = list(tqdm(
            pool.imap(_is_english, chunk_df["text"].tolist(), chunksize=500),
            total=len(chunk_df), desc="Lang detect", unit="chunk",
        ))
    chunk_df = chunk_df[mask].reset_index(drop=True)
    print(f"After lang filter    : {len(chunk_df):,}  (removed {before - len(chunk_df):,})\n")

    # ── Boilerplate filter — pandarallel (nb_workers=40) ─────────────────────
    print("Boilerplate filter (parallel) ...")
    before   = len(chunk_df)
    chunk_df = chunk_df[~chunk_df["text"].parallel_apply(_is_boilerplate)].reset_index(drop=True)
    print(f"After boilerplate    : {len(chunk_df):,}  (removed {before - len(chunk_df):,})\n")

    # ── MinHash Phase A: parallel hashing — ProcessPoolExecutor(40) ──────────
    texts_arr  = chunk_df["text"].tolist()
    mh_batches = [
        (texts_arr[i: i + MINHASH_BATCH], MINHASH_NUM_PERM)
        for i in range(0, len(texts_arr), MINHASH_BATCH)
    ]
    print(f"MinHash Phase A: hashing {len(texts_arr):,} chunks across {NUM_WORKERS} workers ...")
    t_mh            = time.time()
    ordered_results = [None] * len(mh_batches)

    # CHANGED: max_workers=NUM_WORKERS = 40 (was 16/20)
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_compute_minhash_batch, b): i for i, b in enumerate(mh_batches)}
        with tqdm(total=len(texts_arr), desc="Computing hashes", unit="chunk") as pbar:
            for fut in as_completed(futs):
                bi = futs[fut]
                batch_hv = fut.result()
                ordered_results[bi] = batch_hv
                pbar.update(len(batch_hv))
                pbar.set_postfix(
                    batches = f"{sum(1 for x in ordered_results if x is not None)}/{len(mh_batches)}",
                    elapsed = f"{(time.time()-t_mh)/60:.1f}m",
                )

    all_hashvalues = [hv for batch in ordered_results for hv in batch]
    print(f"  Hash phase done in {(time.time()-t_mh)/60:.1f} min")

    # Rebuild MinHash objects from precomputed hashvalues
    print("  Rebuilding MinHash objects from hashvalues ...")
    minhashes = []
    for hv in tqdm(all_hashvalues, desc="Rebuilding MinHash", unit="chunk"):
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        m.hashvalues = np.array(hv, dtype=np.uint64)
        minhashes.append(m)

    # ── MinHash Phase B: sequential LSH (stateful — cannot parallelize) ───────
    print(f"\nMinHash Phase B: LSH dedup (sequential, stateful) ...")
    lsh          = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    keep_indices = []
    t_lsh        = time.time()

    for idx, mh in enumerate(tqdm(minhashes, desc="LSH dedup", unit="chunk")):
        if not lsh.query(mh):
            lsh.insert(str(idx), mh)
            keep_indices.append(idx)
        if (idx + 1) % 50_000 == 0:
            el  = time.time() - t_lsh
            rem = (len(minhashes) - idx - 1) / ((idx + 1) / el)
            print(f"  LSH [{idx+1:,}/{len(minhashes):,}]  kept: {len(keep_indices):,}  |"
                  f"  elapsed: {el/60:.1f}m  |  ~{rem/60:.1f}m remaining")

    before   = len(chunk_df)
    chunk_df = chunk_df.iloc[keep_indices].reset_index(drop=True)
    print(f"\nAfter MinHash LSH    : {len(chunk_df):,}  (removed {before - len(chunk_df):,} near-dupes)")
    print(f"MinHash total time   : {(time.time()-t_mh)/60:.1f} min\n")

    # ── Shuffle + save ────────────────────────────────────────────────────────
    chunk_df = chunk_df.sample(frac=1, random_state=3407).reset_index(drop=True)
    chunk_df.to_json(CACHE_FINAL_CHUNKS, orient="records", lines=True)
    print(f"✓ Saved → {CACHE_FINAL_CHUNKS}  ({len(chunk_df):,} rows)")
    print(f"✓ Total filter time  : {(time.time()-t0)/60:.1f} min")
    return chunk_df
