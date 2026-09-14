# =============================================================================
# data_chunking.py — Parallel chunking of raw COVID papers into token windows
#
# CHANGE from original: NUM_WORKERS sourced from config (20 → 40)
# Uses local tokenizer copy to avoid HF rate limits with 40 parallel workers.
# =============================================================================

import os
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from config import (
    RAW_PAPERS_JSONL, CACHE_RAW_CHUNKS,
    CHUNK_SIZE, OVERLAP, CHUNK_BATCH_SIZE,
    LOCAL_TOK_DIR, NUM_WORKERS,   # CHANGED: NUM_WORKERS now = 40
)


def _chunk_batch(args):
    """
    Worker function — runs in a subprocess.
    Loads tokenizer from LOCAL_TOK_DIR (disk) to avoid 40 simultaneous
    HuggingFace API calls which would trigger 429 rate limiting.
    """
    batch_texts, tok_dir, c_size, ovlp = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    out = []
    for text in batch_texts:
        if not text or not str(text).strip():
            continue
        toks = tok.encode(str(text))
        s = 0
        while s < len(toks):
            e = min(s + c_size, len(toks))
            out.append(tok.decode(toks[s:e], skip_special_tokens=True))
            if e == len(toks):
                break
            s += c_size - ovlp
    return out


def run_chunking(tokenizer) -> pd.DataFrame:
    """
    Chunk all papers in RAW_PAPERS_JSONL into CHUNK_SIZE token windows.
    Checks cache first — skips all computation if cache exists.
    Returns DataFrame with column 'text'.
    """
    if os.path.exists(CACHE_RAW_CHUNKS):
        print(f"✓ Cache hit: {CACHE_RAW_CHUNKS}")
        chunk_df = pd.read_json(CACHE_RAW_CHUNKS, lines=True)
        print(f"  Loaded {len(chunk_df):,} chunks — skipping chunking.")
        return chunk_df

    # Save tokenizer to disk so all 40 workers load from local path
    print(f"Saving tokenizer to {LOCAL_TOK_DIR} for worker use...")
    tokenizer.save_pretrained(LOCAL_TOK_DIR)

    print(f"Loading {RAW_PAPERS_JSONL} ...")
    df = pd.read_json(RAW_PAPERS_JSONL, lines=True)
    total_docs = len(df)
    print(f"Raw papers: {total_docs:,}\n")

    texts = df["text"].tolist()
    batches = [
        (texts[i: i + CHUNK_BATCH_SIZE], LOCAL_TOK_DIR, CHUNK_SIZE, OVERLAP)
        for i in range(0, len(texts), CHUNK_BATCH_SIZE)
    ]

    all_chunks = []
    docs_done  = 0
    t0         = time.time()

    # CHANGED: max_workers=NUM_WORKERS = 40 (was hardcoded 16/20)
    print(f"Chunking {total_docs:,} docs → {len(batches):,} batches × {NUM_WORKERS} workers ...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_chunk_batch, b): len(b[0]) for b in batches}
        with tqdm(total=len(batches), desc="Chunking batches", unit="batch") as pbar:
            for fut in as_completed(futs):
                all_chunks.extend(fut.result())
                docs_done += futs[fut]
                pbar.update(1)
                pbar.set_postfix(
                    chunks  = f"{len(all_chunks):,}",
                    docs    = f"{docs_done:,}/{total_docs:,}",
                    elapsed = f"{(time.time()-t0)/60:.1f}m",
                )

    print(f"\n✓ Total chunks : {len(all_chunks):,}")
    print(f"✓ Chunking time: {(time.time()-t0)/60:.1f} min")

    chunk_df = pd.DataFrame({"text": all_chunks})
    chunk_df.to_json(CACHE_RAW_CHUNKS, orient="records", lines=True)
    print(f"✓ Saved → {CACHE_RAW_CHUNKS}  ({len(chunk_df):,} rows)")
    return chunk_df
