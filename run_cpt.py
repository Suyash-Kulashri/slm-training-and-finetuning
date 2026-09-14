# =============================================================================
# run_cpt.py — CPT pipeline: data → dataset → tokenize → train → save merged
#
# CHANGES for torchrun multi-GPU:
#   - if __name__ == "__main__" guard added (required for torchrun)
#   - dist.init_process_group called at entry (torchrun sets env vars automatically)
#   - Only rank-0 runs data pipeline (preprocessing is single-process)
#   - dist.barrier() used to sync all ranks after data is ready
#   - rank-0 check before prints/saves in training (handled in cpt_trainer.py)
#
# LAUNCH (multi-GPU):
#   torchrun --nproc_per_node=4 --master_port=29500 run_cpt.py
#
# LAUNCH (single GPU, for testing):
#   python run_cpt.py
# =============================================================================

import os
import torch
import torch.distributed as dist

from config import CACHE_CPT_TOKENIZED, MAX_SEQ_LENGTH
from model_loader import load_base_model_and_tokenizer, get_peft_model_cpt
from data_chunking import run_chunking
from data_filtering import run_filtering
from data_dataset import build_dataset
from data_tokenize_cache import tokenize_and_cache_cpt
from cpt_trainer import run_cpt_training


def _setup_distributed():
    """
    Initialize distributed process group if launched via torchrun.
    torchrun automatically sets LOCAL_RANK, RANK, WORLD_SIZE env vars.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return 0  # single-process fallback


def _is_rank0():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def run_data_only():
    """
    Run data pipeline only (no model loading, no GPU required).
    Called by train_model_QAT16_final.py --data-only.
    Runs entirely on CPU using all 40 cores.
    """
    # Load tokenizer only (no model) — needed for chunking + EOS + tokenize
    from transformers import AutoTokenizer
    from config import BASE_MODEL_NAME, LOCAL_TOK_DIR
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_chunking(tokenizer)
    run_filtering()
    dataset = build_dataset(tokenizer)
    tokenize_and_cache_cpt(dataset, tokenizer, MAX_SEQ_LENGTH)
    print("\n✓ Data pipeline complete. All caches written.")


def run_cpt_pipeline(skip_data: bool = False):
    """
    Full CPT pipeline: data (optional) → model load → train → save.
    Designed to be called from torchrun — each process runs this function
    on its assigned GPU.
    """
    local_rank = _setup_distributed()

    # ── Data pipeline: only rank-0 runs this ─────────────────────────────────
    # Preprocessing is not distributed — it uses multiprocessing internally.
    # Running it on all 4 ranks simultaneously would cause file conflicts.
    if not skip_data and _is_rank0():
        from transformers import AutoTokenizer
        from config import BASE_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        run_chunking(tokenizer)
        run_filtering()
        dataset_raw = build_dataset(tokenizer)
        tokenize_and_cache_cpt(dataset_raw, tokenizer, MAX_SEQ_LENGTH)

    # All ranks wait until rank-0 finishes writing caches
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # ── Load model and tokenizer on every rank ────────────────────────────────
    if _is_rank0():
        print("\nLoading base model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer()
    model = get_peft_model_cpt(model)

    # ── Load pre-tokenized dataset (all ranks read from same cache) ───────────
    dataset = tokenize_and_cache_cpt(None, tokenizer, MAX_SEQ_LENGTH)

    # ── Train ─────────────────────────────────────────────────────────────────
    run_cpt_training(model, tokenizer, dataset)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun entry point
    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically
    # Each of the 4 processes runs this block on its own GPU
    run_cpt_pipeline(skip_data=False)
