# =============================================================================
# run_sft.py — SFT pipeline: load CPT model → data → tokenize → train → save
#
# CHANGES for torchrun multi-GPU:
#   - if __name__ == "__main__" guard added (required for torchrun)
#   - dist.init_process_group called at entry
#   - Only rank-0 runs data formatting and profiling
#   - dist.barrier() used to sync ranks after data/tokenization is ready
#   - rank-0 check before prints/saves (handled in sft_trainer.py)
#
# LAUNCH (multi-GPU):
#   torchrun --nproc_per_node=4 --master_port=29500 run_sft.py
#
# LAUNCH (single GPU, for testing):
#   python run_sft.py
# =============================================================================

import os
import torch
import torch.distributed as dist

from config import MAX_SEQ_LENGTH, SFT_CACHE_TOKENIZED
from model_loader import load_cpt_model_for_sft, get_peft_model_sft
from sft_trainer import (
    apply_chat_template,
    load_and_format_sft_dataset,
    profile_token_lengths,
    run_sft_training,
)
from data_tokenize_cache import tokenize_and_cache_sft


def _setup_distributed():
    """Initialize distributed process group if launched via torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return 0


def _is_rank0():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def run_sft_pipeline():
    """
    Full SFT pipeline: load CPT model → format data → tokenize → train → save.
    Designed to run under torchrun — each process handles its own GPU.
    """
    local_rank = _setup_distributed()

    # ── Load CPT model + tokenizer on every rank ──────────────────────────────
    if _is_rank0():
        print("\nLoading CPT model for SFT...")
    model, tokenizer = load_cpt_model_for_sft()

    # ── Apply chat template ───────────────────────────────────────────────────
    tokenizer = apply_chat_template(tokenizer)

    # ── SFT data pipeline: only rank-0 runs formatting and profiling ──────────
    # (same reason as CPT: multiprocessing-based, not distributed)
    sft_max_seq_length = MAX_SEQ_LENGTH  # default, overridden by profiling below

    if _is_rank0():
        sft_dataset_raw = load_and_format_sft_dataset(tokenizer)
        sft_max_seq_length = profile_token_lengths(sft_dataset_raw, tokenizer)
        tokenize_and_cache_sft(sft_dataset_raw, tokenizer, sft_max_seq_length)

    # Broadcast sft_max_seq_length from rank-0 to all other ranks
    if dist.is_available() and dist.is_initialized():
        # Broadcast as a single-element tensor
        seq_len_tensor = torch.tensor(
            [sft_max_seq_length], dtype=torch.int32,
            device=f"cuda:{local_rank}"
        )
        dist.broadcast(seq_len_tensor, src=0)
        sft_max_seq_length = int(seq_len_tensor.item())
        dist.barrier()

    # ── All ranks load from tokenized cache ───────────────────────────────────
    sft_dataset = tokenize_and_cache_sft(None, tokenizer, sft_max_seq_length)

    # ── Apply LoRA for SFT ────────────────────────────────────────────────────
    model = get_peft_model_sft(model)

    # ── Train ─────────────────────────────────────────────────────────────────
    run_sft_training(model, tokenizer, sft_dataset, sft_max_seq_length)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_sft_pipeline()
