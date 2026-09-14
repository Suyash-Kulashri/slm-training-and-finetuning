# =============================================================================
# cpt_trainer.py — CPT training with standard HuggingFace Trainer (DDP-safe)
#
# KEY CHANGES from Unsloth version:
#   - UnslothTrainer → transformers.Trainer
#   - UnslothTrainingArguments → transformers.TrainingArguments
#   - embedding_learning_rate removed (Unsloth-only param)
#   - adamw_8bit → adamw_bnb_8bit (bitsandbytes naming in HF)
#   - ddp_find_unused_parameters=False added (required for LoRA + DDP)
#   - DataCollatorForLanguageModeling added (handles labels for CLM)
#   - gradient_checkpointing set in TrainingArguments (model already enabled it)
#   - Only rank-0 process saves merged model and prints summaries
# =============================================================================

import os
import re
import gc
import math
import torch
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from config import (
    CPT_PER_DEVICE_BATCH, CPT_GRAD_ACCUM, CPT_NUM_EPOCHS,
    CPT_LR, CPT_WARMUP_RATIO, CPT_LR_SCHEDULER,
    CPT_WEIGHT_DECAY, CPT_MAX_GRAD_NORM, CPT_OPTIM,
    CPT_EVAL_STEPS, CPT_SAVE_STEPS, CPT_SAVE_TOTAL_LIMIT,
    CPT_DATALOADER_WORKERS, CPT_CHECKPOINT_DIR, CPT_MERGED_DIR,
    RANDOM_SEED,
)


def _is_rank0() -> bool:
    """True if this is the main process (rank 0) or not using distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _get_valid_checkpoints(checkpoint_dir: str):
    """Return sorted list of (step, path) for all valid checkpoints."""
    if not os.path.isdir(checkpoint_dir):
        return []
    checkpoints = []
    for d in os.listdir(checkpoint_dir):
        m = re.match(r"checkpoint-(\d+)$", d)
        if not m:
            continue
        full_path = os.path.join(checkpoint_dir, d)
        if os.path.exists(os.path.join(full_path, "config.json")):
            checkpoints.append((int(m.group(1)), full_path))
    return sorted(checkpoints, key=lambda x: x[0])


def create_cpt_trainer(model, tokenizer, dataset):
    """
    Build and return a HuggingFace Trainer for CPT.

    Effective batch size = per_device × num_gpus × grad_accum
                        = 4 × 4 × 4 = 64
    (same training dynamics as the original single-GPU config of 16×4=64)
    """
    # DataCollatorForLanguageModeling handles:
    #   - right-padding sequences to max length in the batch
    #   - setting labels = input_ids (CLM objective)
    #   - masking pad tokens in labels with -100
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,      # causal LM, not masked LM
        pad_to_multiple_of=8,  # efficient for tensor cores
    )

    training_args = TrainingArguments(
        output_dir=CPT_CHECKPOINT_DIR,

        # ── Batch ────────────────────────────────────────────────────────────
        per_device_train_batch_size=CPT_PER_DEVICE_BATCH,   # 4 per GPU
        gradient_accumulation_steps=CPT_GRAD_ACCUM,         # 4×4×4=64 effective
        num_train_epochs=CPT_NUM_EPOCHS,
        warmup_ratio=CPT_WARMUP_RATIO,

        # ── Precision ────────────────────────────────────────────────────────
        bf16=True,
        fp16=False,
        tf32=True,                  # SXM4 supports TF32 — free ~10% throughput

        # ── Gradient checkpointing ───────────────────────────────────────────
        # Already enabled on model via gradient_checkpointing_enable().
        # Setting here too ensures TrainingArguments knows about it.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ── Optimizer ────────────────────────────────────────────────────────
        learning_rate=CPT_LR,
        # NOTE: embedding_learning_rate is Unsloth-only — removed here.
        # All parameters use the same LR in standard TrainingArguments.
        optim=CPT_OPTIM,            # adamw_bnb_8bit (bitsandbytes 8-bit Adam)
        weight_decay=CPT_WEIGHT_DECAY,
        max_grad_norm=CPT_MAX_GRAD_NORM,
        lr_scheduler_type=CPT_LR_SCHEDULER,

        # ── Data loading ─────────────────────────────────────────────────────
        # 48 cores total / 4 GPUs = 12 per GPU, use 10 to leave headroom
        dataloader_num_workers=CPT_DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # ── Eval + save ──────────────────────────────────────────────────────
        eval_strategy="steps",
        eval_steps=CPT_EVAL_STEPS,
        save_strategy="steps",
        save_steps=CPT_SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=CPT_SAVE_TOTAL_LIMIT,

        # ── Multi-GPU DDP ────────────────────────────────────────────────────
        # ddp_find_unused_parameters=False is REQUIRED for LoRA + DDP.
        # LoRA freezes most parameters. DDP would otherwise scan all params
        # looking for unused ones, which hangs training with frozen weights.
        ddp_find_unused_parameters=False,

        # ── Misc ─────────────────────────────────────────────────────────────
        logging_steps=10,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


def run_cpt_training(model, tokenizer, dataset):
    """
    Run CPT training with checkpoint resume support.
    Only rank-0 prints summaries and saves the merged model.
    """
    trainer = create_cpt_trainer(model, tokenizer, dataset)

    # Sanity print — only rank 0 to avoid duplicate output
    if _is_rank0():
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        eff_batch = CPT_PER_DEVICE_BATCH * num_gpus * CPT_GRAD_ACCUM
        total_steps = (len(dataset["train"]) // eff_batch) * CPT_NUM_EPOCHS
        print(f"\n{'='*60}")
        print(f"CPT TRAINING — {num_gpus} GPU(s)")
        print(f"  Train samples      : {len(dataset['train']):,}")
        print(f"  Val samples        : {len(dataset['test']):,}")
        print(f"  Per-device batch   : {CPT_PER_DEVICE_BATCH}")
        print(f"  Num GPUs           : {num_gpus}")
        print(f"  Grad accumulation  : {CPT_GRAD_ACCUM}")
        print(f"  Effective batch    : {eff_batch}")
        print(f"  Estimated steps    : ~{total_steps:,}")
        print(f"  Eval every         : {CPT_EVAL_STEPS} steps")
        print(f"  ~{total_steps // CPT_EVAL_STEPS} evaluations total")
        est_hrs = total_steps * 0.4 / 3600  # ~0.4s/step on 4x A100-40GB
        print(f"  Estimated time     : ~{est_hrs:.1f} hours")
        print(f"{'='*60}\n")

    # Resume from latest valid checkpoint if available
    valid = _get_valid_checkpoints(CPT_CHECKPOINT_DIR)
    if valid:
        last_step, last_ckpt = valid[-1]
        if _is_rank0():
            print(f"Resuming from checkpoint step {last_step}: {last_ckpt}")
        trainer_stats = trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        if _is_rank0():
            print("No valid checkpoint found — starting from scratch.")
        trainer_stats = trainer.train()

    # ── Save merged model — RANK 0 ONLY ──────────────────────────────────────
    # In DDP every process has a full copy of the model, but we only want
    # one process to write to disk to avoid race conditions and duplicate files.
    if _is_rank0():
        print(f"\nCPT training complete.")
        print(f"  Runtime: {trainer_stats.metrics['train_runtime']/3600:.2f} hours")
        if "eval_loss" in trainer_stats.metrics:
            ppl = math.exp(trainer_stats.metrics["eval_loss"])
            print(f"  Final eval loss : {trainer_stats.metrics['eval_loss']:.4f}")
            print(f"  Final perplexity: {ppl:.2f}")

        print(f"\nSaving merged CPT model → {CPT_MERGED_DIR}/")
        # Unwrap DDP wrapper before saving
        unwrapped = trainer.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.save_pretrained(CPT_MERGED_DIR)
        tokenizer.save_pretrained(CPT_MERGED_DIR)
        print(f"✓ Saved → {CPT_MERGED_DIR}/")

    # Barrier: all processes wait until rank-0 finishes saving
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Free GPU memory before SFT
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return trainer_stats
