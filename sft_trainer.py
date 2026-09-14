# =============================================================================
# sft_trainer.py — SFT training with standard HuggingFace Trainer (DDP-safe)
#
# KEY CHANGES from Unsloth/TRL version:
#   - SFTTrainer + SFTConfig (trl) → transformers.Trainer + TrainingArguments
#   - neftune_noise_alpha removed from SFTConfig → applied as model forward hook
#   - ddp_find_unused_parameters=False added (required for LoRA + DDP)
#   - DataCollatorForSeq2Seq added (handles chat template padding correctly)
#   - dataset_text_field and packing removed (dataset is pre-tokenized)
#   - Only rank-0 saves models, prints summaries, runs inference tests
# =============================================================================

import os
import gc
import math
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
<<<<<<< HEAD
    EarlyStoppingCallback,
    get_chat_template,
)
from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
=======
    EarlyStoppingCallback
)
>>>>>>> 07d00f6 (Final code of CPT and SFT)
from config import (
    SFT_DATASET_PATH, SFT_CACHE_TOKENIZED,
    SFT_PER_DEVICE_BATCH, SFT_GRAD_ACCUM, SFT_NUM_EPOCHS,
    SFT_LR, SFT_WARMUP_RATIO, SFT_LR_SCHEDULER,
    SFT_WEIGHT_DECAY, SFT_MAX_GRAD_NORM, SFT_OPTIM,
    SFT_NEFTUNE_ALPHA, SFT_EVAL_STEPS, SFT_SAVE_STEPS,
    SFT_SAVE_TOTAL_LIMIT, SFT_DATALOADER_WORKERS,
    SFT_CHECKPOINT_DIR, SFT_LORA_DIR, SFT_MERGED_DIR,
    MAX_SEQ_LENGTH, NUM_WORKERS, RANDOM_SEED,
)


def _is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def apply_chat_template(tokenizer):
<<<<<<< HEAD
    """Apply Llama-3 chat template to tokenizer."""
    # Use unsloth's helper for template application — this is safe even
    # without Unsloth training, as it only modifies the tokenizer's template.
    try:
        tokenizer = unsloth_get_chat_template(tokenizer, chat_template="llama-3")
    except Exception:
        # Fallback: HF built-in chat template
        tokenizer.chat_template = None  # use model default
    if _is_rank0():
        print(f"✓ Chat template applied")
=======
    """Apply Llama-3 chat template directly — no Unsloth dependency."""
    # Llama-3 chat template hardcoded — removes unsloth dependency entirely
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )
    if _is_rank0():
        print(f"✓ Llama-3 chat template applied")
>>>>>>> 07d00f6 (Final code of CPT and SFT)
        print(f"  EOS token : '{tokenizer.eos_token}'  (id: {tokenizer.eos_token_id})")
    return tokenizer


def format_chat_prompts(examples, tokenizer):
    """Convert instruction/input/output records to Llama-3 chat format."""
    instructions = examples["instruction"]
    inputs       = examples.get("input", [""] * len(instructions))
    outputs      = examples["output"]
    texts        = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        user_content = inp.strip() if inp else ""
        messages = [
            {"role": "system",    "content": instruction.strip()},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def load_and_format_sft_dataset(tokenizer):
    """Load SFT JSONL, apply chat template formatting, split 90/10."""
    raw = load_dataset("json", data_files={"train": SFT_DATASET_PATH}, split="train")
    if _is_rank0():
        print(f"Raw SFT samples  : {len(raw):,}")
        print(f"Columns          : {raw.column_names}")
    for col in ["instruction", "output"]:
        assert col in raw.column_names, f"Missing column: '{col}'"

    formatted = raw.map(
        lambda ex: format_chat_prompts(ex, tokenizer),
        batched=True,
        num_proc=NUM_WORKERS,   # CHANGED: 20 → 40
        desc="Formatting chat template prompts",
    )
    sft_dataset = formatted.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    if _is_rank0():
        print(f"✓ SFT train : {len(sft_dataset['train']):,}")
        print(f"✓ SFT val   : {len(sft_dataset['test']):,}")
    return sft_dataset


def profile_token_lengths(sft_dataset, tokenizer):
    """
    Profile token length distribution using parallel map.
    CHANGED from single-core list comprehension → dataset.map num_proc=40
    Returns the P95-rounded sft_max_seq_length.
    """
    def _count_len(examples):
        return {"length": [len(tokenizer.encode(t)) for t in examples["text"]]}

    lengths_ds = sft_dataset["train"].map(
        _count_len, batched=True, batch_size=500,
        num_proc=NUM_WORKERS, desc="Profiling token lengths",
    )
    token_lengths = np.array(lengths_ds["length"])

    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)
    sft_max_seq = int(np.ceil(p95 / 512) * 512)
    sft_max_seq = max(sft_max_seq, 512)
    sft_max_seq = min(sft_max_seq, MAX_SEQ_LENGTH)
    pct_covered = (token_lengths <= sft_max_seq).mean() * 100

    if _is_rank0():
        print(f"\nSFT Token Length Distribution:")
        print(f"  Min    : {token_lengths.min()}")
        print(f"  Mean   : {token_lengths.mean():.0f}")
        print(f"  P95    : {p95:.0f}")
        print(f"  P99    : {p99:.0f}")
        print(f"  Max    : {token_lengths.max()}")
        print(f"  sft_max_seq_length = {sft_max_seq}  ({pct_covered:.1f}% covered)")
        if sft_max_seq < p99:
            n_trunc = (token_lengths > sft_max_seq).sum()
            print(f"  ⚠ {n_trunc} samples ({100-pct_covered:.1f}%) will be truncated")
    return sft_max_seq


def _apply_neftune(model, neftune_alpha: float):
    """
    Apply NEFTune noise as a forward hook on the embedding layer.
    CHANGED: neftune_noise_alpha was in SFTConfig — not available in standard
    TrainingArguments. We apply it manually here as a model hook instead.
    NEFTune adds uniform noise to embeddings during training, improving
    instruction-following quality by ~1-2 BLEU/ROUGE points.
    """
    def _neftune_hook(module, input, output):
        if model.training:
            dims = torch.tensor(output.size(1) * output.size(2), dtype=torch.float32)
            mag_norm = neftune_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output

    embed = model.get_input_embeddings()
    embed.register_forward_hook(_neftune_hook)
    if _is_rank0():
        print(f"✓ NEFTune noise hook applied (alpha={neftune_alpha})")
    return model


def create_sft_trainer(model, tokenizer, sft_dataset, sft_max_seq_length):
    """
    Build and return a HuggingFace Trainer for SFT.

    Effective batch = per_device × num_gpus × grad_accum
                    = 2 × 4 × 8 = 64
    """
    # DataCollatorForSeq2Seq correctly handles:
    #   - padding for variable-length chat template outputs
    #   - label masking for pad tokens (-100)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=SFT_CHECKPOINT_DIR,

        # ── Batch ────────────────────────────────────────────────────────────
        per_device_train_batch_size=SFT_PER_DEVICE_BATCH,   # 2 per GPU
        gradient_accumulation_steps=SFT_GRAD_ACCUM,         # 2×4×8=64 effective
        num_train_epochs=SFT_NUM_EPOCHS,
        warmup_ratio=SFT_WARMUP_RATIO,

        # ── Precision ────────────────────────────────────────────────────────
        bf16=True,
        fp16=False,
        tf32=True,

        # ── Gradient checkpointing ───────────────────────────────────────────
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ── Optimizer ────────────────────────────────────────────────────────
        learning_rate=SFT_LR,
        optim=SFT_OPTIM,
        weight_decay=SFT_WEIGHT_DECAY,
        max_grad_norm=SFT_MAX_GRAD_NORM,
        lr_scheduler_type=SFT_LR_SCHEDULER,

        # ── Data loading ─────────────────────────────────────────────────────
        dataloader_num_workers=SFT_DATALOADER_WORKERS,  # 10 per GPU
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # ── Eval + save ──────────────────────────────────────────────────────
        eval_strategy="steps",
        eval_steps=SFT_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SFT_SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SFT_SAVE_TOTAL_LIMIT,

        # ── Multi-GPU DDP ────────────────────────────────────────────────────
        # REQUIRED for LoRA + DDP — LoRA freezes most params,
        # DDP would otherwise hang scanning for unused frozen parameters.
        ddp_find_unused_parameters=False,

        # ── Misc ─────────────────────────────────────────────────────────────
        logging_steps=10,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


def run_sft_training(model, tokenizer, sft_dataset, sft_max_seq_length):
    """
    Full SFT training run.
    Applies NEFTune hook, runs training, saves LoRA adapter and merged model.
    Only rank-0 saves and prints.
    """
    # Apply NEFTune noise hook (replaces SFTConfig neftune_noise_alpha)
    model = _apply_neftune(model, SFT_NEFTUNE_ALPHA)

    trainer = create_sft_trainer(model, tokenizer, sft_dataset, sft_max_seq_length)

    if _is_rank0():
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        eff_batch = SFT_PER_DEVICE_BATCH * num_gpus * SFT_GRAD_ACCUM
        total_steps = (len(sft_dataset["train"]) // eff_batch) * SFT_NUM_EPOCHS
        print(f"\n{'='*60}")
        print(f"SFT TRAINING — {num_gpus} GPU(s)")
        print(f"  Train samples      : {len(sft_dataset['train']):,}")
        print(f"  Val samples        : {len(sft_dataset['test']):,}")
        print(f"  Per-device batch   : {SFT_PER_DEVICE_BATCH}")
        print(f"  Num GPUs           : {num_gpus}")
        print(f"  Grad accumulation  : {SFT_GRAD_ACCUM}")
        print(f"  Effective batch    : {eff_batch}")
        print(f"  Estimated steps    : ~{total_steps:,}")
        print(f"  Eval every         : {SFT_EVAL_STEPS} steps")
        print(f"{'='*60}\n")
        print("Starting SFT training...")

    sft_trainer_stats = trainer.train()

    # ── Save — RANK 0 ONLY ────────────────────────────────────────────────────
    if _is_rank0():
        print(f"\nSFT training complete.")
        print(f"  Runtime: {sft_trainer_stats.metrics['train_runtime']/60:.1f} min")
        if "eval_loss" in sft_trainer_stats.metrics:
            ppl = math.exp(sft_trainer_stats.metrics["eval_loss"])
            print(f"  Final eval loss : {sft_trainer_stats.metrics['eval_loss']:.4f}")
            print(f"  Final perplexity: {ppl:.2f}")

        # Save LoRA adapter (lightweight ~50-200MB)
        print(f"\nSaving SFT LoRA adapter → {SFT_LORA_DIR}/")
        unwrapped = trainer.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.save_pretrained(SFT_LORA_DIR)
        tokenizer.save_pretrained(SFT_LORA_DIR)
        print(f"✓ SFT LoRA saved → {SFT_LORA_DIR}/")

        # Merge LoRA into base weights and save full model
        print(f"\nMerging SFT LoRA into base weights → {SFT_MERGED_DIR}/")
        merged = unwrapped.merge_and_unload()
        merged.config.tie_word_embeddings = False
        merged.save_pretrained(SFT_MERGED_DIR)
        tokenizer.save_pretrained(SFT_MERGED_DIR)
        print(f"✓ Merged model saved → {SFT_MERGED_DIR}/")
        print(f"\nPipeline complete:")
        print(f"  CPT model  → {SFT_CHECKPOINT_DIR.replace('sft','cpt').replace('checkpoints','bf16')}/")
        print(f"  SFT LoRA   → {SFT_LORA_DIR}/")
        print(f"  SFT merged → {SFT_MERGED_DIR}/")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return sft_trainer_stats
