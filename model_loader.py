# =============================================================================
# model_loader.py — Model and tokenizer loading for multi-GPU torchrun training
#
# KEY CHANGES from Unsloth version:
#   - FastLanguageModel replaced with AutoModelForCausalLM + PEFT LoraConfig
#   - attn_implementation="flash_attention_2" replaces Unsloth's fused kernels
#   - device_map=None — torchrun/DDP places each process on its own GPU
#   - gradient_checkpointing_enable() replaces use_gradient_checkpointing="unsloth"
#   - embedding_learning_rate removed (Unsloth-specific, not in standard Trainer)
# =============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from config import (
    BASE_MODEL_NAME, CPT_MERGED_DIR, MAX_SEQ_LENGTH,
    CPT_LORA_R, CPT_LORA_ALPHA, CPT_LORA_DROPOUT,
    CPT_USE_RSLORA, CPT_TARGET_MODULES,
    SFT_LORA_R, SFT_LORA_ALPHA, SFT_LORA_DROPOUT,
    SFT_USE_RSLORA, SFT_TARGET_MODULES,
    RANDOM_SEED,
)


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer and ensure pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_base_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load base model in BF16 with Flash Attention 2.

    device_map=None is REQUIRED for torchrun/DDP.
    Each torchrun process calls .to(local_rank) itself via the Trainer.
    Using device_map="auto" with DDP causes NCCL hangs.

    flash_attention_2 replaces Unsloth's fused attention kernels.
    Requires: pip install flash-attn --no-build-isolation
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,   # MUST be None for DDP — torchrun handles placement
    )
    model.config.tie_word_embeddings = False
    return model


def load_base_model_and_tokenizer():
    """
    Load Llama-3.1-8B base model + tokenizer for CPT.
    Returns (model, tokenizer).
    """
    tokenizer = _load_tokenizer(BASE_MODEL_NAME)
    model = _load_base_model(BASE_MODEL_NAME)
    return model, tokenizer


def get_peft_model_cpt(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Apply LoRA for CPT (Continued Pre-Training).

    Includes embed_tokens + lm_head in target_modules — required for CPT
    domain adaptation so the model can learn new token distributions.

    gradient_checkpointing_enable() replaces Unsloth's
    use_gradient_checkpointing="unsloth". On A100-SXM4-40GB this is
    essential — reduces activation memory ~60%, allowing batch=4 at seq=2048.
    """
    lora_config = LoraConfig(
        r=CPT_LORA_R,
        lora_alpha=CPT_LORA_ALPHA,
        target_modules=CPT_TARGET_MODULES,
        lora_dropout=CPT_LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=CPT_USE_RSLORA,
    )
    model = get_peft_model(model, lora_config)

    # gradient_checkpointing: replaces Unsloth's "unsloth" mode
    # use_reentrant=False is required for DDP compatibility
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model.print_trainable_parameters()
    return model


def load_cpt_model_for_sft():
    """
    Load the merged CPT model from disk for SFT fine-tuning.
    Returns (model, tokenizer).
    """
    tokenizer = _load_tokenizer(CPT_MERGED_DIR)
    model = _load_base_model(CPT_MERGED_DIR)
    return model, tokenizer


def get_peft_model_sft(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Apply LoRA for SFT (Supervised Fine-Tuning).

    Does NOT include embed_tokens/lm_head — SFT only needs to adapt
    attention and FFN layers to follow instructions, not learn new tokens.
    use_rslora=False for SFT (standard LoRA is fine at r=16).
    """
    lora_config = LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        target_modules=SFT_TARGET_MODULES,
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=SFT_USE_RSLORA,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    return model
