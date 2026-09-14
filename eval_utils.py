# =============================================================================
# eval_utils.py — Evaluation: perplexity, CPT completions, SFT Q&A spot-checks
#
# NOTE for multi-GPU: these functions must only run on rank-0.
# They are called from train_model_QAT16_final.py after training completes,
# which may still be inside a torchrun context.
# All public functions check _is_rank0() at entry and return silently on
# non-zero ranks to avoid duplicate output and file writes.
# =============================================================================

import math
import torch
import torch.distributed as dist


def _is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ── CPT evaluation ────────────────────────────────────────────────────────────

CPT_SPOT_CHECK_PROMPTS = [
    "COVID-19 is caused by",
    "The spike protein of SARS-CoV-2",
    "Common symptoms of COVID-19 include",
    "mRNA vaccines work by",
    "Long COVID is characterized by",
    "The ACE2 receptor is",
    "PCR testing for SARS-CoV-2 involves",
    "Cytokine storm in COVID-19 refers to",
]


def run_cpt_eval(model=None, tokenizer=None, model_path: str = None):
    """
    Evaluate the CPT model:
      1. Perplexity on the validation set (requires model + eval dataset)
      2. Free-form completion spot-checks on COVID-19 prompts

    Only runs on rank-0. If model/tokenizer are None, loads from model_path
    (defaults to config.CPT_MERGED_DIR).
    """
    if not _is_rank0():
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_from_disk
    from config import CPT_MERGED_DIR, CACHE_CPT_TOKENIZED, MAX_SEQ_LENGTH

    mp = model_path or CPT_MERGED_DIR

    if model is None or tokenizer is None:
        print(f"\nLoading CPT model for eval from {mp} ...")
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16
        ).cuda()
    model.eval()

    # ── Perplexity on val set ────────────────────────────────────────────────
    print("\n--- CPT Evaluation: Perplexity ---")
    try:
        dataset = load_from_disk(CACHE_CPT_TOKENIZED)
        val = dataset["test"].select(range(min(200, len(dataset["test"]))))
        total_loss  = 0.0
        total_tokens = 0
        with torch.no_grad():
            for sample in val:
                ids = torch.tensor(sample["input_ids"]).unsqueeze(0).cuda()
                ids = ids[:, :MAX_SEQ_LENGTH]
                out = model(input_ids=ids, labels=ids)
                seq_tokens  = ids.shape[-1]
                total_loss  += out.loss.item() * seq_tokens
                total_tokens += seq_tokens
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        print(f"  Val loss   : {avg_loss:.4f}")
        print(f"  Perplexity : {ppl:.2f}")
    except Exception as e:
        print(f"  ⚠ Perplexity eval failed: {e}")

    # ── Completion spot-checks ───────────────────────────────────────────────
    print("\n--- CPT Spot-Check Completions ---")
    model.eval()
    for prompt in CPT_SPOT_CHECK_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"\n  PROMPT : {prompt}")
        print(f"  OUTPUT : {completion.strip()}")


# ── SFT evaluation ────────────────────────────────────────────────────────────

SFT_SPOT_CHECK_QA = [
    ("You are a helpful medical assistant.", "What are the main symptoms of COVID-19?"),
    ("You are a helpful medical assistant.", "How do mRNA COVID-19 vaccines work?"),
    ("You are a helpful medical assistant.", "What is the difference between isolation and quarantine?"),
    ("You are a helpful medical assistant.", "What are the risk factors for severe COVID-19 disease?"),
    ("You are a helpful medical assistant.", "Can you explain what long COVID is?"),
    ("You are a helpful medical assistant.", "What treatments are available for COVID-19?"),
]


def run_sft_eval(model=None, tokenizer=None, model_path: str = None):
    """
    Evaluate the SFT model:
      1. Perplexity on SFT validation set
      2. Medical Q&A spot-checks using the Llama-3 chat template

    Only runs on rank-0.
    """
    if not _is_rank0():
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_from_disk
    from config import SFT_MERGED_DIR, SFT_CACHE_TOKENIZED

    mp = model_path or SFT_MERGED_DIR

    if model is None or tokenizer is None:
        print(f"\nLoading SFT model for eval from {mp} ...")
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16
        ).cuda()
    model.eval()

    # ── Perplexity on SFT val set ────────────────────────────────────────────
    print("\n--- SFT Evaluation: Perplexity ---")
    try:
        dataset = load_from_disk(SFT_CACHE_TOKENIZED)
        val = dataset["test"].select(range(min(200, len(dataset["test"]))))
        total_loss   = 0.0
        total_tokens = 0
        with torch.no_grad():
            for sample in val:
                ids = torch.tensor(sample["input_ids"]).unsqueeze(0).cuda()
                out = model(input_ids=ids, labels=ids)
                seq_tokens  = ids.shape[-1]
                total_loss  += out.loss.item() * seq_tokens
                total_tokens += seq_tokens
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        print(f"  Val loss   : {avg_loss:.4f}")
        print(f"  Perplexity : {ppl:.2f}")
    except Exception as e:
        print(f"  ⚠ Perplexity eval failed: {e}")

    # ── Medical Q&A spot-checks ──────────────────────────────────────────────
    print("\n--- SFT Spot-Check Medical Q&A ---")
    for system_prompt, question in SFT_SPOT_CHECK_QA:
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": question},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"\n  Q: {question}")
        print(f"  A: {answer.strip()}")
        print("-" * 60)
