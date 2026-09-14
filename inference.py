# =============================================================================
# inference.py — Streaming and non-streaming HF inference (completion + chat)
# No multi-GPU changes needed — inference is always single-process.
# =============================================================================

import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from config import SFT_MERGED_DIR


def load_inference_model(model_path: str = None):
    """Load model and tokenizer for inference."""
    mp = model_path or SFT_MERGED_DIR
    print(f"Loading inference model from {mp} ...")
    tokenizer = AutoTokenizer.from_pretrained(mp)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mp,
        torch_dtype=torch.bfloat16,
<<<<<<< HEAD
        attn_implementation="flash_attention_2",
=======
        attn_implementation="sdpa",
>>>>>>> 07d00f6 (Final code of CPT and SFT)
        device_map="auto",   # OK for inference — not DDP
    )
    model.eval()
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


# ── Completion inference ───────────────────────────────────────────────────────

def complete(model, tokenizer, prompt: str,
             max_new_tokens: int = 200,
             temperature: float = 0.7,
             top_p: float = 0.9,
             repetition_penalty: float = 1.1,
             do_sample: bool = True) -> str:
    """Non-streaming text completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )


def complete_streaming(model, tokenizer, prompt: str,
                       max_new_tokens: int = 200,
                       temperature: float = 0.7,
                       top_p: float = 0.9):
    """Streaming text completion — yields tokens one at a time."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    t = Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()
    for token in streamer:
        yield token
    t.join()


# ── Chat inference ─────────────────────────────────────────────────────────────

def chat(model, tokenizer,
         question: str,
         system_prompt: str = "You are a helpful COVID-19 medical assistant.",
<<<<<<< HEAD
         max_new_tokens: int = 300,
=======
         max_new_tokens: int = 1024,
>>>>>>> 07d00f6 (Final code of CPT and SFT)
         temperature: float = 0.7,
         top_p: float = 0.9,
         stream: bool = False):
    """
    Chat-style inference using the Llama-3 chat template.
    If stream=True, prints tokens live and returns the full response string.
    If stream=False, returns the response string directly.
    """
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if stream:
        print(f"\nQ: {question}")
        print("A: ", end="", flush=True)
        full = ""
        for token in complete_streaming(model, tokenizer, prompt,
                                        max_new_tokens=max_new_tokens,
                                        temperature=temperature,
                                        top_p=top_p):
            print(token, end="", flush=True)
            full += token
        print()
        return full
    else:
        return complete(model, tokenizer, prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p)


def run_hf_inference(model_path: str = None, question: str = None, stream: bool = True):
    """
    Load the final SFT model and run a sample Q&A.
    Called by run_inference.py and train_model_QAT16_final.py --inference.
    """
    model, tokenizer = load_inference_model(model_path)

    q = question or "What are the key differences between COVID-19 variants?"
    print(f"\n{'='*60}")
    print(f"HF Inference — {model_path or SFT_MERGED_DIR}")
    print(f"{'='*60}")
    answer = chat(model, tokenizer, question=q, stream=stream)
    if not stream:
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    return answer
