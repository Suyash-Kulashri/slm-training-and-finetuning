#!/usr/bin/env python3
# =============================================================================
# run_inference.py — CLI entry point for HF and GGUF inference
# No multi-GPU changes — always runs as a single process.
#
# USAGE:
#   # HF inference (default, streaming):
#   python run_inference.py
#
#   # HF inference with a specific question:
#   python run_inference.py --question "What are COVID-19 treatment options?"
#
#   # HF inference with a specific model path:
#   python run_inference.py --model-path llama3.1-8b-covid-sft-merged-bf16
#
#   # GGUF inference (requires llama-cpp-python):
#   python run_inference.py --gguf
#   python run_inference.py --gguf --model-path llama3.1-8b-covid-sft-f16.gguf
#
#   # Non-streaming (prints all at once):
#   python run_inference.py --no-stream
# =============================================================================

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="COVID SLM Inference — HuggingFace or GGUF"
    )
    parser.add_argument(
        "--gguf", action="store_true",
        help="Use GGUF model for inference (requires llama-cpp-python).",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to HF model directory or GGUF file. "
             "Defaults to SFT_MERGED_DIR (HF) or SFT_GGUF_PATH (GGUF) from config.py.",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Question to ask the model. Defaults to a COVID-19 sample question.",
    )
    parser.add_argument(
        "--system", type=str,
        default="You are a helpful COVID-19 medical assistant.",
        help="System prompt for chat inference.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="Maximum new tokens to generate (default: 300).",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming output (print full response at once).",
    )
    parser.add_argument(
        "--cpt", action="store_true",
        help="Use CPT model instead of SFT model for completion inference.",
    )
    parser.add_argument(
        "--convert-gguf", choices=["cpt", "sft", "both"], default=None,
        help="Convert HF model(s) to GGUF before inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Optional: convert to GGUF first ──────────────────────────────────────
    if args.convert_gguf:
        from gguf_utils import convert_cpt_to_gguf, convert_sft_to_gguf
        if args.convert_gguf in ("cpt", "both"):
            convert_cpt_to_gguf()
        if args.convert_gguf in ("sft", "both"):
            convert_sft_to_gguf()
        if not args.gguf:
            return  # just convert, don't run inference

    # ── GGUF inference ────────────────────────────────────────────────────────
    if args.gguf:
        from gguf_utils import run_sft_gguf_inference, run_cpt_gguf_inference
        if args.cpt:
            run_cpt_gguf_inference(
                gguf_path=args.model_path,
                prompt=args.question or "COVID-19 is caused by",
                max_tokens=args.max_tokens,
            )
        else:
            run_sft_gguf_inference(
                gguf_path=args.model_path,
                question=args.question or "What are the key symptoms of COVID-19?",
                system_prompt=args.system,
                max_tokens=args.max_tokens,
            )
        return

    # ── HF inference ──────────────────────────────────────────────────────────
    from inference import run_hf_inference
    run_hf_inference(
        model_path=args.model_path,
        question=args.question,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
