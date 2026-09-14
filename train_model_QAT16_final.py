#!/usr/bin/env python3
# =============================================================================
# train_model_QAT16_final.py — Main entry point for COVID SLM training pipeline
#
# CHANGES for torchrun multi-GPU:
#   - if __name__ == "__main__" guard (required for torchrun)
#   - --data-only runs on CPU without torchrun
#   - --cpt-only and --sft-only run under torchrun
#   - Updated help text with correct launch commands
#
# USAGE:
#   # Data preprocessing only (CPU, no torchrun needed):
#   python train_model_QAT16_final.py --data-only
#
#   # Full pipeline (4 GPUs):
#   torchrun --nproc_per_node=4 --master_port=29500 train_model_QAT16_final.py
#
#   # CPT only (assumes data caches exist):
#   torchrun --nproc_per_node=4 --master_port=29500 \
#       train_model_QAT16_final.py --cpt-only --skip-data
#
#   # SFT only (assumes CPT model exists):
#   torchrun --nproc_per_node=4 --master_port=29500 \
#       train_model_QAT16_final.py --sft-only
#
#   # Inference (single process, no torchrun):
#   python train_model_QAT16_final.py --inference
# =============================================================================

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="COVID SLM Training Pipeline — 4x A100-SXM4-40GB + torchrun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Launch commands:
  Data only (CPU, no torchrun):
    python train_model_QAT16_final.py --data-only

  Full pipeline (4 GPUs):
    torchrun --nproc_per_node=4 --master_port=29500 train_model_QAT16_final.py

  CPT only (data caches must exist):
    torchrun --nproc_per_node=4 --master_port=29500 \\
        train_model_QAT16_final.py --cpt-only --skip-data

  SFT only (CPT model must exist):
    torchrun --nproc_per_node=4 --master_port=29500 \\
        train_model_QAT16_final.py --sft-only

  Inference (single process):
    python train_model_QAT16_final.py --inference
        """,
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Run data pipeline only (chunking, filtering, tokenization). "
             "No GPU required. Uses all 40 CPU cores. Do NOT use torchrun for this.",
    )
    parser.add_argument(
        "--cpt-only", action="store_true",
        help="Run data pipeline (unless --skip-data) + CPT training only.",
    )
    parser.add_argument(
        "--sft-only", action="store_true",
        help="Run SFT training only. Requires CPT merged model to exist.",
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data pipeline — assume all caches already exist.",
    )
    parser.add_argument(
        "--inference", action="store_true",
        help="Run inference on the final SFT model. Single process, no torchrun.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation (perplexity + spot-checks) after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Data only — no GPU, no torchrun ──────────────────────────────────────
    if args.data_only:
        print("=" * 60)
        print("DATA PIPELINE ONLY — CPU mode, 40 cores")
        print("=" * 60)
        from run_cpt import run_data_only
        run_data_only()
        return

    # ── Inference — single process, no torchrun ───────────────────────────────
    if args.inference:
        print("=" * 60)
        print("INFERENCE MODE")
        print("=" * 60)
        from run_inference import run_hf_inference
        run_hf_inference()
        return

    # ── Training modes — require torchrun when using multiple GPUs ────────────
    # Warn if LOCAL_RANK is not set (likely forgot torchrun)
    if "LOCAL_RANK" not in os.environ:
        print("\n⚠️  WARNING: LOCAL_RANK not set. Running in single-GPU mode.")
        print("   For multi-GPU training, launch with:")
        print("   torchrun --nproc_per_node=4 --master_port=29500 train_model_QAT16_final.py\n")

    if args.sft_only:
        print("=" * 60)
        print("SFT ONLY")
        print("=" * 60)
        from run_sft import run_sft_pipeline
        run_sft_pipeline()
        return

    if args.cpt_only:
        print("=" * 60)
        print("CPT ONLY")
        print("=" * 60)
        from run_cpt import run_cpt_pipeline
        run_cpt_pipeline(skip_data=args.skip_data)
        # Optionally run eval
        if not args.no_eval:
            from eval_utils import run_cpt_eval
            run_cpt_eval()
        return

    # ── Full pipeline: CPT → SFT ──────────────────────────────────────────────
    print("=" * 60)
    print("FULL PIPELINE: Data → CPT → SFT")
    print("=" * 60)

    from run_cpt import run_cpt_pipeline
    run_cpt_pipeline(skip_data=args.skip_data)

    if not args.no_eval:
        from eval_utils import run_cpt_eval
        run_cpt_eval()

    from run_sft import run_sft_pipeline
    run_sft_pipeline()

    if not args.no_eval:
        from eval_utils import run_sft_eval
        run_sft_eval()

    print("\n🎉 Full pipeline complete.")


# torchrun entry point guard — REQUIRED for torchrun to work correctly.
# torchrun spawns N processes, each running this file.
# Without this guard, the argparse + setup code runs inside each worker
# process during module import, causing errors.
if __name__ == "__main__":
    main()
