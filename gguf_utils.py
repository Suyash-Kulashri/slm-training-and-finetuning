# =============================================================================
# gguf_utils.py — Convert HuggingFace models to GGUF and run GGUF inference
# No multi-GPU changes needed — post-training, single process only.
# =============================================================================

import os
import subprocess
from config import (
    LLAMA_CPP_SCRIPT,
    CPT_MERGED_DIR, CPT_GGUF_PATH,
    SFT_MERGED_DIR, SFT_GGUF_PATH,
)


def _convert_to_gguf(model_dir: str, gguf_path: str, quant_type: str = "f16"):
    """
    Convert a HuggingFace model directory to GGUF using llama.cpp.

    Prerequisites:
      git clone https://github.com/ggerganov/llama.cpp
      cd llama.cpp && pip install -r requirements.txt

    config.py → LLAMA_CPP_SCRIPT = "llama.cpp/convert_hf_to_gguf.py"
    """
    if not os.path.isfile(LLAMA_CPP_SCRIPT):
        raise FileNotFoundError(
            f"llama.cpp convert script not found at: {LLAMA_CPP_SCRIPT}\n"
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Converting {model_dir} → {gguf_path} ({quant_type}) ...")
    cmd = [
        "python3", LLAMA_CPP_SCRIPT,
        model_dir,
        "--outfile", gguf_path,
        "--outtype", quant_type,
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed (exit {result.returncode})")

    size_mb = os.path.getsize(gguf_path) / 1e6
    print(f"✓ GGUF saved → {gguf_path}  ({size_mb:.0f} MB)")


def convert_cpt_to_gguf(quant_type: str = "f16"):
    """Convert CPT merged model to GGUF."""
    _convert_to_gguf(CPT_MERGED_DIR, CPT_GGUF_PATH, quant_type)


def convert_sft_to_gguf(quant_type: str = "f16"):
    """Convert SFT merged model to GGUF."""
    _convert_to_gguf(SFT_MERGED_DIR, SFT_GGUF_PATH, quant_type)


# ── GGUF inference ────────────────────────────────────────────────────────────

def _check_llama_cpp_python():
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed.\n"
            "Install: pip install llama-cpp-python\n"
            "For GPU acceleration: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
        )


def run_cpt_gguf_inference(
    gguf_path: str = None,
    prompt: str = "COVID-19 is caused by",
    max_tokens: int = 200,
    n_gpu_layers: int = -1,   # -1 = offload all layers to GPU
):
    """
    Run CPT model inference from a GGUF file using llama-cpp-python.

    n_gpu_layers=-1 uses all available GPU layers.
    Set to 0 for CPU-only inference.
    """
    Llama = _check_llama_cpp_python()
    gp = gguf_path or CPT_GGUF_PATH
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"GGUF file not found: {gp}")

    print(f"Loading CPT GGUF from {gp} ...")
    llm = Llama(model_path=gp, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=False)
    print(f"\nPrompt: {prompt}\n")
    output = llm(prompt, max_tokens=max_tokens, stop=["<|eot_id|>"], echo=False)
    response = output["choices"][0]["text"]
    print(f"Response: {response}")
    return response


def run_sft_gguf_inference(
    gguf_path: str = None,
    question: str = "What are the main symptoms of COVID-19?",
    system_prompt: str = "You are a helpful COVID-19 medical assistant.",
    max_tokens: int = 300,
    n_gpu_layers: int = -1,
):
    """
    Run SFT model chat inference from a GGUF file using llama-cpp-python.
    Uses the Llama-3 chat template format directly in the prompt string.
    """
    Llama = _check_llama_cpp_python()
    gp = gguf_path or SFT_GGUF_PATH
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"GGUF file not found: {gp}")

    print(f"Loading SFT GGUF from {gp} ...")
    llm = Llama(model_path=gp, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=False)

    # Llama-3 chat template format
    prompt = (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    print(f"\nQ: {question}")
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        echo=False,
    )
    response = output["choices"][0]["text"]
    print(f"A: {response.strip()}")
    return response
