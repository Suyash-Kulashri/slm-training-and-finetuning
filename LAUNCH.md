# Launch Instructions — 4x A100-SXM4-40GB Multi-GPU Training

## One-Time Setup

```bash
# 1. Set environment variables permanently (do this ONCE, survives reboots)
echo 'export PYTORCH_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
source ~/.bashrc

# 2. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Flash Attention — try prebuilt wheel first (no nvcc needed)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# If above 404s, use sdpa fallback — change attn_implementation="sdpa" in model_loader.py and inference.py

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Log in to HuggingFace (required once per machine for Llama-3.1 access)
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens (Read permission)
# Accept Llama-3.1 license at: https://huggingface.co/meta-llama/Llama-3.1-8B
```

---

## Every Session — Set These Before Training

Run these at the start of every new terminal session (if not added to ~/.bashrc):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd ~/slm_training_no_unsloth
source .venv/bin/activate
```

---

## Running the Pipeline

### Step 1 — Data Preprocessing (CPU only, run once)

No torchrun needed. Uses all 40 CPU cores. No GPU required.

```bash
python train_model_QAT16_final.py --data-only
```

This produces all caches:
- `_covid_cache_01_chunks_raw.jsonl`
- `_covid_cache_02_chunks_final.jsonl`
- `covid19_data_chunked_deduped.jsonl`
- `_covid_tokenized_cache/`

---

### Step 2 — CPT Training (4 GPUs, runs in background)

```bash
# Set env vars just before launching
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Start a named screen session so training survives laptop close / internet drop
screen -S cpt_training

# Inside screen — launch CPT training with logging
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --cpt-only --skip-data 2>&1 | tee cpt_training.log

# Detach from screen WITHOUT stopping training:
# Press:  Ctrl+A  then  D
# You can now safely close your laptop or disconnect.
```

**Reattach to check progress anytime:**
```bash
screen -r cpt_training          # reattach to live output
tail -f cpt_training.log        # or just watch the log
tail -50 cpt_training.log       # see last 50 lines
watch -n 10 nvidia-smi          # monitor GPU utilization
```

**If training crashes — resume from checkpoint automatically:**
```bash
screen -S cpt_training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --cpt-only --skip-data 2>&1 | tee -a cpt_training.log
# -a appends to existing log instead of overwriting
```

Outputs:
- `llama3.1-8b-covid-cpt-checkpoints/` — rolling checkpoints (every 500 steps)
- `llama3.1-8b-covid-cpt-bf16/` — final merged model

**If merge failed after training (directory missing), run manually:**
```bash
python merge_cpt.py
```

---

### Step 3 — SFT Training (4 GPUs, runs in background)

```bash
# Set env vars just before launching
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Start a new screen session for SFT
screen -S sft_training

# Inside screen — launch SFT training with logging
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --sft-only 2>&1 | tee sft_training.log

# Detach from screen WITHOUT stopping training:
# Press:  Ctrl+A  then  D
```

**Reattach to check progress:**
```bash
screen -r sft_training
tail -f sft_training.log
```

**If training crashes — resume:**
```bash
screen -S sft_training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_model_QAT16_final.py --sft-only 2>&1 | tee -a sft_training.log
```

Outputs:
- `llama3.1-8b-covid-sft-checkpoints/` — rolling checkpoints (every 200 steps)
- `llama3.1-8b-covid-sft-lora/` — LoRA adapter only (~150 MB)
- `llama3.1-8b-covid-sft-merged-bf16/` — final merged model (~16 GB)

---

### Step 4 — Fix Tokenizer (Required Before GGUF Conversion)

The tokenizer must be copied from the base model before converting to GGUF.
Run this once after SFT completes:

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tok.save_pretrained('llama3.1-8b-covid-sft-merged-bf16')
print('Tokenizer fixed.')
"
```

---

### Step 5 — Convert to GGUF F16

```bash
# Convert merged SFT model to GGUF F16 (single file, full precision)
python3 llama.cpp/convert_hf_to_gguf.py llama3.1-8b-covid-sft-merged-bf16 --outfile llama3.1-8b-covid-sft-f16.gguf --outtype f16

# Verify output
ls -lah llama3.1-8b-covid-sft-f16.gguf
# Expected: ~16 GB
```

**Optional — also convert CPT model to GGUF:**
```bash
# Fix CPT tokenizer first
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tok.save_pretrained('llama3.1-8b-covid-cpt-bf16')
print('CPT tokenizer fixed.')
"

# Convert CPT to GGUF
python3 llama.cpp/convert_hf_to_gguf.py llama3.1-8b-covid-cpt-bf16 --outfile llama3.1-8b-covid-cpt-f16.gguf --outtype f16
```

---

### Step 6 — Test GGUF Inference

```bash
# Build llama.cpp CLI (one time)
cd ~/llama.cpp && make llama-cli -j4
cd ~/slm_training_no_unsloth

# Run inference on the GGUF model
~/llama.cpp/llama-cli \
    -m llama3.1-8b-covid-sft-f16.gguf \
    -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful COVID-19 medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat are the symptoms of COVID-19?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
    -n 512 \
    --temp 0.7 \
    --repeat-penalty 1.15 \
    --no-penalize-nl \
    -ngl 99
# -ngl 99 offloads all layers to GPU for fast inference
```

---

### Inference via Python (HuggingFace)

```bash
python train_model_QAT16_final.py --inference
# or
python run_inference.py
python run_inference.py --question "What are COVID-19 symptoms?"
python run_inference.py --gguf --model-path llama3.1-8b-covid-sft-f16.gguf
```

---

## Screen Session Management

```bash
screen -ls                    # list all active screen sessions
screen -r cpt_training        # reattach to CPT session
screen -r sft_training        # reattach to SFT session
screen -X -S cpt_training quit  # kill a session by name
```

**Inside any screen session:**
- `Ctrl+A then D` — detach (training keeps running)
- `Ctrl+A then [` — scroll up through output
- `Ctrl+C` — stop training (safe — checkpoint already saved)

---

## Expected Training Times (4x A100-SXM4-40GB)

| Stage | Steps | Est. Time |
|---|---|---|
| Data pipeline | — | ~30–60 min |
| CPT training | ~4,500 | ~15–20 hours |
| SFT training | ~3,300 | ~3–5 hours |
| GGUF conversion | — | ~10–15 min |

Compare to single A100-80GB: ~50–80 hours CPT → **~4x speedup**

---

## Effective Batch Sizes

| Stage | Per-GPU | GPUs | Grad Accum | Effective |
|---|---|---|---|---|
| CPT | 2 | 4 | 8 | **64** |
| SFT | 2 | 4 | 8 | **64** |

---

## Final Output Files

```
llama3.1-8b-covid-cpt-bf16/          ← CPT merged model (HF format, ~16 GB)
llama3.1-8b-covid-sft-lora/          ← SFT LoRA adapter only (~150 MB)
llama3.1-8b-covid-sft-merged-bf16/   ← SFT merged model (HF format, ~16 GB)
llama3.1-8b-covid-sft-f16.gguf       ← Final deployment model (single file, ~16 GB)
```

---

## Troubleshooting

**`NCCL timeout` or hanging at first step:**
- Check `ddp_find_unused_parameters=False` is set in TrainingArguments
- Ensure `use_reentrant=False` in gradient_checkpointing_kwargs

**`CUDA out of memory` during eval:**
- Ensure `per_device_eval_batch_size=1` is set in cpt_trainer.py
- Reduce `CPT_PER_DEVICE_BATCH` from 2 to 1 in config.py
- Increase `CPT_GRAD_ACCUM` from 8 to 16 to maintain effective batch of 64

**`ImportError: cannot import name get_chat_template`:**
- Remove the unsloth import from sft_trainer.py
- Use the hardcoded Llama-3 chat template in apply_chat_template()

**`ValueError: Tokenizer class TokenizersBackend does not exist`:**
- Run the tokenizer fix command in Step 4 above before GGUF conversion

**`ModuleNotFoundError: flash_attn`:**
- Change `attn_implementation="sdpa"` in model_loader.py and inference.py
- sdpa is built into PyTorch 2.0+ — no install needed

**Port already in use:**
- Change `--master_port=29500` to any free port (e.g. 29501)

**Only 1 GPU being used:**
- Make sure you launched with `torchrun`, not `python`
- Check `CUDA_VISIBLE_DEVICES` is not set to a single GPU

**Screen session lost / can't reattach:**
- Check `screen -ls` — session may have a different name
- Check `tail -f cpt_training.log` — log file persists even if screen dies
- Training checkpoints are safe regardless — just relaunch to resume
