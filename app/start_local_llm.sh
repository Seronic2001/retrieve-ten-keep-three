#!/usr/bin/env bash
# Starts the local LLM for the chat's "Qwen 4B (local)" option. Run inside WSL:
#   bash start_local_llm.sh                                   # Qwen3.5-4B Q4_K_M, all on GPU, no thinking (default)
#   MODEL_NAME=Qwen3-4B-Instruct-2507 bash start_local_llm.sh # Qwen3-4B-Instruct-2507 Q4_K_M, all on GPU
#   (Qwen3 files are not kept locally; get them from huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
#   MODEL_NAME=Qwen3-4B-Instruct-2507 QUANT=Q5_K_M KV_TYPE=f16 GPU_LAYERS=auto bash start_local_llm.sh  # Qwen3 Q5, split
# llama-server exposes an OpenAI-compatible API on port 8080; WSL2 forwards it to Windows localhost.
#
# Measured on a GTX 1050 (3 GB), RAG prompt of ~1,000 tokens:
#   Qwen3.5 Q4_K_M, all 32 layers on GPU:   2,855 MiB VRAM, ~16 tok/s output; on 7 test questions:
#                                           key fact 6/6, correct article cited 5/6
#   Qwen3 Q4_K_M, same run conditions:      2,798 MiB VRAM, ~15.5 tok/s output; key fact 6/6, article 3/6
#   Qwen3 Q4_K_M, all 36 layers on GPU, q8_0 KV: 2,705 MiB VRAM, ~295 tok/s prompt, ~18 tok/s output
#   Qwen3 Q5_K_M, --fit split GPU/CPU,   f16 KV: 2,345 MiB VRAM, ~170-250 tok/s prompt, ~6.7 tok/s output
# VRAM notes:
# - --fit (on by default) picks how many layers go to the GPU; the rest run on the CPU.
#   It keeps FIT_MARGIN_MIB free (stock default 1024, a third of this card; 256 here).
# - CTX 2048 is enough: the RAG prompt is 3 chunks (~1,050 tokens) plus the answer.
# - KV_TYPE=q8_0 halves the KV cache (288 -> 144 MiB at 2048 tokens).
# - CACHE_RAM caps llama-server's prompt cache in system RAM. Its default (8192 MiB) is more than WSL gets on
#   this machine (6.9 GB), so over a long evaluation it grew to 6.5 GB and the OOM killer stopped the server.
#   RAG prompts rarely repeat, so a small cache loses nothing.
# - Start the FastAPI backend with RAG_DEVICE=cpu so both encoders stay on the CPU. Sharing the GPU
#   (RAG_RERANK_DEVICE=cuda) also fits, with only ~110 MiB to spare; see rag_core.py.
# Thinking: Qwen3.5 thinks before answering by default, which at ~15 tok/s adds tens of seconds per
# answer, so it is switched off (THINKING=1 turns it back on). Qwen3-Instruct-2507 has no thinking mode.
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama-bin/llama.cpp}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
QUANT="${QUANT:-Q4_K_M}"
MODEL="${MODEL:-$HOME/models/$MODEL_NAME-$QUANT.gguf}"
GPU_LAYERS="${GPU_LAYERS:-all}"   # all fits only with the backend set up as above
CTX="${CTX:-2048}"
KV_TYPE="${KV_TYPE:-q8_0}"
CACHE_RAM="${CACHE_RAM:-512}"
if [ "${THINKING:-0}" = "1" ]; then THINK_ARGS=(); else
  THINK_ARGS=(--chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0); fi

cd "$LLAMA_DIR"
exec ./llama-server -m "$MODEL" --host 0.0.0.0 --port 8080 \
  -c "$CTX" -np 1 -ngl "$GPU_LAYERS" --fit-target "${FIT_MARGIN_MIB:-256}" \
  -ctk "$KV_TYPE" -ctv "$KV_TYPE" --cache-ram "$CACHE_RAM" --flash-attn on --jinja "${THINK_ARGS[@]}"
