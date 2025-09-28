#!/bin/bash
#SBATCH --job-name=be-llm-server
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=be-llm-server.out
#SBATCH --error=be-llm-server.err

set -euo pipefail
echo "== $(date) : starting on $(hostname -f)"
echo "SLURM_NODELIST=$SLURM_NODELIST ; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- Config ----
: "${MODEL:=mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
: "${PORT:=8000}"
: "${OUT_DIR:=$PWD/runs}"
: "${MAX_MODEL_LEN:=8192}"

# Ensure OUT_DIR is writable; fallback if needed
if ! mkdir -p "$OUT_DIR" 2>/dev/null; then
  for cand in "$PWD/runs" "$HOME/runs" "${SLURM_TMPDIR:-${TMPDIR:-}}/runs"; do
    if [[ -n "$cand" ]] && mkdir -p "$cand" 2>/dev/null; then
      OUT_DIR="$cand"
      break
    fi
  done
fi
mkdir -p "$OUT_DIR"
echo "== OUT_DIR=$OUT_DIR"

# ---- Modules & scratch venv ----
module purge
module load anaconda3/2024.10 || true
module load nvidia/cuda/12.1.0 || true

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
SCRATCH_DIR="$(mktemp -d -p "$SCRATCH_BASE" "srv_${SLURM_JOB_ID:-$$}_XXXX")"
trap 'echo "== cleanup scratch $SCRATCH_DIR"; rm -rf "$SCRATCH_DIR"' EXIT

python -m venv "$SCRATCH_DIR/venv"
source "$SCRATCH_DIR/venv/bin/activate"
python -V
python -m pip install -U pip setuptools wheel
# Torch for CUDA 12.1 (vLLM compat)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
python -m pip install "vllm>=0.10.1" "transformers>=4.46.0" "tokenizers>=0.20.0" safetensors

export HF_HOME="${HF_HOME:-$OUT_DIR/hf_cache}"
mkdir -p "$HF_HOME"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Node & URLs ----
if command -v scontrol >/dev/null 2>&1; then
  NODE_FQDN="$(scontrol show hostname "$SLURM_NODELIST" | head -n1)"
else
  NODE_FQDN="$(hostname -f)"
fi
API_BASE="http://${NODE_FQDN}:${PORT}/v1"
LOCAL_BASE="http://127.0.0.1:${PORT}"
LOG="${OUT_DIR}/vllm_server.log"

# Determine TP from visible GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
  TP=${#DEVS[@]}
else
  TP=2
fi
echo "== Starting vLLM on $NODE_FQDN : port $PORT : TP=$TP"
echo "== Model: $MODEL"
echo "== Log:   $LOG"

# ---- Start server ----
vllm serve "$MODEL" \
  --download-dir "$HF_HOME" \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --tensor-parallel-size "${TP}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --disable-custom-all-reduce \
  >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"

# ---- Readiness (HTTP + log progress) ----
WAIT_MAX_SEC="${WAIT_MAX_SEC:-3600}"
echo "== Waiting for readiness (up to ${WAIT_MAX_SEC}s) at $LOCAL_BASE"
start_ts=$(date +%s)
last_size=0
while true; do
  now=$(date +%s); elapsed=$((now - start_ts))
  if (( elapsed > WAIT_MAX_SEC )); then
    echo "!! Timeout after ${WAIT_MAX_SEC}s"; tail -n 200 "$LOG" || true; exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! Server exited during wait"; tail -n 300 "$LOG" || true; exit 1
  fi
  if curl -fsS "${LOCAL_BASE}/health" >/dev/null 2>&1 || curl -fsS "${LOCAL_BASE}/v1/models" >/dev/null 2>&1; then
    echo "== Ready after ${elapsed}s"
    break
  fi
  size=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  if [[ "$size" -gt "$last_size" ]]; then last_size="$size"; echo "… still loading (log ${size} bytes, ${elapsed}s)"; else echo "… waiting (${elapsed}s)"; fi
  sleep 3
done

# ---- Publish URL for clients ----
#echo "$API_BASE" | tee "$OUT_DIR/api_base_url.txt"
#echo "Health: $(curl -s -o /dev/null -w '%{http_code}' "$API_BASE/health")"
#echo "Models: $(curl -s -o /dev/null -w '%{http_code}' "$API_BASE/models")"
# ---- Publish URL for clients ----
HOST_FQDN="$(hostname -f)"                    # important: -f (FQDN), NOT -s
API_BASE="http://${HOST_FQDN}:8000/v1"

echo "$API_BASE" | tee "$OUT_DIR/api_base_url.txt"

# sanity probes (should both be 200)
echo "Health: $(curl -s -o /dev/null -w '%{http_code}' "$API_BASE/health")"
echo "Models: $(curl -s -o /dev/null -w '%{http_code}' "$API_BASE/models")"


# ---- Keep server alive for the job lifetime ----
echo "== Server running. Tail the log below; cancel job to stop."
tail -f "$LOG"
