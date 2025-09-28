#!/bin/bash
#SBATCH --job-name=embed-cluster-concepts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=embed_cluster_%j.out
#SBATCH --error=embed_cluster_%j.err

set -euo pipefail

echo "== $(date) : starting on $(hostname -f)"

###############
# PARAMETERS  #
###############
# Paths to your two CSVs (each must have columns: id, merged)
DATASETS_CSV="${DATASETS_CSV:-./datasets_merged.csv}"
PUBLICATIONS_CSV="${PUBLICATIONS_CSV:-./publications_merged.csv}"

# Output directory for artifacts
OUTDIR="${OUTDIR:-$PWD/out_emb_cluster}"
mkdir -p "$OUTDIR"

# Embedding model to SERVE (OpenAI-compatible with /v1/embeddings)
# Default: multilingual, strong clustering performance
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"

# Serve locally with vLLM? If 0, we will use REMOTE_BASE_URL instead.
SERVE_LOCALLY="${SERVE_LOCALLY:-1}"
PORT="${PORT:-8000}"  # change if needed

# If using a remote server, set REMOTE_BASE_URL (e.g., http://gpu006.cluster:8000)
REMOTE_BASE_URL="${REMOTE_BASE_URL:-}"

# Conda environment (edit to match your cluster setup)
CONDA_ENV="${CONDA_ENV:-vllm-embed}"

# Clustering params
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-20}"

echo "== Config =="
echo "DATASETS_CSV=$DATASETS_CSV"
echo "PUBLICATIONS_CSV=$PUBLICATIONS_CSV"
echo "EMBED_MODEL=$EMBED_MODEL"
echo "SERVE_LOCALLY=$SERVE_LOCALLY"
echo "PORT=$PORT"
echo "REMOTE_BASE_URL=$REMOTE_BASE_URL"
echo "CONDA_ENV=$CONDA_ENV"
echo "OUTDIR=$OUTDIR"
echo "MIN_CLUSTER_SIZE=$MIN_CLUSTER_SIZE"

########################
# ENV / MODULES / CONDA
########################
if command -v module &>/dev/null; then
  module purge || true
  # Edit these to match your cluster
  module load anaconda3 || true
  module load cuda || true
fi

if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

#################################
# (Optional) SERVE vLLM LOCALLY #
#################################
API_BASE=""
SERVER_PID=""

if [[ "$SERVE_LOCALLY" == "1" ]]; then
  echo "== Launching vLLM OpenAI server for embeddings on port $PORT with model $EMBED_MODEL"
  # --task embedding is required for embedding models
  # If you need quantization or tensor-parallel, add flags here.
  python -m vllm.entrypoints.openai.api_server     --host 0.0.0.0     --port "$PORT"     --model "$EMBED_MODEL"     --task embedding     --dtype auto     --max-model-len 8192     > "$OUTDIR/vllm_${SLURM_JOB_ID:-manual}.log" 2>&1 &
  SERVER_PID=$!
  API_BASE="http://127.0.0.1:${PORT}"

  echo "== Waiting for server to become ready at $API_BASE ..."
  for i in $(seq 1 120); do
    if curl -s "${API_BASE}/v1/models" | grep -q '"id"'; then
      echo "✅ vLLM embeddings server is up."
      break
    fi
    sleep 3
  done
  if ! curl -s "${API_BASE}/v1/models" | grep -q '"id"'; then
    echo "❌ vLLM server did not come up in time. See $OUTDIR/vllm_${SLURM_JOB_ID:-manual}.log"
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" || true
    exit 1
  fi
else
  if [[ -z "$REMOTE_BASE_URL" ]]; then
    echo "❌ SERVE_LOCALLY=0 but REMOTE_BASE_URL is empty."
    exit 1
  fi
  API_BASE="$REMOTE_BASE_URL"
  echo "== Using remote embeddings API at $API_BASE"
  # Quick connectivity check
  curl -s "$API_BASE/v1/models" >/dev/null || { echo "❌ Can't reach $API_BASE"; exit 1; }
  echo "✅ Remote models endpoint reachable"
fi

#########################
# PIPELINE: concat → embed → cluster+label
#########################
COMBINED="$OUTDIR/combined_inputs.csv"
EMBED_NPZ="$OUTDIR/embeddings.npz"
CLUSTERS_CSV="$OUTDIR/clusters.csv"
TOPICS_CSV="$OUTDIR/topics.csv"

echo "== Step 1/3: Concatenate inputs"
python concat_inputs.py   --inputs "$DATASETS_CSV" "$PUBLICATIONS_CSV"   --id-col id   --text-col merged   --out "$COMBINED"

echo "== Step 2/3: Embed via OpenAI-compatible /v1/embeddings"
python embed_client.py   --api-base "$API_BASE"   --model "$EMBED_MODEL"   --input-csv "$COMBINED"   --id-col id   --text-col text   --batch-size 64   --out "$EMBED_NPZ"

echo "== Step 3/3: Cluster and label"
python cluster_and_label.py   --input-csv "$COMBINED"   --embeddings "$EMBED_NPZ"   --min-cluster-size "$MIN_CLUSTER_SIZE"   --clusters-out "$CLUSTERS_CSV"   --topics-out "$TOPICS_CSV"

echo "== Artifacts =="
echo "COMBINED: $COMBINED"
echo "EMBEDDINGS: $EMBED_NPZ"
echo "CLUSTERS: $CLUSTERS_CSV"
echo "TOPICS: $TOPICS_CSV"

# Cleanup server
if [[ -n "$SERVER_PID" ]]; then
  echo "== Stopping vLLM server (PID=$SERVER_PID)"
  kill "$SERVER_PID" || true
  sleep 2
fi

echo "== $(date) : done"
