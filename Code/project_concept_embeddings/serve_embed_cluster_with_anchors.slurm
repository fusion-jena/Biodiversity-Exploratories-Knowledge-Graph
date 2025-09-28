#!/bin/bash
#SBATCH --job-name=embed-cluster-anchors
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

# ---------- Paths ----------
DATASETS_CSV="${DATASETS_CSV:-./datasets_merged.csv}"
PUBLICATIONS_CSV="${PUBLICATIONS_CSV:-./publications_merged.csv}"
ANCHORS_CSV="${ANCHORS_CSV:-./NASA-EarthData-GCMD-Keywords.csv}"

# Columns in your anchors CSV (adjust if needed)
ANCHORS_ID_COL="${ANCHORS_ID_COL:-keyword}"    # e.g., "keyword" or "label"
ANCHORS_TEXT_COL="${ANCHORS_TEXT_COL:-merged}" # you've said "merged" exists
ANCHORS_LABEL_COL="${ANCHORS_LABEL_COL:-keyword}"

OUTDIR="${OUTDIR:-$PWD/out_emb_cluster}"
mkdir -p "$OUTDIR"

# ---------- Models ----------
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
SERVE_LOCALLY="${SERVE_LOCALLY:-1}"
PORT="${PORT:-8000}"
REMOTE_BASE_URL="${REMOTE_BASE_URL:-}"

# Optional generator for refine_labels (unchanged)
GEN_API_BASE="${GEN_API_BASE:-}"
GEN_MODEL="${GEN_MODEL:-}"

# ---------- Env ----------
CONDA_ENV="${CONDA_ENV:-vllm-embed}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-20}"

# Anchor assignment params
ANCHOR_THRESHOLD="${ANCHOR_THRESHOLD:-0.25}"
ANCHOR_TOPK="${ANCHOR_TOPK:-1}"

echo "== Config =="
echo "DATASETS_CSV=$DATASETS_CSV"
echo "PUBLICATIONS_CSV=$PUBLICATIONS_CSV"
echo "ANCHORS_CSV=$ANCHORS_CSV"
echo "ANCHORS_ID_COL=$ANCHORS_ID_COL  ANCHORS_TEXT_COL=$ANCHORS_TEXT_COL  ANCHORS_LABEL_COL=$ANCHORS_LABEL_COL"
echo "EMBED_MODEL=$EMBED_MODEL  SERVE_LOCALLY=$SERVE_LOCALLY  PORT=$PORT  REMOTE_BASE_URL=$REMOTE_BASE_URL"
echo "CONDA_ENV=$CONDA_ENV  MIN_CLUSTER_SIZE=$MIN_CLUSTER_SIZE"
echo "ANCHOR_THRESHOLD=$ANCHOR_THRESHOLD  ANCHOR_TOPK=$ANCHOR_TOPK"
echo "GEN_API_BASE=$GEN_API_BASE  GEN_MODEL=$GEN_MODEL"

# ---------- Modules / conda ----------
if command -v module &>/dev/null; then module purge || true; module load anaconda3 || true; module load cuda || true; fi
if command -v conda  &>/dev/null; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV" || true; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# ---------- Serve embeddings ----------
API_BASE=""; SERVER_PID=""
if [[ "$SERVE_LOCALLY" == "1" ]]; then
  echo "== Launch vLLM embeddings on port $PORT model $EMBED_MODEL"
  python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port "$PORT" --model "$EMBED_MODEL" --task embedding --dtype auto --max-model-len 8192 \
    > "$OUTDIR/vllm_${SLURM_JOB_ID:-manual}.log" 2>&1 &
  SERVER_PID=$!; API_BASE="http://127.0.0.1:${PORT}"
  for _ in $(seq 1 120); do curl -s "${API_BASE}/v1/models" | grep -q '"id"' && break; sleep 3; done
  curl -s "${API_BASE}/v1/models" | grep -q '"id"' || { echo "❌ vLLM failed to start"; kill "$SERVER_PID" || true; exit 1; }
else
  [[ -n "$REMOTE_BASE_URL" ]] || { echo "❌ Need REMOTE_BASE_URL"; exit 1; }
  API_BASE="$REMOTE_BASE_URL"; curl -s "$API_BASE/v1/models" >/dev/null || { echo "❌ Can't reach $API_BASE"; exit 1; }
fi

# ---------- Files ----------
COMBINED="$OUTDIR/combined_inputs.csv"
EMBED_NPZ="$OUTDIR/embeddings.npz"
CLUSTERS_CSV="$OUTDIR/clusters.csv"
TOPICS_CSV="$OUTDIR/topics.csv"
REFINED_CSV="$OUTDIR/topics_refined.csv"

ANCHOR_NPZ="$OUTDIR/anchor_embeddings.npz"
DOC_ANCHORS_CSV="$OUTDIR/anchor_assignments.csv"
ANCHORS_SUMMARY_CSV="$OUTDIR/anchors_summary.csv"
ANCHORS_TOPICS_CSV="$OUTDIR/anchors_topics.csv"

# ---------- Pipeline ----------
echo "== Step 1/6: Concat"
python concat_inputs.py --inputs "$DATASETS_CSV" "$PUBLICATIONS_CSV" --id-col id --text-col merged --out "$COMBINED"

echo "== Step 2/6: Embed docs"
python embed_client.py --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$COMBINED" --id-col id --text-col text --batch-size 64 --out "$EMBED_NPZ"

echo "== Step 3/6: Cluster + label"
python cluster_and_label.py --input-csv "$COMBINED" --embeddings "$EMBED_NPZ" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" --clusters-out "$CLUSTERS_CSV" --topics-out "$TOPICS_CSV"

echo "== Step 4/6: Refine labels (optional LLM polish)"
REFINE_ARGS=( --clusters-csv "$CLUSTERS_CSV" --topics-csv "$TOPICS_CSV" --out "$REFINED_CSV" --keyword --sentence )
if [[ -n "$GEN_API_BASE" && -n "$GEN_MODEL" ]]; then
  REFINE_ARGS+=( --llm-api-base "$GEN_API_BASE" --llm-model "$GEN_MODEL" )
fi
python refine_labels.py "${REFINE_ARGS[@]}"

echo "== Step 5/6: Embed anchors (NASA GCMD)"
python embed_client.py --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$ANCHORS_CSV" --id-col "$ANCHORS_ID_COL" --text-col "$ANCHORS_TEXT_COL" \
  --batch-size 128 --out "$ANCHOR_NPZ"

echo "== Step 6/6: Assign docs to anchors and rank"
python anchor_assign.py \
  --doc-embeddings "$EMBED_NPZ" \
  --doc-csv "$COMBINED" \
  --anchor-embeddings "$ANCHOR_NPZ" \
  --anchor-csv "$ANCHORS_CSV" \
  --anchor-id-col "$ANCHORS_ID_COL" \
  --anchor-label-col "$ANCHORS_LABEL_COL" \
  --anchor-text-col "$ANCHORS_TEXT_COL" \
  --topk "$ANCHOR_TOPK" \
  --threshold "$ANCHOR_THRESHOLD" \
  --assignments-out "$DOC_ANCHORS_CSV" \
  --summary-out "$ANCHORS_SUMMARY_CSV" \
  --topics-out "$ANCHORS_TOPICS_CSV"

echo "== Artifacts =="
echo "COMBINED:            $COMBINED"
echo "EMBEDDINGS:          $EMBED_NPZ"
echo "CLUSTERS:            $CLUSTERS_CSV"
echo "TOPICS:              $TOPICS_CSV"
echo "REFINED:             $REFINED_CSV"
echo "ANCHOR_EMBEDDINGS:   $ANCHOR_NPZ"
echo "DOC_ANCHORS:         $DOC_ANCHORS_CSV"
echo "ANCHORS_SUMMARY:     $ANCHORS_SUMMARY_CSV"
echo "ANCHORS_TOPICS:      $ANCHORS_TOPICS_CSV"

# ---------- Cleanup ----------
if [[ -n "$SERVER_PID" ]]; then echo "== Stop vLLM PID=$SERVER_PID"; kill "$SERVER_PID" || true; fi
echo "== $(date) : done"
