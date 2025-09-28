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

# Fail fast + treat unset vars as errors
set -euo pipefail

echo "== $(date) : starting on $(hostname -f)"

#####################################
# Paths & inputs (override via env) #
#####################################
DATASETS_CSV="${DATASETS_CSV:-./datasets_merged.csv}"
PUBLICATIONS_CSV="${PUBLICATIONS_CSV:-./publications_merged.csv}"
ANCHORS_CSV="${ANCHORS_CSV:-./NASA-EarthData-GCMD-Keywords.csv}"

# Anchors CSV column names
ANCHORS_ID_COL="${ANCHORS_ID_COL:-keyword}"     # IDs used when embedding anchors
ANCHORS_TEXT_COL="${ANCHORS_TEXT_COL:-merged}"  # your merged (keyword + def + extras)
ANCHORS_LABEL_COL="${ANCHORS_LABEL_COL:-keyword}"

# Output directory
OUTDIR="${OUTDIR:-$PWD/out_emb_cluster}"
mkdir -p "$OUTDIR"

################
# Model / env  #
################
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
SERVE_LOCALLY="${SERVE_LOCALLY:-1}"
PORT="${PORT:-8000}"
REMOTE_BASE_URL="${REMOTE_BASE_URL:-}"

CONDA_ENV="${CONDA_ENV:-vllm-embed}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-20}"

# Optional generator for refine_labels
GEN_API_BASE="${GEN_API_BASE:-}"
GEN_MODEL="${GEN_MODEL:-}"

###################################
# Anchor assignment configuration #
###################################
ANCHOR_THRESHOLD="${ANCHOR_THRESHOLD:-0.25}"       # hard floor
ANCHOR_TOPK="${ANCHOR_TOPK:-1}"

# NEW: adaptive thresholds (Patch 1)
ADAPTIVE_STRATEGY="${ADAPTIVE_STRATEGY:-per-anchor-quantile}"  # none | global-quantile | per-anchor-quantile
ADAPTIVE_QUANTILE="${ADAPTIVE_QUANTILE:-0.80}"

# Labeling knobs for anchors_topics
STOPWORDS_LANG="${STOPWORDS_LANG:-en,de}"
EXTRA_STOPWORDS="${EXTRA_STOPWORDS:-study,studies,analysis,analyses,data,dataset,approach,methods,using,result,results,impact,impacts}"
ANCHOR_NGRAM_MAX="${ANCHOR_NGRAM_MAX:-3}"
ANCHOR_MIN_DF="${ANCHOR_MIN_DF:-5}"
ANCHOR_TOP_K_TERMS="${ANCHOR_TOP_K_TERMS:-12}"

echo "== Config =="
echo "DATASETS_CSV=$DATASETS_CSV"
echo "PUBLICATIONS_CSV=$PUBLICATIONS_CSV"
echo "ANCHORS_CSV=$ANCHORS_CSV (id=$ANCHORS_ID_COL text=$ANCHORS_TEXT_COL label=$ANCHORS_LABEL_COL)"
echo "EMBED_MODEL=$EMBED_MODEL  SERVE_LOCALLY=$SERVE_LOCALLY  PORT=$PORT  REMOTE_BASE_URL=$REMOTE_BASE_URL"
echo "CONDA_ENV=$CONDA_ENV  MIN_CLUSTER_SIZE=$MIN_CLUSTER_SIZE"
echo "ANCHOR_THRESHOLD=$ANCHOR_THRESHOLD  ANCHOR_TOPK=$ANCHOR_TOPK"
echo "ADAPTIVE_STRATEGY=$ADAPTIVE_STRATEGY  ADAPTIVE_QUANTILE=$ADAPTIVE_QUANTILE"
echo "STOPWORDS_LANG=$STOPWORDS_LANG  EXTRA_STOPWORDS=$EXTRA_STOPWORDS"
echo "ANCHOR_NGRAM_MAX=$ANCHOR_NGRAM_MAX  ANCHOR_MIN_DF=$ANCHOR_MIN_DF  ANCHOR_TOP_K_TERMS=$ANCHOR_TOP_K_TERMS"
echo "GEN_API_BASE=$GEN_API_BASE  GEN_MODEL=$GEN_MODEL"

########################
# Sanity: scripts/data #
########################
for f in concat_inputs.py embed_client.py cluster_and_label.py refine_labels.py anchor_assign.py; do
  [[ -f "$f" ]] || { echo "❌ Missing $f in $(pwd)"; exit 1; }
done
for f in "$DATASETS_CSV" "$PUBLICATIONS_CSV" "$ANCHORS_CSV"; do
  [[ -f "$f" ]] || { echo "❌ Missing input CSV: $f"; exit 1; }
done

########################
# Modules / Conda env  #
########################
if command -v module &>/dev/null; then
  module purge || true
  module load anaconda3 || true
  module load cuda || true
fi
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# flags:
#export ADAPTIVE_STRATEGY=per-anchor-quantile
#export ADAPTIVE_QUANTILE=0.85   # stricter than 0.80
#export ANCHOR_THRESHOLD=0.28    # keep a hard minimum floor

#export ADAPTIVE_STRATEGY=none



########################
# Serve embeddings API #
########################
API_BASE=""
SERVER_PID=""
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

#############
# Artifacts #
#############
COMBINED="$OUTDIR/combined_inputs.csv"
EMBED_NPZ="$OUTDIR/embeddings.npz"
CLUSTERS_CSV="$OUTDIR/clusters.csv"
TOPICS_CSV="$OUTDIR/topics.csv"
REFINED_CSV="$OUTDIR/topics_refined.csv"

ANCHOR_NPZ="$OUTDIR/anchor_embeddings.npz"
DOC_ANCHORS_CSV="$OUTDIR/anchor_assignments.csv"
ANCHORS_SUMMARY_CSV="$OUTDIR/anchors_summary.csv"
ANCHORS_TOPICS_CSV="$OUTDIR/anchors_topics.csv"

########################
# Pipeline (6 steps)   #
########################
echo "== Step 1/6: Concat"
python concat_inputs.py \
  --inputs "$DATASETS_CSV" "$PUBLICATIONS_CSV" \
  --id-col id --text-col merged --out "$COMBINED"

echo "== Step 2/6: Embed docs"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$COMBINED" --id-col id --text-col text \
  --batch-size 64 --out "$EMBED_NPZ"

echo "== Step 3/6: Cluster + label"
python cluster_and_label.py \
  --input-csv "$COMBINED" --embeddings "$EMBED_NPZ" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --clusters-out "$CLUSTERS_CSV" --topics-out "$TOPICS_CSV"

echo "== Step 4/6: Refine labels (optional LLM polish)"
REFINE_ARGS=( --clusters-csv "$CLUSTERS_CSV" --topics-csv "$TOPICS_CSV" --out "$REFINED_CSV" --keyword --sentence )
if [[ -n "$GEN_API_BASE" && -n "$GEN_MODEL" ]]; then
  REFINE_ARGS+=( --llm-api-base "$GEN_API_BASE" --llm-model "$GEN_MODEL" )
fi
python refine_labels.py "${REFINE_ARGS[@]}"

echo "== Step 5/6: Embed anchors (NASA GCMD)"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$ANCHORS_CSV" --id-col "$ANCHORS_ID_COL" --text-col "$ANCHORS_TEXT_COL" \
  --batch-size 128 --out "$ANCHOR_NPZ"

echo "== Step 6/6: Assign docs to anchors (adaptive thresholds) + topics"
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
  --adaptive "$ADAPTIVE_STRATEGY" \
  --quantile "$ADAPTIVE_QUANTILE" \
  --stopwords-lang "$STOPWORDS_LANG" \
  --extra-stopwords "$EXTRA_STOPWORDS" \
  --ngram-max "$ANCHOR_NGRAM_MAX" \
  --min-df "$ANCHOR_MIN_DF" \
  --top-k-terms "$ANCHOR_TOP_K_TERMS" \
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

# Cleanup
if [[ -n "$SERVER_PID" ]]; then
  echo "== Stop vLLM PID=$SERVER_PID"
  kill "$SERVER_PID" || true
fi
echo "== $(date) : done"
