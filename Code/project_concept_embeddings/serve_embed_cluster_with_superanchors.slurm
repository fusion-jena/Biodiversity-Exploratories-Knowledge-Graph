#!/bin/bash
#SBATCH --job-name=embed-cluster-superanchors
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

# ---- Inputs ----
DATASETS_CSV="${DATASETS_CSV:-./datasets_merged.csv}"
PUBLICATIONS_CSV="${PUBLICATIONS_CSV:-./publications_merged.csv}"
ANCHORS_CSV="${ANCHORS_CSV:-./NASA-EarthData-GCMD-Keywords.csv}"
ANCHORS_ID_COL="${ANCHORS_ID_COL:-keyword}"
ANCHORS_TEXT_COL="${ANCHORS_TEXT_COL:-merged}"
ANCHORS_LABEL_COL="${ANCHORS_LABEL_COL:-keyword}"

OUTDIR="${OUTDIR:-$PWD/out_emb_cluster}"
mkdir -p "$OUTDIR"

# ---- Models / env ----
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
SERVE_LOCALLY="${SERVE_LOCALLY:-1}"
PORT="${PORT:-8000}"
REMOTE_BASE_URL="${REMOTE_BASE_URL:-}"
CONDA_ENV="${CONDA_ENV:-vllm-embed}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-20}"

# Anchor assignment (adaptive from earlier)
ANCHOR_THRESHOLD="${ANCHOR_THRESHOLD:-0.25}"
ANCHOR_TOPK="${ANCHOR_TOPK:-1}"
ADAPTIVE_STRATEGY="${ADAPTIVE_STRATEGY:-per-anchor-quantile}"
ADAPTIVE_QUANTILE="${ADAPTIVE_QUANTILE:-0.80}"
STOPWORDS_LANG="${STOPWORDS_LANG:-en,de}"
EXTRA_STOPWORDS="${EXTRA_STOPWORDS:-study,studies,analysis,analyses,data,dataset,approach,methods,using,result,results,impact,impacts}"
ANCHOR_NGRAM_MAX="${ANCHOR_NGRAM_MAX:-3}"
ANCHOR_MIN_DF="${ANCHOR_MIN_DF:-5}"
ANCHOR_TOP_K_TERMS="${ANCHOR_TOP_K_TERMS:-12}"

# Super-anchors controls
N_SUPER="${N_SUPER:-0}"         # if >0, fixed; else auto between MIN_K..MAX_K
SUPER_MIN_K="${SUPER_MIN_K:-25}"
SUPER_MAX_K="${SUPER_MAX_K:-60}"

echo "== Config =="
echo "EMBED_MODEL=$EMBED_MODEL SERVE_LOCALLY=$SERVE_LOCALLY PORT=$PORT REMOTE_BASE_URL=$REMOTE_BASE_URL"
echo "MIN_CLUSTER_SIZE=$MIN_CLUSTER_SIZE  N_SUPER=$N_SUPER  SUPER_MIN_K=$SUPER_MIN_K  SUPER_MAX_K=$SUPER_MAX_K"

# ---- Sanity: scripts ----
for f in concat_inputs.py embed_client.py cluster_and_label.py refine_labels.py anchor_assign.py cluster_anchors.py; do
  [[ -f "$f" ]] || { echo "❌ Missing $f in $(pwd)"; exit 1; }
done

# ---- Modules / conda ----
if command -v module &>/dev/null; then module purge || true; module load anaconda3 || true; module load cuda || true; fi
if command -v conda &>/dev/null; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV" || true; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

export ANCHOR_MIN_DF=8
export EXTRA_STOPWORDS="et,al,table,figure,km,mm,ppb,ppm"




# ---- Serve embeddings ----
API_BASE=""; SERVER_PID=""
if [[ "$SERVE_LOCALLY" == "1" ]]; then
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

# ---- Artifacts ----
COMBINED="$OUTDIR/combined_inputs.csv"
EMBED_NPZ="$OUTDIR/embeddings.npz"
CLUSTERS_CSV="$OUTDIR/clusters.csv"
TOPICS_CSV="$OUTDIR/topics.csv"
REFINED_CSV="$OUTDIR/topics_refined.csv"

ANCHOR_NPZ="$OUTDIR/anchor_embeddings.npz"
DOC_ANCHORS_CSV="$OUTDIR/anchor_assignments.csv"
ANCHORS_SUMMARY_CSV="$OUTDIR/anchors_summary.csv"
ANCHORS_TOPICS_CSV="$OUTDIR/anchors_topics.csv"

SUPER_NPZ="$OUTDIR/super_anchor_embeddings.npz"
SUPER_CSV="$OUTDIR/super_anchors.csv"
A2S_MAP="$OUTDIR/anchor_to_super.csv"

DOC_SUPER_ASSIGN_CSV="$OUTDIR/super_assignments.csv"
SUPER_SUMMARY_CSV="$OUTDIR/super_summary.csv"
SUPER_TOPICS_CSV="$OUTDIR/super_topics.csv"

# ---- Pipeline ----
echo "== Step 1/7: Concat"
python concat_inputs.py \
  --inputs "$DATASETS_CSV" "$PUBLICATIONS_CSV" \
  --id-col id --text-col merged --out "$COMBINED"

echo "== Step 2/7: Embed docs"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$COMBINED" --id-col id --text-col text \
  --batch-size 64 --out "$EMBED_NPZ"

echo "== Step 3/7: Cluster + label (docs)"
python cluster_and_label.py \
  --input-csv "$COMBINED" --embeddings "$EMBED_NPZ" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --clusters-out "$CLUSTERS_CSV" --topics-out "$TOPICS_CSV"

echo "== Step 4/7: Refine labels (docs)"
REFINE_ARGS=( --clusters-csv "$CLUSTERS_CSV" --topics-csv "$TOPICS_CSV" --out "$REFINED_CSV" --keyword --sentence )
python refine_labels.py "${REFINE_ARGS[@]}"

echo "== Step 5/7: Embed anchors"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$ANCHORS_CSV" --id-col "$ANCHORS_ID_COL" --text-col "$ANCHORS_TEXT_COL" \
  --batch-size 128 --out "$ANCHOR_NPZ"


export ANCHORS_ID_COL=keyword
export ANCHORS_LABEL_COL=keyword
export ANCHORS_TEXT_COL=merged

echo "== Step 6/7: Cluster anchors -> super-anchors"
python cluster_anchors.py \
  --anchor-embeddings "$ANCHOR_NPZ" \
  --anchor-csv "$ANCHORS_CSV" \
  --anchor-id-col "$ANCHORS_ID_COL" \
  --anchor-label-col "$ANCHORS_LABEL_COL" \
  --anchor-text-col "$ANCHORS_TEXT_COL" \
  --n-super "$N_SUPER" \
  --min-k "$SUPER_MIN_K" \
  --max-k "$SUPER_MAX_K" \
  --out-super-emb "$SUPER_NPZ" \
  --out-super-csv "$SUPER_CSV" \
  --out-mapping "$A2S_MAP"

echo "== Step 7/7: Assign docs to super-anchors (adaptive) + topics"
python anchor_assign.py \
  --doc-embeddings "$EMBED_NPZ" \
  --doc-csv "$COMBINED" \
  --anchor-embeddings "$SUPER_NPZ" \
  --anchor-csv "$SUPER_CSV" \
  --anchor-id-col super_id \
  --anchor-label-col super_label \
  --anchor-text-col super_text \
  --topk "$ANCHOR_TOPK" \
  --threshold "$ANCHOR_THRESHOLD" \
  --adaptive "$ADAPTIVE_STRATEGY" \
  --quantile "$ADAPTIVE_QUANTILE" \
  --stopwords-lang "$STOPWORDS_LANG" \
  --extra-stopwords "$EXTRA_STOPWORDS" \
  --ngram-max "$ANCHOR_NGRAM_MAX" \
  --min-df "$ANCHOR_MIN_DF" \
  --top-k-terms "$ANCHOR_TOP_K_TERMS" \
  --assignments-out "$DOC_SUPER_ASSIGN_CSV" \
  --summary-out     "$SUPER_SUMMARY_CSV" \
  --topics-out      "$SUPER_TOPICS_CSV"

echo "== Artifacts =="
echo "  SUPER ANCHORS: $SUPER_CSV"
echo "  MAP (anchor->super): $A2S_MAP"
echo "  SUPER EMBEDDINGS:    $SUPER_NPZ"
echo "  DOC->SUPER:          $DOC_SUPER_ASSIGN_CSV"
echo "  SUPER SUMMARY:       $SUPER_SUMMARY_CSV"
echo "  SUPER TOPICS:        $SUPER_TOPICS_CSV"

# ---- Cleanup ----
if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" || true; fi
echo "== $(date) : done"
