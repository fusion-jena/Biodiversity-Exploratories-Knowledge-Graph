#!/bin/bash
#SBATCH --job-name=biodiv-goal-classify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=goals_%j.out
#SBATCH --error=goals_%j.err

set -euo pipefail
echo "== $(date) : starting on $(hostname -f)"

#######################
# Paths / config (env)
#######################
# Inputs: two CSVs with columns: id, merged  (merged = "Title: ... \n Abstract: ...")
DATASETS_CSV="${DATASETS_CSV:-./datasets_merged.csv}"
PUBLICATIONS_CSV="${PUBLICATIONS_CSV:-./publications_merged.csv}"

# Output directory
OUTDIR="${OUTDIR:-$PWD/out_biodiv_goals}"
mkdir -p "$OUTDIR"

# Embedding model / serving
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
SERVE_LOCALLY="${SERVE_LOCALLY:-1}"
PORT="${PORT:-8000}"
REMOTE_BASE_URL="${REMOTE_BASE_URL:-}"
CONDA_ENV="${CONDA_ENV:-vllm-embed}"

# Assignment knobs
THRESHOLD="${THRESHOLD:-0.25}"             # hard floor for acceptance
TOPK="${TOPK:-1}"
ADAPTIVE_STRATEGY="${ADAPTIVE_STRATEGY:-per-anchor-quantile}"  # none|global-quantile|per-anchor-quantile
ADAPTIVE_QUANTILE="${ADAPTIVE_QUANTILE:-0.80}"

# Labeling (top term extraction) knobs
STOPWORDS_LANG="${STOPWORDS_LANG:-en,de}"
EXTRA_STOPWORDS="${EXTRA_STOPWORDS:-study,studies,analysis,analyses,data,dataset,approach,methods,using,result,results,impact,impacts}"
NGRAM_MAX="${NGRAM_MAX:-3}"
MIN_DF="${MIN_DF:-5}"
TOP_K_TERMS="${TOP_K_TERMS:-12}"

echo "== Config =="
echo "DATASETS_CSV=$DATASETS_CSV"
echo "PUBLICATIONS_CSV=$PUBLICATIONS_CSV"
echo "OUTDIR=$OUTDIR"
echo "EMBED_MODEL=$EMBED_MODEL  SERVE_LOCALLY=$SERVE_LOCALLY PORT=$PORT  REMOTE_BASE_URL=$REMOTE_BASE_URL"
echo "THRESHOLD=$THRESHOLD  TOPK=$TOPK  ADAPTIVE_STRATEGY=$ADAPTIVE_STRATEGY  ADAPTIVE_QUANTILE=$ADAPTIVE_QUANTILE"
echo "STOPWORDS_LANG=$STOPWORDS_LANG  EXTRA_STOPWORDS=$EXTRA_STOPWORDS  NGRAM_MAX=$NGRAM_MAX  MIN_DF=$MIN_DF  TOP_K_TERMS=$TOP_K_TERMS"

########################
# Sanity: scripts/data
########################
for f in build_goals_csv.py embed_client.py concat_inputs.py classify_to_goals.py; do
  [[ -f "$f" ]] || { echo "❌ Missing $f in $(pwd)"; exit 1; }
done
for f in "$DATASETS_CSV" "$PUBLICATIONS_CSV"; do
  [[ -f "$f" ]] || { echo "❌ Missing input CSV: $f"; exit 1; }
done

########################
# Modules / Conda env
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

########################
# Serve embeddings API
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
  curl -s "${API_BASE}/v1/models" >/dev/null || { echo "❌ vLLM failed to start"; kill "$SERVER_PID" || true; exit 1; }
else
  [[ -n "$REMOTE_BASE_URL" ]] || { echo "❌ Need REMOTE_BASE_URL"; exit 1; }
  API_BASE="$REMOTE_BASE_URL"; curl -s "$API_BASE/v1/models" >/dev/null || { echo "❌ Can't reach $API_BASE"; exit 1; }
fi

#############
# Artifacts
#############
GOALS_CSV="$OUTDIR/goals.csv"
COMBINED="$OUTDIR/combined_inputs.csv"
DOC_EMB="$OUTDIR/doc_embeddings.npz"
GOAL_EMB="$OUTDIR/goal_embeddings.npz"
ASSIGN_CSV="$OUTDIR/goal_assignments.csv"
SUMMARY_CSV="$OUTDIR/goals_summary.csv"
TOPICS_CSV="$OUTDIR/goals_topics.csv"

########################
# Pipeline
########################
echo "== Step 1/5: Write Biodiversity Exploratories goals"
python build_goals_csv.py --out "$GOALS_CSV"

echo "== Step 2/5: Concatenate inputs"
python concat_inputs.py \
  --inputs "$DATASETS_CSV" "$PUBLICATIONS_CSV" \
  --id-col id --text-col merged \
  --out "$COMBINED"

echo "== Step 3/5: Embed documents"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$COMBINED" --id-col id --text-col text \
  --batch-size 64 --out "$DOC_EMB"

echo "== Step 4/5: Embed goals"
python embed_client.py \
  --api-base "$API_BASE" --model "$EMBED_MODEL" \
  --input-csv "$GOALS_CSV" --id-col id --text-col text \
  --batch-size 16 --out "$GOAL_EMB"

echo "== Step 5/5: Classify docs to nearest goal (with optional adaptive cutoff) + topic terms"
python classify_to_goals.py \
  --doc-embeddings "$DOC_EMB" \
  --doc-csv "$COMBINED" \
  --goal-embeddings "$GOAL_EMB" \
  --goal-csv "$GOALS_CSV" \
  --topk "$TOPK" \
  --threshold "$THRESHOLD" \
  --adaptive "$ADAPTIVE_STRATEGY" \
  --quantile "$ADAPTIVE_QUANTILE" \
  --stopwords-lang "$STOPWORDS_LANG" \
  --extra-stopwords "$EXTRA_STOPWORDS" \
  --ngram-max "$NGRAM_MAX" \
  --min-df "$MIN_DF" \
  --top-k-terms "$TOP_K_TERMS" \
  --assignments-out "$ASSIGN_CSV" \
  --summary-out     "$SUMMARY_CSV" \
  --topics-out      "$TOPICS_CSV"

echo "== Artifacts =="
echo "GOALS_CSV:       $GOALS_CSV"
echo "COMBINED:        $COMBINED"
echo "DOC_EMB:         $DOC_EMB"
echo "GOAL_EMB:        $GOAL_EMB"
echo "ASSIGNMENTS:     $ASSIGN_CSV"
echo "SUMMARY:         $SUMMARY_CSV"
echo "TOPICS:          $TOPICS_CSV"

if [[ -n "$SERVER_PID" ]]; then echo "== Stop vLLM PID=$SERVER_PID"; kill "$SERVER_PID" || true; fi
echo "== $(date) : done"
