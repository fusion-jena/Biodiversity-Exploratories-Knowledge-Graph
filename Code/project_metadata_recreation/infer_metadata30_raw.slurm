#!/bin/bash
#SBATCH --job-name=be-infer-client
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=be-infer-client.out
#SBATCH --error=be-infer-client.err

set -euo pipefail
echo "== $(date) : starting on $(hostname -f)"

# ---- Config ----
: "${LIMIT_N:=4000}"  # process only N rows per split for this test run
: "${INPUT_DIR:=/input}"
: "${OUT_DIR:=$PWD/runs}"
: "${OPENAI_BASE_URL:=}"            # optional; can come from file
: "${API_URL_FILE:=$OUT_DIR/api_base_url.txt}"  # default location written by the server job
: "${MODEL:=mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
mkdir -p "$OUT_DIR" || true

# Resolve INPUT_DIR if the default doesn't exist
if [[ ! -d "$INPUT_DIR" ]]; then
  for cand in /input /inputs "$PWD/input" "$PWD/inputs" /mnt/data; do
    if [[ -d "$cand" ]]; then INPUT_DIR="$cand"; break; fi
  done
fi
echo "== INPUT_DIR=$INPUT_DIR"; ls -la "$INPUT_DIR" || true
echo "== OUT_DIR=$OUT_DIR"

# Resolve server URL
if [[ -z "${OPENAI_BASE_URL}" ]]; then
  if [[ -f "$API_URL_FILE" ]]; then
    OPENAI_BASE_URL="$(cat "$API_URL_FILE")"
  else
    echo "!! No OPENAI_BASE_URL and no $API_URL_FILE present. Set OPENAI_BASE_URL or point API_URL_FILE to the server-url file."
    exit 2
  fi
fi
OPENAI_BASE_URL="${OPENAI_BASE_URL%/}"


###

# If host has no dot (shortname like gpu014), append cluster domain

if [[ "$OPENAI_BASE_URL" =~ ^http://([^/:]+):([0-9]+)(/.*)?$ ]]; then

  host="${BASH_REMATCH[1]}"

  port="${BASH_REMATCH[2]}"

  rest="${BASH_REMATCH[3]}"

  if [[ "$host" != *.* ]]; then

    OPENAI_BASE_URL="http://${host}.cluster:${port}${rest}"

    echo "== Normalized OPENAI_BASE_URL → $OPENAI_BASE_URL"

  fi

fi



# Quick probes (non-fatal but informative)

for ep in health models; do

  code="$(curl -s -o /dev/null -w '%{http_code}' "$OPENAI_BASE_URL/$ep" || true)"

  echo "Probe $ep: $code"

done

###


echo "== Using OPENAI_BASE_URL=$OPENAI_BASE_URL"

# Quick health check (non-fatal)
for ep in health models; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "$OPENAI_BASE_URL/$ep" || true)"
  echo "Probe $ep: $code"
done

# Derived input files
PUB_INPUT="${INPUT_DIR}/publications_input.csv"
DS_INPUT="${INPUT_DIR}/datasets_input.csv"
PUB_MERGED="${INPUT_DIR}/publications_merged.csv"
DS_MERGED="${INPUT_DIR}/datasets_merged.csv"
PUB_XLSX="${INPUT_DIR}/publication_categories.xlsx"
DS_XLSX="${INPUT_DIR}/dataset_categories.xlsx"

# ---- Lightweight venv ----
module purge
module load anaconda3/2024.10 || true

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
SCRATCH_DIR="$(mktemp -d -p "$SCRATCH_BASE" "cli_${SLURM_JOB_ID:-$$}_XXXX")"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

python -m venv "$SCRATCH_DIR/venv"
source "$SCRATCH_DIR/venv/bin/activate"
python -V
PYTHON_BIN="$(which python)"
python -m pip install -U pip setuptools wheel
python -m pip install requests pandas jsonschema openpyxl

# ---- Build input CSVs from *_merged.csv if needed ----
$PYTHON_BIN - <<'PY'
import re, pandas as pd, os, sys
from pathlib import Path
INPUT = Path(os.environ.get("INPUT_DIR","/input"))
def split(merged: str):
    if not isinstance(merged, str): return "",""
    txt = merged.strip()
    m = re.search(r"(?is)title:\s*(.+?)(?:\n+|$)", txt); title = m.group(1).strip() if m else ""
    m2 = re.search(r"(?is)abstract:\s*(.+)$", txt); abstract = m2.group(1).strip() if m2 else ""
    if not title and txt:
        first = txt.splitlines()[0]
        if len(first) <= 200:
            import re as _re
            title = _re.sub(r"^title:\s*", "", first, flags=_re.I).strip()
    return title, abstract
def make(src, out):
    if out.exists():
        print(f"== Found {out}, skip")
        return
    if not src.exists():
        print(f"!! Missing {src}", file=sys.stderr)
        return
    df = pd.read_csv(src)
    assert {"id","merged"}.issubset(df.columns), f"{src} missing id/merged"
    rows = []
    for _, r in df.iterrows():
        t,a = split(r["merged"])
        rows.append({"id": r["id"], "title": t, "abstract": a})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"== Wrote {out} ({len(rows)} rows)")
make(INPUT/"publications_merged.csv", INPUT/"publications_input.csv")
make(INPUT/"datasets_merged.csv", INPUT/"datasets_input.csv")
PY

# ---- Embedded runner (writes to scratch) ----
RUNNER_PY="$SCRATCH_DIR/run_metadata_inference_v2.py"
cat > "$RUNNER_PY" <<'PYCODE'
#!/usr/bin/env python3
import os, json, argparse, re, requests
import pandas as pd

BASE_PROMPT_TEMPLATE = """You are an expert metadata curator for the Biodiversity Exploratories (BE) project.
Your task: given a TITLE and ABSTRACT, propose metadata as JSON.
RULES
- Output STRICTLY valid JSON (no comments). Do not include any text before or after the JSON.
- Choose ALL applicable values for multi-select-like fields.
- Free-text style fields are marked with "any".
- For free-text style fields: return an array of short phrases (e.g., ["soil biodiversity","grassland"]). Avoid hallucinations.
- For free-text style fields marked with "any long", longer phrases are allowed like "All species data have been standardized. This was done by dividing the recorded cover value of each species through the sum of all species cover values of a subplot. ..."
- If information is missing, you MAY return empty arrays [] or nulls. Do not invent specifics.
SCHEMA KEYS:
{schema_keys}
MULTI-SELECT FIELDS & ENUMS (if a list is provided below, prefer those exact labels; otherwise use short phrases):
{enum_block}
Return only a single JSON object with keys drawn from the schema keys above. Do NOT include any explanation."""

def normalize_key(s: str) -> str:
    s = re.sub(r'\s+', ' ', str(s).strip())
    s = s.replace('/', ' ').replace('\\', ' ').replace('–', '-').replace('—', '-')
    s = re.sub(r'[^0-9A-Za-z _-]+', '', s)
    s = s.strip().lower().replace(' ', '_')
    return s

def parse_allowed_answers(val):
    """Parse the 'allowed answers' column into a simple list (enum) when possible."""
    if val is None or (isinstance(val, float) and pd.isna(val)):  # NaN
        return []
    txt = str(val).strip()
    if not txt:
        return []
    # Split by semicolons or commas
    cleaned = txt.replace('’', "'").replace('`', "'")
    parts = re.split(r'\s*[;,]\s*', re.sub(r"^'+|'+$", "", cleaned))
    vals = [p.strip().strip("'").strip('"').strip() for p in parts if p.strip()]
    # Deduplicate case-insensitively while preserving order
    seen = set()
    uniq = []
    for v in vals:
        lv = v.lower()
        if lv not in seen:
            uniq.append(v)
            seen.add(lv)
    return uniq

def build_schema_info_from_xlsx(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    required_cols = {"category name","category definition","allowed answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{xlsx_path}: expected columns {sorted(required_cols)}, got {df.columns.tolist()}")
    props = {}
    for _, row in df.iterrows():
        orig = str(row["category name"]).strip()
        key = normalize_key(orig)
        desc = str(row["category definition"]).strip() if pd.notna(row["category definition"]) else ""
        enum = parse_allowed_answers(row.get("allowed answers"))
        props[key] = {"original": orig, "description": desc, "enum": enum}
    return props

def build_prompts(schema_info, title, abstract):
    schema_keys = ", ".join(schema_info.keys())
    enum_lines = []
    for k, meta in schema_info.items():
        if meta["enum"]:
            enum_lines.append(f"- {k}: {meta['enum']}")
        else:
            enum_lines.append(f"- {k}: (free-text array OK)")
    enum_block = "\n".join(enum_lines)
    system_prompt = BASE_PROMPT_TEMPLATE.format(schema_keys=schema_keys, enum_block=enum_block)
    user_prompt = f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}\n\nReturn ONLY the JSON object (no prose)."
    return system_prompt, user_prompt

def call_openai_chat(*, model, base_url, api_key, system_prompt, user_prompt, temperature=0.0, max_tokens=1500):
    url = base_url.rstrip('/') + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    jd = r.json()
    return jd["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--categories_xlsx", required=True)
    ap.add_argument("--schema_title", required=True)  # kept for CLI compatibility (not used directly)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"id","title","abstract"}
    if not required.issubset(df.columns):
        raise ValueError(f"{args.input_csv} must have columns id,title,abstract")
    if args.limit is not None:
        df = df.head(int(args.limit))

    schema_info = build_schema_info_from_xlsx(args.categories_xlsx)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    ok = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            rid = row["id"]
            system_prompt, user_prompt = build_prompts(schema_info, row["title"], row["abstract"])
            try:
                content = call_openai_chat(
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                fout.write(json.dumps({"id": rid, "raw": content}, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                fout.write(json.dumps({"id": rid, "error": str(e)}, ensure_ascii=False) + "\n")
    print(f"Wrote {ok}/{len(df)} raw outputs to {args.output_jsonl}")

if __name__ == "__main__":
    main()

PYCODE
chmod +x "$RUNNER_PY"

# ---- Run Publications ----
if [[ -f "$PUB_INPUT" && -f "$PUB_XLSX" ]]; then
  OPENAI_BASE_URL="$OPENAI_BASE_URL" "$PYTHON_BIN" "$RUNNER_PY" \
    --input_csv       "$PUB_INPUT" \
    --output_jsonl    "$OUT_DIR/publications_predictions.jsonl" \
    --categories_xlsx "$PUB_XLSX" \
    --schema_title    "BE Publication Metadata (multi-select)" \
    --model "$MODEL" \
    --temperature 0.0 \
    --max_tokens 1500 \
    --limit ${LIMIT_N:-30}
else
  echo "!! Skipping publications: missing $PUB_INPUT or $PUB_XLSX"
fi

# ---- Run Datasets ----
if [[ -f "$DS_INPUT" && -f "$DS_XLSX" ]]; then
  OPENAI_BASE_URL="$OPENAI_BASE_URL" "$PYTHON_BIN" "$RUNNER_PY" \
    --input_csv       "$DS_INPUT" \
    --output_jsonl    "$OUT_DIR/datasets_predictions.jsonl" \
    --categories_xlsx "$DS_XLSX" \
    --schema_title    "BE Dataset Metadata (multi-select)" \
    --model "$MODEL" \
    --temperature 0.0 \
    --max_tokens 1500 \
    --limit ${LIMIT_N:-30}
else
  echo "!! Skipping datasets: missing $DS_INPUT or $DS_XLSX"
fi

echo "== Artifacts =="
ls -la "$OUT_DIR" || true
echo "Done."