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
import os, json, argparse, csv, re, requests
from jsonschema import Draft7Validator
BASE_PROMPT_TEMPLATE = """You are an expert metadata curator for the Biodiversity Exploratories (BE) project.
Your task: given a TITLE and ABSTRACT, fill in the BE metadata as JSON.
RULES
- Output STRICTLY valid JSON (no comments). Do not include any text before or after the JSON.
- Use the JSON schema provided below to decide field names and types.
- For fields with ENUMS: choose ALL applicable values from the list. Return an array of strings (e.g., ["GP","EP"]). If unknown, return an empty array [].
- For free-text fields: return an array of short phrases (e.g., ["soil biodiversity","grassland"]). No hallucinated facts.
- If the information is missing, use [] for arrays or null for integers. Do not invent.
- Use consistent casing. Prefer the exact labels from the enum list.
- Dates: use ISO 8601 (YYYY-MM-DD) if a date is requested.
SCHEMA KEYS:
{schema_keys}
MULTI-SELECT FIELDS & ENUMS (choose ALL applicable values; return arrays of strings):
{enum_fields}
FIELD DEFINITIONS (use these to decide what belongs where):
{field_defs}
Return only a single JSON object whose keys match the schema. Do NOT include any explanation.
"""
def call_openai_chat(model, base_url, api_key, system_prompt, user_prompt, temperature, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]}
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status(); data = r.json()
    return data["choices"][0]["message"]["content"]
def normalize_key(name:str)->str:
    s = name.strip().lower().replace("/", " or ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s); s = re.sub(r"\s+", "_", s).strip("_"); return s
def parse_allowed_answers(raw):
    if raw is None or str(raw).strip() == "": return {}
    txt = str(raw).strip()
    if txt.lower() in {"integer","int"}: return {"type":"integer"}
    cleaned = txt.replace("’","'").replace("`","'")
    parts = re.split(r"\s*[;,]\s*", re.sub(r"^'+|'+$", "", cleaned))
    vals = [p.strip().strip("'").strip('"').strip() for p in parts if p.strip()]
    seen=set(); uniq=[]
    for v in vals:
        lv=v.lower()
        if lv not in seen:
            uniq.append(v); seen.add(lv)
    return {"type":"string","enum":uniq} if uniq else {}
def build_schema_from_xlsx_multiselect(xlsx_path: str, title: str):
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    required_cols = {"category name","category definition","allowed answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{xlsx_path}: expected {required_cols}, got {df.columns.tolist()}")
    props={}
    for _, row in df.iterrows():
        orig=str(row["category name"]).strip(); key=normalize_key(orig)
        desc=str(row["category definition"]).strip() if pd.notna(row["category definition"]) else ""
        spec=parse_allowed_answers(row["allowed answers"] if "allowed answers" in df.columns else None)
        if spec.get("type")=="integer":
            prop={"type":"integer","description":desc}
        else:
            items={"type":"string"}
            if "enum" in spec: items["enum"]=spec["enum"]
            prop={"type":"array","items":items,"uniqueItems":False,"description":desc}
        prop["x_original_name"]=orig; props[key]=prop
    return {"$schema":"http://json-schema.org/draft-07/schema#","title":title,"type":"object","additionalProperties":False,"properties":props}
def build_prompts(schema, title, abstract):
    props=schema.get("properties",{}); schema_keys=", ".join(props.keys())
    enum_lines=[]
    for k,v in props.items():
        if v.get("type")=="array" and isinstance(v.get("items",{}),dict) and "enum" in v["items"]:
            enum_lines.append(f"- {k}: "+", ".join(map(str, v["items"]["enum"])))
        elif "enum" in v:
            enum_lines.append(f"- {k}: "+", ".join(map(str, v["enum"])))
    enum_block="\n".join(enum_lines) if enum_lines else "(none)"
    def_lines=[]
    for k,v in props.items():
        desc=v.get("description") or v.get("x_original_name") or ""
        if desc: def_lines.append(f"- {k}: {desc}")
    defs_block="\n".join(def_lines) if def_lines else "(none)"
    system_prompt=BASE_PROMPT_TEMPLATE.format(schema_keys=schema_keys, enum_fields=enum_block, field_defs=defs_block)
    user_prompt=f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}\n\nNow produce ONLY the JSON object with fields as per the schema keys above."
    return system_prompt, user_prompt
def clean_and_validate(candidate, validator, schema):
    errors=[]
    try:
        text=candidate.strip(); text=re.sub(r"^```json\s*|\s*```$","",text,flags=re.I|re.S).strip()
        data=json.loads(text)
    except Exception as e:
        return None, [f"JSON parse error: {e}"]
    props=schema.get("properties",{})
    for k,p in props.items():
        if p.get("type")=="array" and k in data:
            if isinstance(data[k], str): data[k]=[data[k]]
            elif data[k] is None: data[k]=[]
            if "items" in p and isinstance(p["items"],dict) and "enum" in p["items"]:
                enums=p["items"]["enum"]; fixed=[]
                for item in data[k]:
                    if isinstance(item,str):
                        match=next((e for e in enums if e.lower()==item.lower()), item)
                        fixed.append(match)
                data[k]=fixed
    errs=sorted(validator.iter_errors(data), key=lambda e: e.path)
    for e in errs: errors.append(f"{list(e.path)}: {e.message}")
    return (None, errors) if errors else (data, [])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--categories_xlsx", required=True)
    ap.add_argument("--schema_title", default="BE Metadata (multi-select)")
    ap.add_argument("--model", default=os.environ.get("MODEL","mistral"))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL","http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1500)
    args=ap.parse_args()
    schema=build_schema_from_xlsx_multiselect(args.categories_xlsx, args.schema_title)
    validator=Draft7Validator(schema)
    n=ok=0
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.input_csv, newline="", encoding="utf-8") as fin, open(args.output_jsonl,"w",encoding="utf-8") as fout:
        r=csv.DictReader(fin); need={"id","title","abstract"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV must contain columns {need}, got {r.fieldnames}")
        for row in r:
            n+=1; rid=row["id"]
            system_prompt,user_prompt=build_prompts(schema,row["title"],row["abstract"])
            content=None
            try:
                content=call_openai_chat(model=args.model, base_url=args.base_url, api_key=args.api_key,
                                         system_prompt=system_prompt, user_prompt=user_prompt,
                                         temperature=args.temperature, max_tokens=args.max_tokens)
                data,errors=clean_and_validate(content, validator, schema)
            except Exception as e:
                content=None; data,errors=None,[f"LLM call failed: {e}"]
            valid = data is not None and not errors
            if valid: ok+=1
            fout.write(json.dumps({"id":rid,"valid":valid,"errors":errors,"prediction":data if valid else None,"raw":content if not valid else None},ensure_ascii=False)+"\n")
    print(f"Valid {ok}/{n}")
if __name__=="__main__": main()
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
    --max_tokens 1500
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
    --max_tokens 1500
else
  echo "!! Skipping datasets: missing $DS_INPUT or $DS_XLSX"
fi

echo "== Artifacts =="
ls -la "$OUT_DIR" || true
echo "Done."
