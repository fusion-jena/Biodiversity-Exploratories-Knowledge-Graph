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

def call_openai_chat(model: str, base_url: str, api_key: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def normalize_key(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("/", " or ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s

def parse_allowed_answers(raw: str):
    if raw is None or str(raw).strip() == "":
        return {}
    txt = str(raw).strip()
    if txt.lower() in {"integer", "int"}:
        return {"type": "integer"}
    cleaned = txt.replace("’", "'").replace("`", "'")
    parts = re.split(r"\s*[;,]\s*", re.sub(r"^'+|'+$", "", cleaned))
    vals = []
    for p in parts:
        p2 = p.strip().strip("'").strip('"').strip()
        if p2:
            vals.append(p2)
    seen = set()
    uniq = []
    for v in vals:
        lv = v.lower()
        if lv not in seen:
            uniq.append(v)
            seen.add(lv)
    if uniq:
        return {"type": "string", "enum": uniq}
    return {}

def build_schema_from_xlsx_multiselect(xlsx_path: str, title: str):
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    required_cols = {"category name", "category definition", "allowed answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{xlsx_path}: expected columns {required_cols}, got {df.columns.tolist()}")
    props = {}
    for _, row in df.iterrows():
        orig = str(row["category name"]).strip()
        key = normalize_key(orig)
        desc = str(row["category definition"]).strip() if pd.notna(row["category definition"]) else ""
        spec = parse_allowed_answers(row["allowed answers"] if "allowed answers" in df.columns else None)
        if spec.get("type") == "integer":
            prop = {"type": "integer", "description": desc}
        else:
            items = {"type": "string"}
            if "enum" in spec:
                items["enum"] = spec["enum"]
            prop = {"type": "array", "items": items, "uniqueItems": False, "description": desc}
        prop["x_original_name"] = orig
        props[key] = prop

    return {"$schema":"http://json-schema.org/draft-07/schema#","title":title,"type":"object","additionalProperties":False,"properties":props}

def build_prompts(schema, title: str, abstract: str):
    props = schema.get("properties", {})
    schema_keys = ", ".join(props.keys())

    enum_lines = []
    for k, v in props.items():
        if v.get("type") == "array" and isinstance(v.get("items", {}), dict) and "enum" in v["items"]:
            enum_lines.append(f"- {k}: " + ", ".join(map(str, v["items"]["enum"])) )
        elif "enum" in v:
            enum_lines.append(f"- {k}: " + ", ".join(map(str, v["enum"])) )
    enum_block = "\n".join(enum_lines) if enum_lines else "(none)"

    def_lines = []
    for k, v in props.items():
        desc = v.get("description") or v.get("x_original_name") or ""
        if desc:
            def_lines.append(f"- {k}: {desc}")
    defs_block = "\n".join(def_lines) if def_lines else "(none)"

    system_prompt = BASE_PROMPT_TEMPLATE.format(schema_keys=schema_keys, enum_fields=enum_block, field_defs=defs_block)

    user_prompt = f"""TITLE:
{title}

ABSTRACT:
{abstract}

Now produce ONLY the JSON object with fields as per the schema keys above."""
    return system_prompt, user_prompt

def clean_and_validate(candidate: str, validator: Draft7Validator, schema):
    errors = []
    try:
        text = candidate.strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL).strip()
        data = json.loads(text)
    except Exception as e:
        return None, [f"JSON parse error: {e}"]

    # For array-typed fields, coerce scalar string -> [string] and normalize enum casing
    props = schema.get("properties", {})
    for k, p in props.items():
        if p.get("type") == "array" and k in data:
            if isinstance(data[k], str):
                data[k] = [data[k]]
            elif data[k] is None:
                data[k] = []
            if "items" in p and isinstance(p["items"], dict) and "enum" in p["items"]:
                enums = p["items"]["enum"]
                fixed = []
                for item in data[k]:
                    if isinstance(item, str):
                        match = next((e for e in enums if e.lower() == item.lower()), item)
                        fixed.append(match)
                data[k] = fixed

    errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
    for e in errs:
        errors.append(f"{list(e.path)}: {e.message}")
    return (None, errors) if errors else (data, [])

def main():
    ap = argparse.ArgumentParser(description="Infer BE metadata from TITLE+ABSTRACT via LLM, using Excel categories.")
    ap.add_argument("--input_csv", required=True, help="CSV with columns: id,title,abstract")
    ap.add_argument("--output_jsonl", required=True, help="Write JSONL predictions here.")
    ap.add_argument("--categories_xlsx", required=True, help="Excel with 'category name/definition/allowed answers' to build schema.")
    ap.add_argument("--schema_title", default="BE Metadata (multi-select)")
    ap.add_argument("--model", default=os.environ.get("MODEL","mistral"))
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL","http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1500)
    args = ap.parse_args()

    # Build schema from Excel
    schema = build_schema_from_xlsx_multiselect(args.categories_xlsx, args.schema_title)
    validator = Draft7Validator(schema)

    n, ok = 0, 0
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.input_csv, newline="", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        need = {"id","title","abstract"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV must contain columns {need}, got {r.fieldnames}")
        for row in r:
            n += 1
            rid = row["id"]
            system_prompt, user_prompt = build_prompts(schema, row["title"], row["abstract"])
            content = None
            try:
                content = call_openai_chat(model=args.model, base_url=args.base_url, api_key=args.api_key,
                                           system_prompt=system_prompt, user_prompt=user_prompt,
                                           temperature=args.temperature, max_tokens=args.max_tokens)
                data, errors = clean_and_validate(content, validator, schema)
            except Exception as e:
                content = None
                data, errors = None, [f"LLM call failed: {e}"]

            valid = data is not None and not errors
            if valid: ok += 1
            fout.write(json.dumps({"id": rid, "valid": valid, "errors": errors, "prediction": data if valid else None, "raw": content if not valid else None}, ensure_ascii=False) + "\n")

    print(f"Valid {ok}/{n}")

if __name__ == "__main__":
    main()
