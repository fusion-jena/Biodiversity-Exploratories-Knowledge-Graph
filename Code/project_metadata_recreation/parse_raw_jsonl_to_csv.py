#!/usr/bin/env python3
"""
Parse LLM raw JSONL outputs (with possible ```json fences, arrays/strings/booleans)
into a flattened CSV.

Usage examples:
  python parse_raw_jsonl_to_csv.py       --input out/publications_predictions.jsonl       --output out/publications_predictions.csv

  python parse_raw_jsonl_to_csv.py       --input out/datasets_predictions.jsonl       --output out/datasets_predictions.csv
"""
import argparse, json, re, sys, csv
from typing import Any, Dict, List, Tuple

YES_NO_KEYS_DEFAULT = [
    "alb","sch","hai","grassland","forest","aboveground","belowground",
    "field","laboratory","review","rex1","rex2","lux","fox"
]

def strip_code_fences(s: str) -> str:
    # Remove leading/trailing triple backticks (with or without 'json')
    s = s.strip()
    # drop starting ```json or ```
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.DOTALL)
    # drop trailing ```
    s = re.sub(r"\s*```$", "", s, flags=re.DOTALL)
    return s.strip()

def lenient_json_loads(s: str) -> Any:
    """
    Try to parse JSON with some common cleanups:
    - strip code fences
    - remove trailing commas before ] or }
    - replace smart quotes with straight quotes
    """
    t = strip_code_fences(s)
    t = t.replace("’", "'").replace("“", '"').replace("”", '"').replace("`", "'")
    # remove trailing commas before a closing bracket/brace
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return json.loads(t)

def to_yes_no(val) -> str:
    """
    Normalize yes/no style fields to 'yes' or 'no' (lowercase).
    Accepts 'yes'/'no' strings, booleans, arrays containing yes/no, etc.
    Defaults to '' (empty) if no value could be determined.
    """
    def norm_one(x) -> str:
        if isinstance(x, bool):  # True/False
            return "yes" if x else "no"
        if x is None:
            return ""
        if isinstance(x, (int, float)):
            # Non-zero -> yes; zero -> no (fallback heuristic)
            return "yes" if x else "no"
        s = str(x).strip().lower()
        if s in {"yes","y","true","1"}:
            return "yes"
        if s in {"no","n","false","0"}:
            return "no"
        return ""

    if isinstance(val, list):
        # pick the first decisive answer in the list, else join
        for x in val:
            v = norm_one(x)
            if v in {"yes","no"}:
                return v
        # as fallback, join list as text
        return "; ".join(str(x) for x in val)
    return norm_one(val)

def to_list(val) -> List[str]:
    """Ensure list-of-strings (used for multi-select / keywords etc.)."""
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            if x is None: 
                continue
            out.append(str(x).strip())
        return out
    # single scalar -> single-item list
    return [str(val).strip()]

def flatten_record(obj: Dict[str, Any],
                   yes_no_keys: List[str]) -> Dict[str, str]:
    """
    Convert a dict of fields into CSV-able strings:
    - yes/no keys -> 'yes'/'no'
    - lists -> '; ' joined
    - numbers -> keep as string
    - other scalars -> string
    """
    row: Dict[str, str] = {}
    for k, v in obj.items():
        if k in yes_no_keys:
            row[k] = to_yes_no(v)
        else:
            if isinstance(v, list):
                row[k] = "; ".join(str(x) for x in v if x is not None)
            elif isinstance(v, (int, float)):
                row[k] = str(v)
            elif v is None:
                row[k] = ""
            else:
                row[k] = str(v)
    return row

def parse_jsonl_to_rows(path: str,
                        yes_no_keys: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    all_keys: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                rows.append({"id":"", "error": f"JSONL parse error line {line_no}: {e}"})
                continue
            rid = rec.get("id","")
            raw = rec.get("raw")
            err = rec.get("error")
            if raw is None and err:
                rows.append({"id": str(rid), "error": str(err)})
                continue
            if raw is None:
                rows.append({"id": str(rid), "error": "missing raw"})
                continue
            try:
                obj = lenient_json_loads(raw)
            except Exception as e:
                rows.append({"id": str(rid), "error": f"raw parse error: {e}"})
                continue
            if not isinstance(obj, dict):
                rows.append({"id": str(rid), "error": "raw not a JSON object"})
                continue
            flat = flatten_record(obj, yes_no_keys)
            flat["id"] = str(rid)
            rows.append(flat)
            for k in flat.keys():
                if k not in all_keys:
                    all_keys.append(k)
    # Ensure 'id' first, 'error' second if present
    if "id" in all_keys:
        all_keys.remove("id")
    all_keys = ["id"] + (["error"] if "error" in all_keys else []) + [k for k in all_keys if k not in {"id","error"}]
    return rows, all_keys

def write_csv(rows: List[Dict[str, str]], header: List[str], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to *.jsonl file with {'id', 'raw'} lines" )
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--yesno", nargs="*", default=YES_NO_KEYS_DEFAULT, help="Keys to coerce to yes/no")
    args = p.parse_args()

    rows, header = parse_jsonl_to_rows(args.input, args.yesno)
    write_csv(rows, header, args.output)
    print(f"Wrote {len(rows)} rows -> {args.output}")

if __name__ == "__main__":
    main()
