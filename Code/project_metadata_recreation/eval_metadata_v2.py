#!/usr/bin/env python3
import argparse, csv, math, os, re
from typing import Dict, List, Tuple, Set, Any

YESNO_DEFAULT = [
    "alb","sch","hai","grassland","forest","aboveground","belowground",
    "field","laboratory","review","rex1","rex2","lux","fox"
]
MULTILABEL_CANDIDATES = [
    "plot_level","keywords","biotic_data_taxon","biotic_data_type",
    "processes_and_services","environmental_descriptors","project","content_type","infrastructure"
]
FUZZY_LIST_CANDIDATES = [
    "sampling_design","sampling_preparation","sample_analysis","equipment",
    "data_preparation","data_analysis","experimental_manipulation","temporal_repetitions","plot_repetitions","study_design"
]
NUMERIC_CANDIDATES = ["number_of_gp","number_of_ep","number_of_mip","number_of_vip"]

def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def split_multivalue(cell: str):
    if cell is None: return []
    cell = str(cell).strip()
    if not cell: return []
    parts = re.split(r"\s*[;,]\s*", cell)
    vals = [norm_text(p) for p in parts if norm_text(p)]
    seen=set(); out=[]
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def parse_yesno(cell: str):
    if cell is None: return ""
    s = norm_text(str(cell))
    if s in {"yes","y","true","1"}: return "yes"
    if s in {"no","n","false","0"}: return "no"
    parts = split_multivalue(s)
    if parts:
        if parts[0] in {"yes","y","true","1"}: return "yes"
        if parts[0] in {"no","n","false","0"}: return "no"
    return s

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]; dp[0] = i
        ca = a[i-1]
        for j in range(1, m+1):
            cb = b[j-1]
            tmp = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def sim_ratio(a: str, b: str) -> float:
    a = norm_text(a); b = norm_text(b)
    denom = max(len(a), len(b), 1)
    return 1.0 - levenshtein(a, b) / denom

def greedy_fuzzy_match(pred, gold, thr=0.85):
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            pairs.append((sim_ratio(p, g), i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for s, i, j in pairs:
        if s < thr: break
        if i in used_p or j in used_g: continue
        used_p.add(i); used_g.add(j)
    return used_p, used_g

def prf1(tp, fp, fn):
    p = 0.0 if (tp+fp)==0 else tp/(tp+fp)
    r = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    f1 = 0.0 if (p+r)==0 else 2*p*r/(p+r)
    return p,r,f1

def autodetect_id_col(cols):
    for c in ["id","Id","ID"]:
        if c in cols: return c
    return cols[0]

def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [{k:(v if v is not None else "") for k,v in row.items()} for row in r]
        return r.fieldnames, rows

def apply_renames(rows, mapping):
    if not mapping: return rows
    out = []
    for row in rows:
        nr = {}
        for k,v in row.items():
            nr[mapping.get(k, k)] = v
        out.append(nr)
    return out

def parse_renames(arg: str):
    """Parse 'old:new,old2:new2' into a dict."""
    m = {}
    if not arg: return m
    for part in arg.split(","):
        part = part.strip()
        if not part: continue
        if ":" not in part: 
            continue
        old, new = part.split(":", 1)
        m[old.strip()] = new.strip()
    return m

def evaluate(pred_path, gold_path, out_dir, yesno_fields, multilabel_fields, fuzzy_fields, numeric_fields, fuzzy_thr=0.85, rename_pred=None, rename_gold=None):
    os.makedirs(out_dir, exist_ok=True)
    pred_cols, pred_rows = load_csv(pred_path)
    gold_cols, gold_rows = load_csv(gold_path)

    pred_rows = apply_renames(pred_rows, parse_renames(rename_pred or ""))
    gold_rows = apply_renames(gold_rows, parse_renames(rename_gold or ""))
    if rename_pred or rename_gold:
        pred_cols = sorted({k for r in pred_rows for k in r.keys()})
        gold_cols = sorted({k for r in gold_rows for k in r.keys()})

    pid = autodetect_id_col(pred_cols)
    gid = autodetect_id_col(gold_cols)

    pmap = {str(r.get(pid,"")).strip(): r for r in pred_rows if str(r.get(pid,"")).strip()}
    gmap = {str(r.get(gid,"")).strip(): r for r in gold_rows if str(r.get(gid,"")).strip()}
    ids = sorted(set(pmap.keys()) & set(gmap.keys()))

    overlap = sorted((set(pred_cols) - {pid}) & (set(gold_cols) - {gid}))
    if not any([yesno_fields, multilabel_fields, fuzzy_fields, numeric_fields]):
        yesno_fields = [f for f in YESNO_DEFAULT if f in overlap]
        numeric_fields = [f for f in NUMERIC_CANDIDATES if f in overlap]
        fuzzy_fields = [f for f in FUZZY_LIST_CANDIDATES if f in overlap]
        multilabel_fields = [f for f in overlap if f not in set(yesno_fields+numeric_fields+fuzzy_fields)]
    else:
        yesno_fields     = [f for f in yesno_fields     if f in overlap]
        multilabel_fields= [f for f in multilabel_fields if f in overlap]
        fuzzy_fields     = [f for f in fuzzy_fields     if f in overlap]
        numeric_fields   = [f for f in numeric_fields   if f in overlap]

    def write_confusion(field, tp, fp, fn, tn):
        with open(os.path.join(out_dir, f"confusion_{field}.csv"), "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf); w.writerow(["", "gold_yes", "gold_no"])
            w.writerow(["pred_yes", tp, fp])
            w.writerow(["pred_no",  fn, tn])

    summary = []

    # YES/NO
    for f in yesno_fields:
        tp=fp=tn=fn=0
        for rid in ids:
            p = parse_yesno(pmap[rid].get(f, ""))
            g = parse_yesno(gmap[rid].get(f, ""))
            if g not in {"yes","no"}: continue
            if p not in {"yes","no"}:
                if g=="yes": fn += 1
                else: tn += 1
                continue
            if p=="yes" and g=="yes": tp+=1
            elif p=="yes" and g=="no": fp+=1
            elif p=="no"  and g=="yes": fn+=1
            else: tn+=1
        denom = (tp+tn+fp+fn) or 1
        acc = (tp+tn)/denom
        p,r,f1 = prf1(tp,fp,fn)
        summary.append({"field":f,"type":"yesno","support":denom,"accuracy":acc,"precision":p,"recall":r,"f1":f1})
        write_confusion(f,tp,fp,fn,tn)

    # NUMERIC
    for f in numeric_fields:
        xs=[]; ys=[]; exact=0; total=0
        for rid in ids:
            p = str(pmap[rid].get(f, "")).strip()
            g = str(gmap[rid].get(f, "")).strip()
            if not p and not g: continue
            try:
                pv = int(float(p)); gv = int(float(g))
            except:
                continue
            xs.append(pv); ys.append(gv); total+=1
            if pv==gv: exact+=1
        if total==0:
            summary.append({"field":f,"type":"numeric","support":0,"exact_match":0.0,"mae":0.0,"rmse":0.0})
        else:
            mae = sum(abs(a-b) for a,b in zip(xs,ys))/total
            rmse = (sum((a-b)**2 for a,b in zip(xs,ys))/total)**0.5
            summary.append({"field":f,"type":"numeric","support":total,"exact_match":exact/total,"mae":mae,"rmse":rmse})

    # MULTILABEL strict
    def eval_multilabel_field(f):
        micro_tp=micro_fp=micro_fn=0
        subset_ok=0; j_list=[]
        for rid in ids:
            pset = set(split_multivalue(pmap[rid].get(f, "")))
            gset = set(split_multivalue(gmap[rid].get(f, "")))
            if not pset and not gset:
                subset_ok += 1; j_list.append(1.0); continue
            micro_tp += len(pset & gset)
            micro_fp += len(pset - gset)
            micro_fn += len(gset - pset)
            if pset == gset: subset_ok += 1
            denom = len(pset | gset) or 1
            j_list.append(len(pset & gset)/denom)
        p,r,f1 = prf1(micro_tp, micro_fp, micro_fn)
        subset_acc = subset_ok/len(ids) if ids else 0.0
        avg_j = sum(j_list)/len(j_list) if j_list else 0.0
        return {"field":f,"type":"multilabel","support":len(ids),"micro_precision":p,"micro_recall":r,"micro_f1":f1,"subset_accuracy":subset_acc,"avg_jaccard":avg_j}

    for f in multilabel_fields:
        summary.append(eval_multilabel_field(f))

    # FUZZY-LIST
    for f in fuzzy_fields:
        micro_tp=micro_fp=micro_fn=0
        soft_js=[]
        for rid in ids:
            pl = split_multivalue(pmap[rid].get(f, ""))
            gl = split_multivalue(gmap[rid].get(f, ""))
            mp, mg = greedy_fuzzy_match(pl, gl, thr=fuzzy_thr)
            tp = len(mp); fp = len(pl)-tp; fn = len(gl)-tp
            micro_tp += tp; micro_fp += fp; micro_fn += fn
            denom = len(set(range(len(pl))) | set(range(len(gl)))) or 1
            soft_js.append(tp/denom)
        p,r,f1 = prf1(micro_tp, micro_fp, micro_fn)
        avg_soft = sum(soft_js)/len(soft_js) if soft_js else 0.0
        summary.append({"field":f,"type":"fuzzy-list","threshold":fuzzy_thr,"support":len(ids),"micro_precision":p,"micro_recall":r,"micro_f1":f1,"avg_soft_jaccard":avg_soft})

    header = sorted({k for d in summary for k in d.keys()})
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for d in summary: w.writerow(d)

    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Aligned IDs: %d\n" % len(ids))
        f.write("Yes/No fields: %s\n" % ", ".join(yesno_fields))
        f.write("Numeric fields: %s\n" % ", ".join(numeric_fields))
        f.write("Multilabel fields: %s\n" % ", ".join(multilabel_fields))
        f.write("Fuzzy-list fields: %s\n" % ", ".join(fuzzy_fields))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yesno-fields", nargs="*", default=[])
    ap.add_argument("--multilabel-fields", nargs="*", default=[])
    ap.add_argument("--fuzzy-list-fields", nargs="*", default=[])
    ap.add_argument("--numeric-fields", nargs="*", default=[])
    ap.add_argument("--fuzzy-threshold", type=float, default=0.85)
    ap.add_argument("--rename-pred", default="", help="Comma-separated 'old:new' pairs to rename columns in predictions")
    ap.add_argument("--rename-gold", default="", help="Comma-separated 'old:new' pairs to rename columns in gold")
    args = ap.parse_args()

    evaluate(
        pred_path=args.pred,
        gold_path=args.gold,
        out_dir=args.out_dir,
        yesno_fields=args.yesno_fields,
        multilabel_fields=args.multilabel_fields,
        fuzzy_fields=args.fuzzy_list_fields,
        numeric_fields=args.numeric_fields,
        fuzzy_thr=args.fuzzy_threshold,
        rename_pred=args.rename_pred,
        rename_gold=args.rename_gold,
    )

if __name__ == "__main__":
    main()
