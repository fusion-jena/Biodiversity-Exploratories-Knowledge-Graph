#!/usr/bin/env python3
"""
Evaluate metadata predictions vs. ground truth with per-field metrics.

Supported field types:
- yes/no (binary): accuracy, precision, recall, F1, confusion matrix
- multilabel (lists of short phrases): micro/macro P/R/F1, subset accuracy, Jaccard
- fuzzy-list (same as multilabel, but matches phrases with normalized Levenshtein >= threshold)
- numeric: exact match rate, MAE, RMSE
- freetext (long strings): token F1, ROUGE-L (LCS-based)

Assumptions:
- Both CSVs share a common ID column (default 'id').
- Multi-valued cells are separated by ';' or ',' (parser handles both).

Example:
  python eval_metadata.py     --pred out/publications_predictions.csv     --gold data/publications_gold.csv     --out_dir eval_pub     --yesno-fields alb sch hai grassland forest aboveground belowground field laboratory review rex1 rex2 lux fox     --multilabel-fields plot_level keywords biotic_data_taxon biotic_data_type processes_and_services environmental_descriptors project content_type     --fuzzy-list-fields sampling_design sampling_preparation sample_analysis equipment data_preparation data_analysis experimental_manipulation temporal_repetitions plot_repetitions     --numeric-fields number_of_gp number_of_ep number_of_mip number_of_vip

"""
import argparse, csv, math, os, re
from typing import Dict, List, Tuple, Set, Any

# -------------------- parsing & normalization helpers --------------------

IGNORED_TOKENS = {"", "n/a", "na", "none", "null"}
YES_SET = {"yes","y","true","1"}
NO_SET  = {"no","n","false","0"}

def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def split_multivalue(cell: str) -> List[str]:
    if cell is None:
        return []
    cell = str(cell).strip()
    if not cell:
        return []
    # split on ; or , but keep slashes
    parts = re.split(r"\s*[;,]\s*", cell)
    vals = [norm_text(p) for p in parts if norm_text(p) not in IGNORED_TOKENS]
    # dedup preserve order (case-insensitive already normalized)
    seen = set(); out = []
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def parse_yesno(cell: str) -> str:
    if cell is None:
        return ""
    s = norm_text(str(cell))
    if s in YES_SET: return "yes"
    if s in NO_SET:  return "no"
    # if it's a list-like string, try first token
    parts = split_multivalue(s)
    if parts:
        if parts[0] in YES_SET: return "yes"
        if parts[0] in NO_SET:  return "no"
    return s  # unknown -> return normalized raw

def levenshtein(a: str, b: str) -> int:
    # classic DP
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        ca = a[i-1]
        for j in range(1, m+1):
            cb = b[j-1]
            tmp = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def sim_ratio(a: str, b: str) -> float:
    # normalized similarity in [0,1]
    a = norm_text(a); b = norm_text(b)
    denom = max(len(a), len(b), 1)
    return 1.0 - levenshtein(a, b) / denom

def greedy_fuzzy_match(pred: List[str], gold: List[str], thr: float=0.85) -> Tuple[Set[int], Set[int], List[float]]:
    """Greedy one-to-one matching between predicted and gold phrases by similarity."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            pairs.append((sim_ratio(p, g), i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    sims = []
    for s, i, j in pairs:
        if s < thr: break
        if i in used_p or j in used_g: continue
        used_p.add(i); used_g.add(j); sims.append(s)
    return used_p, used_g, sims

def rouge_l_scores(pred: str, gold: str) -> Tuple[float, float, float]:
    """ROUGE-L over whitespace tokens via LCS."""
    def tokens(x): return [t for t in re.split(r"\s+", norm_text(x)) if t]
    p_toks, g_toks = tokens(pred), tokens(gold)
    n, m = len(p_toks), len(g_toks)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if p_toks[i-1] == g_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0
    prec = lcs / n
    rec  = lcs / m
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return prec, rec, f1

def prf1(tp, fp, fn) -> Tuple[float,float,float]:
    p = 0.0 if (tp+fp)==0 else tp/(tp+fp)
    r = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    f1 = 0.0 if (p+r)==0 else 2*p*r/(p+r)
    return p,r,f1

def load_csv(path: str, id_col: str) -> Dict[str, Dict[str, str]]:
    data: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get(id_col, "")).strip()
            if not rid:
                continue
            data[rid] = {k: (row[k] if row[k] is not None else "") for k in reader.fieldnames}
    return data

def eval_yesno(field: str, preds: Dict[str, Dict[str,str]], golds: Dict[str, Dict[str,str]], ids: List[str]) -> Dict[str, Any]:
    tp=fp=tn=fn=0
    for rid in ids:
        p = parse_yesno(preds.get(rid,{}).get(field, ""))
        g = parse_yesno(golds.get(rid,{}).get(field, ""))
        if g not in {"yes","no"}:
            continue
        if p not in {"yes","no"}:
            if g == "yes": fn += 1
            else: tn += 1
            continue
        if p == "yes" and g == "yes": tp += 1
        elif p == "yes" and g == "no": fp += 1
        elif p == "no"  and g == "yes": fn += 1
        else: tn += 1
    denom = (tp+tn+fp+fn) or 1
    acc = (tp+tn)/denom
    p,r,f1 = prf1(tp,fp,fn)
    return {"field":field,"type":"yesno","support":denom,"accuracy":acc,"precision":p,"recall":r,"f1":f1,"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def eval_multilabel(field: str, preds: Dict[str, Dict[str,str]], golds: Dict[str, Dict[str,str]], ids: List[str]) -> Dict[str, Any]:
    micro_tp=micro_fp=micro_fn=0
    per_sample_jaccard = []
    subset_ok = 0
    for rid in ids:
        pset = set(split_multivalue(preds.get(rid,{}).get(field,"")))
        gset = set(split_multivalue(golds.get(rid,{}).get(field,"")))
        if not pset and not gset:
            subset_ok += 1
            per_sample_jaccard.append(1.0)
            continue
        micro_tp += len(pset & gset)
        micro_fp += len(pset - gset)
        micro_fn += len(gset - pset)
        if pset == gset: subset_ok += 1
        denom = len(pset | gset) or 1
        per_sample_jaccard.append(len(pset & gset)/denom)
    p,r,f1 = prf1(micro_tp, micro_fp, micro_fn)
    subset_acc = subset_ok/len(ids) if ids else 0.0
    avg_j = sum(per_sample_jaccard)/len(per_sample_jaccard) if per_sample_jaccard else 0.0
    return {"field":field,"type":"multilabel","support":len(ids),"micro_precision":p,"micro_recall":r,"micro_f1":f1,"subset_accuracy":subset_acc,"avg_jaccard":avg_j}

def eval_fuzzy_list(field: str, preds: Dict[str, Dict[str,str]], golds: Dict[str, Dict[str,str]], ids: List[str], thr: float) -> Dict[str, Any]:
    micro_tp=micro_fp=micro_fn=0
    per_sample_soft_jaccard = []
    sims_all = []
    for rid in ids:
        pl = split_multivalue(preds.get(rid,{}).get(field,""))
        gl = split_multivalue(golds.get(rid,{}).get(field,""))
        # Greedy matching
        pairs = []
        for i, p in enumerate(pl):
            for j, g in enumerate(gl):
                pairs.append((sim_ratio(p, g), i, j))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        sims = []
        for s, i, j in pairs:
            if s < thr: break
            if i in used_p or j in used_g: continue
            used_p.add(i); used_g.add(j); sims.append(s)
        tp = len(used_p)
        fp = len(pl) - tp
        fn = len(gl) - tp
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        denom = len(set(range(len(pl))) | set(range(len(gl)))) or 1
        per_sample_soft_jaccard.append(tp/denom)
        sims_all.extend(sims)
    p,r,f1 = prf1(micro_tp, micro_fp, micro_fn)
    avg_soft_j = sum(per_sample_soft_jaccard)/len(per_sample_soft_jaccard) if per_sample_soft_jaccard else 0.0
    avg_sim = sum(sims_all)/len(sims_all) if sims_all else 0.0
    return {"field":field,"type":"fuzzy-list","support":len(ids),"micro_precision":p,"micro_recall":r,"micro_f1":f1,"avg_soft_jaccard":avg_soft_j,"avg_match_sim":avg_sim,"threshold":thr}

def eval_numeric(field: str, preds: Dict[str, Dict[str,str]], golds: Dict[str, Dict[str,str]], ids: List[str]) -> Dict[str, Any]:
    xs=[]; ys=[]; exact=0; total=0
    for rid in ids:
        p = preds.get(rid,{}).get(field, "").strip()
        g = golds.get(rid,{}).get(field, "").strip()
        if p=="" and g=="":
            continue
        try:
            pv = int(float(p))
            gv = int(float(g))
        except:
            continue
        xs.append(pv); ys.append(gv); total += 1
        if pv == gv: exact += 1
    if not xs:
        return {"field":field,"type":"numeric","support":0,"exact_match":0.0,"mae":0.0,"rmse":0.0}
    mae = sum(abs(a-b) for a,b in zip(xs,ys))/len(xs)
    rmse = (sum((a-b)**2 for a,b in zip(xs,ys))/len(xs))**0.5
    exact_rate = exact/total if total else 0.0
    return {"field":field,"type":"numeric","support":total,"exact_match":exact_rate,"mae":mae,"rmse":rmse}

def eval_freetext(field: str, preds: Dict[str, Dict[str,str]], golds: Dict[str, Dict[str,str]], ids: List[str]) -> Dict[str, Any]:
    f1s=[]; precs=[]; recs=[]
    for rid in ids:
        p = preds.get(rid,{}).get(field,"")
        g = golds.get(rid,{}).get(field,"")
        if not p and not g: 
            continue
        # ROUGE-L token-based
        def tokens(x): return [t for t in re.split(r"\s+", norm_text(x)) if t]
        p_toks, g_toks = tokens(p), tokens(g)
        n, m = len(p_toks), len(g_toks)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                if p_toks[i-1] == g_toks[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[n][m]
        if n == 0 or m == 0:
            pr, rc, f1 = 0.0, 0.0, 0.0
        else:
            pr = lcs / n
            rc = lcs / m
            f1 = 0.0 if (pr+rc)==0 else 2*pr*rc/(pr+rc)
        f1s.append(f1); precs.append(pr); recs.append(rc)
    if not f1s:
        return {"field":field,"type":"freetext","support":0,"rougeL_f1":0.0,"rougeL_prec":0.0,"rougeL_rec":0.0}
    return {"field":field,"type":"freetext","support":len(f1s),"rougeL_f1":sum(f1s)/len(f1s),"rougeL_prec":sum(precs)/len(precs),"rougeL_rec":sum(recs)/len(recs)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions CSV path")
    ap.add_argument("--gold", required=True, help="Ground-truth CSV path")
    ap.add_argument("--out_dir", required=True, help="Output directory for reports")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--yesno-fields", nargs="*", default=[])
    ap.add_argument("--multilabel-fields", nargs="*", default=[])
    ap.add_argument("--fuzzy-list-fields", nargs="*", default=[])
    ap.add_argument("--numeric-fields", nargs="*", default=[])
    ap.add_argument("--freetext-fields", nargs="*", default=[])
    ap.add_argument("--fuzzy-threshold", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preds = load_csv(args.pred, args.id_col)
    golds = load_csv(args.gold, args.id_col)
    ids = sorted(set(preds.keys()) & set(golds.keys()))

    summaries: List[Dict[str, Any]] = []

    # yes/no
    for f in args.yesno_fields:
        res = eval_yesno(f, preds, golds, ids)
        summaries.append(res)
        conf_path = os.path.join(args.out_dir, f"confusion_{f}.csv")
        with open(conf_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf); w.writerow(["", "gold_yes", "gold_no"])
            w.writerow(["pred_yes", res["tp"], res["fp"]])
            w.writerow(["pred_no",  res["fn"], res["tn"]])

    # multilabel
    for f in args.multilabel_fields:
        res = eval_multilabel(f, preds, golds, ids)
        summaries.append(res)

    # fuzzy-list
    for f in args.fuzzy_list_fields:
        res = eval_fuzzy_list(f, preds, golds, ids, args.fuzzy_threshold)
        summaries.append(res)

    # numeric
    for f in args.numeric_fields:
        res = eval_numeric(f, preds, golds, ids)
        summaries.append(res)

    # freetext
    for f in args.freetext_fields:
        res = eval_freetext(f, preds, golds, ids)
        summaries.append(res)

    # write summary
    # Determine all keys
    keys = set()
    for d in summaries:
        for k in d.keys(): keys.add(k)
    header = sorted(keys)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for d in summaries:
            w.writerow(d)

    print(f"Wrote per-field summary to {os.path.join(args.out_dir, 'summary.csv')}")
    print("Fields evaluated:")
    for d in summaries:
        print(f" - {d['field']}: type={d['type']}")

if __name__ == "__main__":
    main()
