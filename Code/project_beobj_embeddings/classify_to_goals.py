import argparse, sys
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Stopwords (EN + DE + light domain)
EN_STOP = set("""a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split())
DE_STOP = set("""aber alle als also am an ander andere anderes anderm anderen anderr andern auch auf aus bei bin bis bist da
dadurch dafür dagegen daher darum das dass de der den des dem die dies dieser dieses deren dessen dich dir doch dort du
durch ein eine einem einen einer eines er es etwas euer eure euren eurer eures für gegen gewesen habe haben hat hatte
hatten hier hin hinter ich ihr ihre ihren ihrer ihres im in ist ja jede jedem jeden jeder jedes jener jenes jetzt kann
kannst kein keine keinem keinen keiner keines man mehr mein meine meiner meines mich mir mit muss musst mussten nach
nicht noch nur ob oder ohne sehr seid sein seine seiner seines selbst sich sie sind so solche solchen solcher solches
soll sollen sollst sollt sondern um und uns unser unsere unseren unserer unseres unter vom von vor wann war waren
warst was weg weil weiter wenn wer werde werden werdet wir wird wirst wo wurde wurden zu zum zur zwar zwischen""".split())
DOMAIN_STOP = set("study studies dataset datasets data result results analysis analyses method methods model models effect effects impact impacts evidence significant related associated using based approach approaches".split())
TOKEN_PATTERN = r"(?u)\b[^\W\d_][\w\-'’]{2,}\b"

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T  # [n_docs, n_goals]

def c_tf_idf(texts_per_key: Dict[str, List[str]], ngram_max=3, min_df=5, top_k=12, stop=None):
    keys = list(texts_per_key.keys())
    corpus = [" ".join(texts_per_key[k]).strip() if texts_per_key[k] else "" for k in keys]
    n_docs = sum(1 for s in corpus if s)
    if n_docs == 0:
        return {k: [] for k in keys}
    eff_min_df = max(1, min(min_df, n_docs))  # cap to avoid sklearn error with few groups

    vec = CountVectorizer(
        ngram_range=(1, ngram_max),
        min_df=eff_min_df,
        stop_words=(list(stop) if stop is not None else None),  # ensure list, not set
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
    )
    X = vec.fit_transform(corpus)
    if X.shape[1] == 0:
        return {k: [] for k in keys}
    tf = normalize(X, norm="l1", axis=1)
    df = (X > 0).sum(axis=0).A1
    K = n_docs
    idf = np.log(K / (1 + df))
    ctfidf = tf.multiply(idf)
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for i, k in enumerate(keys):
        row = ctfidf.getrow(i).toarray().ravel()
        idx = np.argsort(-row)[:top_k]
        out[k] = [(terms[j], float(row[j])) for j in idx]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-embeddings", required=True)
    ap.add_argument("--doc-csv", required=True)           # needs id,text
    ap.add_argument("--goal-embeddings", required=True)
    ap.add_argument("--goal-csv", required=True)          # needs id,label,text

    ap.add_argument("--topk", type=int, default=1)        # hard cap if you want fixed-K output
    ap.add_argument("--threshold", type=float, default=0.25)

    # Adaptive thresholding (optional)
    ap.add_argument("--adaptive", default="per-anchor-quantile",
                    choices=["none", "global-quantile", "per-anchor-quantile"])
    ap.add_argument("--quantile", type=float, default=0.80)

    # Acceptance via margin (OR-rule with threshold)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Accept if (top1 - top2) >= this value (even if below threshold/adaptive cut).")
    ap.add_argument("--margin-min-top1", type=float, default=0.0,
                    help="Require top1 >= this value for the margin rule to apply.")

    # NEW: Multi-label when ambiguous (margin is small)
    ap.add_argument("--multi-margin", type=float, default=0.0,
                    help="If (top1 - top2) < this value AND top1 >= multi-min-top1, output multiple labels.")
    ap.add_argument("--multi-min-top1", type=float, default=0.0,
                    help="Require top1 >= this value before allowing multi-label.")
    ap.add_argument("--max-multi", type=int, default=2,
                    help="Maximum number of labels to output when multi-labeling triggers (usually 2).")

    # Topic labeling
    ap.add_argument("--ngram-max", type=int, default=3)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--top-k-terms", type=int, default=12)
    ap.add_argument("--stopwords-lang", default="en,de")
    ap.add_argument("--extra-stopwords", default="")

    ap.add_argument("--assignments-out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--topics-out", required=True)
    args = ap.parse_args()

    # Load embeddings
    dnpz = np.load(args.doc_embeddings, allow_pickle=False)
    gnpz = np.load(args.goal_embeddings, allow_pickle=False)
    doc_ids = [str(x) for x in dnpz["ids"].tolist()]
    goal_ids = [str(x) for x in gnpz["ids"].tolist()]
    D = dnpz["embeddings"].astype(np.float32)
    G = gnpz["embeddings"].astype(np.float32)

    # Load CSVs
    docs = pd.read_csv(args.doc_csv).astype({"id": str})
    goals = pd.read_csv(args.goal_csv).astype({"id": str})
    if list(docs["id"]) != doc_ids:
        docs = docs.set_index("id").reindex(doc_ids).reset_index().rename(columns={"index":"id"})
    id_to_label = dict(zip(goals["id"], goals["label"]))

    # Similarity
    S = cosine_sim_matrix(D, G)  # [N_docs, N_goals]
    top1 = S.max(axis=1)
    winners = S.argmax(axis=1)

    # Adaptive thresholds
    global_cut = None
    per_goal_cut = None
    if args.adaptive == "global-quantile":
        global_cut = float(np.quantile(top1, args.quantile))
        print(f"[adaptive] global-quantile={args.quantile:.2f} -> cut={global_cut:.4f}", file=sys.stderr)
    elif args.adaptive == "per-anchor-quantile":
        per_goal_cut = {}
        for j in range(S.shape[1]):
            vals = top1[winners == j]
            per_goal_cut[j] = float(np.quantile(vals, args.quantile)) if vals.size >= 10 else args.threshold
        print(f"[adaptive] per-goal-quantile={args.quantile:.2f} set for {len(per_goal_cut)} goals", file=sys.stderr)

    rows = []
    Gdim = S.shape[1]
    need_k = max(args.topk, args.max_multi, 2)  # ensure we can see top2 for margins

    for i, doc_id in enumerate(doc_ids):
        # full sorted ranking for this doc (ensure at least top2)
        order = np.argsort(S[i])[::-1][:need_k]
        pairs = [(int(j), float(S[i, j])) for j in order]

        sc1 = pairs[0][1] if pairs else 0.0
        sc2 = pairs[1][1] if len(pairs) > 1 else 0.0
        margin = sc1 - sc2

        # acceptance threshold (adaptive + floor)
        accept = args.threshold
        if args.adaptive == "global-quantile" and global_cut is not None:
            accept = max(accept, global_cut)
        elif args.adaptive == "per-anchor-quantile" and per_goal_cut is not None and pairs:
            accept = max(accept, per_goal_cut.get(pairs[0][0], args.threshold))

        ok_by_threshold = (pairs and sc1 >= accept)
        ok_by_margin = (pairs and (margin >= args.min_margin) and (sc1 >= args.margin_min_top1))

        if ok_by_threshold or ok_by_margin:
            # decide how many labels to output
            n_out = min(args.topk, len(pairs)) if args.topk > 0 else 1
            if (args.multi_margin > 0.0) and (margin < args.multi_margin) and (sc1 >= args.multi_min_top1):
                n_out = min(max(n_out, 2), args.max_multi, len(pairs))

            out = {"id": doc_id, "margin": round(margin, 6), "n_labels": n_out}
            for rank, (j, sc) in enumerate(pairs[:n_out], start=1):
                gid = goal_ids[j]
                out[f"goal{rank}_id"] = gid
                out[f"goal{rank}_label"] = id_to_label.get(gid, gid)
                out[f"goal{rank}_score"] = round(sc, 6)
            rows.append(out)
        else:
            rows.append({"id": doc_id, "margin": round(margin, 6), "n_labels": 0,
                         "goal1_id": "", "goal1_label": "", "goal1_score": 0.0})

    assign = pd.DataFrame(rows)
    assign.to_csv(args.assignments_out, index=False)

    # Summary (still top-1 only, to avoid double-counting)
    top1_assign = assign[assign["goal1_id"].astype(str) != ""]
    summary = (top1_assign.groupby(["goal1_id", "goal1_label"])
               .size().reset_index(name="n_docs")
               .sort_values("n_docs", ascending=False))
    summary.to_csv(args.summary_out, index=False)

    # Topics per goal via c-TF-IDF (still based on top-1)
    texts_per: Dict[str, List[str]] = {}
    merged = docs.merge(top1_assign[["id","goal1_id","goal1_label"]], on="id", how="inner")
    for gid, grp in merged.groupby("goal1_id"):
        texts_per[str(gid)] = grp["text"].astype(str).tolist()

    # stopwords
    langs = {s.strip().lower() for s in args.stopwords_lang.split(",") if s.strip()}
    stop = set()
    if "en" in langs: stop |= EN_STOP
    if "de" in langs: stop |= DE_STOP
    stop |= DOMAIN_STOP
    if args.extra_stopwords:
        stop |= {w.strip().lower() for w in args.extra_stopwords.split(",") if w.strip()}

    ctfidf = c_tf_idf(texts_per, ngram_max=args.ngram_max, min_df=args.min_df, top_k=args.top_k_terms, stop=stop)
    rows2 = []
    for gid, terms in ctfidf.items():
        rows2.append({
            "goal_id": gid,
            "goal_label": dict(zip(goals["id"], goals["label"])).get(gid, gid),
            "n_docs": len(texts_per.get(gid, [])),
            "top_terms": "; ".join([t for t,_ in terms])
        })
    topics = pd.DataFrame(rows2).sort_values("n_docs", ascending=False)
    topics.to_csv(args.topics_out, index=False)

    print(f"Wrote {args.assignments_out}, {args.summary_out}, {args.topics_out}", file=sys.stderr)

if __name__ == "__main__":
    main()
