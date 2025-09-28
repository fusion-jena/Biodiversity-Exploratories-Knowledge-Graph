import argparse, sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# --- Stopwords (EN + DE) ---
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

# Optional domain stopwords that often pollute topic labels
DOMAIN_STOP = set("""
study studies dataset datasets data result results analysis analyses method methods approach approaches using model models
effect effects impact impacts evidence significant related associated
""".split())

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Row-normalize then dot
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T  # [n_docs, n_anchors]

def c_tf_idf(texts_per_key: Dict[str, List[str]], ngram_max=3, min_df=5,
             top_k=10, stop_words=None, token_pattern=r"(?u)\b[^\W\d_][\w\-]{2,}\b"):
    """
    Class-based TF-IDF labeling with stopword removal.
    token_pattern: starts with a letter; length >=3; allows hyphens; excludes pure digits/underscores.
    """
    keys = list(texts_per_key.keys())
    corpus = [" ".join(texts_per_key[k]) if texts_per_key[k] else "" for k in keys]
    if not any(corpus):
        return {k: [] for k in keys}
    vec = CountVectorizer(ngram_range=(1, ngram_max),
                          min_df=min_df,
                          stop_words=stop_words,
                          token_pattern=token_pattern,
                          lowercase=True)
    X = vec.fit_transform(corpus)  # (K, V)
    if X.shape[1] == 0:
        return {k: [] for k in keys}
    tf = normalize(X, norm="l1", axis=1)
    df = (X > 0).sum(axis=0).A1
    K = len(keys)
    idf = np.log(K / (1 + df))
    ctfidf = tf.multiply(idf)
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for i, k in enumerate(keys):
        row = ctfidf.getrow(i).toarray().ravel()
        if row.size == 0:
            out[k] = []
            continue
        idx = np.argsort(-row)[:top_k]
        out[k] = [(terms[j], float(row[j])) for j in idx]
    return out

def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--doc-embeddings", required=True, help=".npz with ids, embeddings for documents")
    ap.add_argument("--doc-csv", required=True, help="CSV used to embed docs (needs id,text)")
    ap.add_argument("--anchor-embeddings", required=True, help=".npz with ids, embeddings for anchors")
    ap.add_argument("--anchor-csv", required=True, help="CSV for anchors")
    ap.add_argument("--anchor-id-col", default="keyword", help="Column in anchor CSV used as ID (must match ids in anchor-embeddings)")
    ap.add_argument("--anchor-label-col", default="keyword", help="Human-readable label column for anchors")
    ap.add_argument("--anchor-text-col", default="merged", help="Text column (for diagnostics/topics)")

    # Assignment params
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.25)

    # NEW: Adaptive thresholding
    ap.add_argument("--adaptive", default="none",
                    choices=["none", "global-quantile", "per-anchor-quantile"],
                    help="Adaptive acceptance threshold strategy")
    ap.add_argument("--quantile", type=float, default=0.80,
                    help="Quantile used for adaptive thresholds (e.g., 0.8)")

    # Outputs
    ap.add_argument("--assignments-out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--topics-out", required=True)

    # Topic labeling params
    ap.add_argument("--ngram-max", type=int, default=3)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--top-k-terms", type=int, default=10)

    # Stopword controls
    ap.add_argument("--stopwords-lang", default="en,de",
                    help="comma-separated langs to include: en,de (default en,de)")
    ap.add_argument("--extra-stopwords", default="",
                    help="comma-separated extra stopwords to remove (case-insensitive)")
    ap.add_argument("--token-pattern", default=r"(?u)\b[^\W\d_][\w\-]{2,}\b",
                    help="regex for tokenization; default keeps letters>=3 and hyphens")

    args = ap.parse_args()

    # Load embeddings
    d_npz = np.load(args.doc_embeddings, allow_pickle=False)
    a_npz = np.load(args.anchor_embeddings, allow_pickle=False)
    doc_ids = [str(x) for x in d_npz["ids"].tolist()]
    anc_ids = [str(x) for x in a_npz["ids"].tolist()]
    D = d_npz["embeddings"].astype(np.float32)
    A = a_npz["embeddings"].astype(np.float32)

    # Load CSVs
    docs_df = pd.read_csv(args.doc_csv).astype({"id": str})
    if "text" not in docs_df.columns:
        raise SystemExit("doc CSV must contain 'id' and 'text'")
    anchors_df = pd.read_csv(args.anchor_csv)
    if args.anchor_id_col not in anchors_df.columns:
        raise SystemExit(f"anchor CSV missing id-col '{args.anchor_id_col}'")
    if args.anchor_label_col not in anchors_df.columns:
        raise SystemExit(f"anchor CSV missing label-col '{args.anchor_label_col}'")
    if args.anchor_text_col not in anchors_df.columns:
        raise SystemExit(f"anchor CSV missing text-col '{args.anchor_text_col}'")

    # Align document order
    if list(docs_df["id"]) != doc_ids:
        docs_df = docs_df.set_index("id").reindex(doc_ids).reset_index().rename(columns={"index": "id"})

    # Map anchor ids to labels
    anchors_df[args.anchor_id_col] = anchors_df[args.anchor_id_col].astype(str)
    id_to_label = dict(zip(anchors_df[args.anchor_id_col], anchors_df[args.anchor_label_col]))

    # Similarity
    if D.size == 0 or A.size == 0:
        raise SystemExit("Empty embeddings provided")
    S = cosine_sim_matrix(D, A)  # [N_docs, N_anchors]

    # --- Adaptive thresholds ---
    top1_scores = S.max(axis=1)           # top-1 score for each doc
    winners = S.argmax(axis=1)            # index of winning anchor for each doc

    global_cut = None
    per_anchor_cut = None

    if args.adaptive == "global-quantile":
        global_cut = float(np.quantile(top1_scores, args.quantile))
        print(f"[adaptive] global-quantile={args.quantile:.2f} -> cut={global_cut:.4f}", file=sys.stderr)

    elif args.adaptive == "per-anchor-quantile":
        per_anchor_cut = {}
        for j in range(S.shape[1]):
            vals = top1_scores[winners == j]
            if vals.size >= 20:  # need enough support; otherwise fall back to base threshold
                per_anchor_cut[j] = float(np.quantile(vals, args.quantile))
            else:
                per_anchor_cut[j] = args.threshold
        print(f"[adaptive] per-anchor-quantile={args.quantile:.2f} set for {len(per_anchor_cut)} anchors", file=sys.stderr)

    # Top-k selection (per doc)
    topk = max(1, args.topk)
    ksel = min(topk, S.shape[1])
    top_idx = np.argpartition(-S, kth=ksel-1, axis=1)[:, :ksel]

    rows = []
    for i, doc_id in enumerate(doc_ids):
        pairs = [(j, float(S[i, j])) for j in top_idx[i]]
        pairs.sort(key=lambda t: -t[1])

        # Determine acceptance cutoff for this doc/anchor
        accept_cut = args.threshold
        if args.adaptive == "global-quantile" and global_cut is not None:
            accept_cut = max(args.threshold, global_cut)
        elif args.adaptive == "per-anchor-quantile" and per_anchor_cut is not None and pairs:
            j0 = pairs[0][0]  # winning anchor index
            accept_cut = max(args.threshold, per_anchor_cut.get(j0, args.threshold))

        if pairs and pairs[0][1] >= accept_cut:
            out = {"id": doc_id}
            for k, (j, sc) in enumerate(pairs, start=1):
                anc_id = anc_ids[j]
                out[f"anchor{k}_id"] = anc_id
                out[f"anchor{k}_label"] = id_to_label.get(anc_id, anc_id)
                out[f"anchor{k}_score"] = round(sc, 6)
            rows.append(out)
        else:
            rows.append({"id": doc_id, "anchor1_id": "", "anchor1_label": "", "anchor1_score": 0.0})

    assign_df = pd.DataFrame(rows)
    assign_df.to_csv(args.assignments_out, index=False)

    # Summary by anchor1
    top1 = assign_df[assign_df["anchor1_id"].astype(str) != ""]
    summary = (top1.groupby(["anchor1_id", "anchor1_label"])
                    .size().reset_index(name="n_docs")
                    .sort_values("n_docs", ascending=False))
    summary.to_csv(args.summary_out, index=False)

    # --- Topics per anchor via c-TF-IDF (anchor1 only) with stopwords ---
    texts_per_anchor: Dict[str, List[str]] = {}
    merged = docs_df.merge(top1[["id", "anchor1_id", "anchor1_label"]], on="id", how="inner")
    for (aid, _), grp in merged.groupby(["anchor1_id", "anchor1_label"]):
        texts_per_anchor[aid] = grp["text"].astype(str).tolist()

    # Compose stopwords
    langs = {s.strip().lower() for s in args.stopwords_lang.split(",") if s.strip()}
    stop = set()
    if "en" in langs: stop |= EN_STOP
    if "de" in langs: stop |= DE_STOP
    stop |= DOMAIN_STOP
    if args.extra_stopwords:
        stop |= {w.strip().lower() for w in args.extra_stopwords.split(",") if w.strip()}

    ctfidf = c_tf_idf(
        texts_per_anchor,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        top_k=args.top_k_terms,
        stop_words=list(stop),
        token_pattern=args.token_pattern
    )

    rows2 = []
    for aid, terms in ctfidf.items():
        rows2.append({
            "anchor_id": aid,
            "n_docs": len(texts_per_anchor.get(aid, [])),
            "top_terms": "; ".join([t for t, _ in terms])
        })
    topics_df = pd.DataFrame(rows2).sort_values("n_docs", ascending=False)
    topics_df["anchor_label"] = topics_df["anchor_id"].map(lambda x: id_to_label.get(x, x))
    topics_df = topics_df[["anchor_id", "anchor_label", "n_docs", "top_terms"]]
    topics_df.to_csv(args.topics_out, index=False)

    print(f"Wrote {args.assignments_out}, {args.summary_out}, {args.topics_out}")

if __name__ == "__main__":
    main()
