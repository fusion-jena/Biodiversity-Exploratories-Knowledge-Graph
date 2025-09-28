import argparse, sys, os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

# Stopwords and tokenization (EN+DE + light domain)
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

def choose_k(X_unit: np.ndarray, k: int=None, kmin:int=25, kmax:int=60) -> int:
    if k is not None and k > 1:
        return k
    best_k, best_score = None, -1.0
    # Try every K in [kmin, kmax]
    for kk in range(max(2,kmin), max(3,kmax+1)):
        cl = AgglomerativeClustering(n_clusters=kk, metric="cosine", linkage="average")
        labels = cl.fit_predict(X_unit)
        # Guard: need at least 2 labels
        if len(set(labels)) < 2: 
            continue
        sc = silhouette_score(X_unit, labels, metric="cosine")
        if sc > best_score:
            best_k, best_score = kk, sc
    if best_k is None:
        best_k = max(2, kmin)
    print(f"[cluster_anchors] auto-chosen K={best_k} (silhouette={best_score:.4f})", file=sys.stderr)
    return best_k

def ctfidf_terms(texts_per_cluster: Dict[int, List[str]], ngram_max=3, min_df=5, top_k=12):
    keys = sorted(texts_per_cluster.keys())
    corpus = [" ".join(texts_per_cluster[k]) if texts_per_cluster[k] else "" for k in keys]
    if not any(corpus): 
        return {k: [] for k in keys}
    vec = CountVectorizer(ngram_range=(1, ngram_max),
                          min_df=min_df,
                          stop_words=list(EN_STOP | DE_STOP | DOMAIN_STOP),
                          token_pattern=r"(?u)\b[^\W\d_][\w\-]{2,}\b",
                          lowercase=True)
    X = vec.fit_transform(corpus)  # (K,V)
    if X.shape[1] == 0:
        return {k: [] for k in keys}
    X = normalize(X, norm="l1", axis=1)
    df = (X > 0).sum(axis=0).A1
    K = len(keys)
    idf = np.log(K / (1 + df))
    ctfidf = X.multiply(idf)
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
    ap.add_argument("--anchor-embeddings", required=True, help=".npz with 'ids' + 'embeddings' for anchors")
    ap.add_argument("--anchor-csv", required=True, help="CSV with anchors (id/label/text)")
    ap.add_argument("--anchor-id-col", default="keyword")
    ap.add_argument("--anchor-label-col", default="keyword")
    ap.add_argument("--anchor-text-col", default="merged")

    # Clustering controls
    ap.add_argument("--n-super", type=int, default=0, help="Fixed number of super-anchors (if >0). Else auto in range.")
    ap.add_argument("--min-k", type=int, default=25)
    ap.add_argument("--max-k", type=int, default=60)

    # Outputs
    ap.add_argument("--out-super-emb", required=True, help=".npz of super-anchor centroids")
    ap.add_argument("--out-super-csv", required=True, help="CSV of super-anchors (labels, members, terms)")
    ap.add_argument("--out-mapping", required=True, help="CSV map: anchor_id -> super_id")

    # Labeling controls
    ap.add_argument("--ngram-max", type=int, default=3)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--top-k-terms", type=int, default=12)
    args = ap.parse_args()

    # Load embeddings
    npz = np.load(args.anchor_embeddings, allow_pickle=False)
    a_ids = [str(x) for x in npz["ids"].tolist()]
    A = npz["embeddings"].astype(np.float32)
    if A.shape[0] != len(a_ids):
        raise SystemExit("anchor embeddings malformed")

    # Normalize for cosine clustering
    A_unit = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    # Load CSV with labels & texts
    df = pd.read_csv(args.anchor_csv)
    if args.anchor_id_col not in df.columns or args.anchor_label_col not in df.columns or args.anchor_text_col not in df.columns:
        raise SystemExit("anchor CSV must include id/label/text columns")
    df[args.anchor_id_col] = df[args.anchor_id_col].astype(str)
    # Align to embeddings order
    meta = df.set_index(args.anchor_id_col).reindex(a_ids)
    
    if meta.isnull().any().any():
        missing = [i for i,x in zip(a_ids, meta.index.isnull()) if x]
        raise SystemExit(f"Anchor IDs in embeddings not found in CSV: {missing[:5]}...")

    # Decide K
    K = choose_k(A_unit, k=args.n_super if args.n_super>0 else None, kmin=args.min_k, kmax=args.max_k)

    # Cluster
    clustering = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average")
    labels = clustering.fit_predict(A_unit)  # 0..K-1

    # Build centroids
    centroids = np.zeros((K, A.shape[1]), dtype=np.float32)
    members: Dict[int, List[int]] = {k: [] for k in range(K)}
    for idx, c in enumerate(labels):
        members[c].append(idx)
    for c, idxs in members.items():
        centroids[c] = A_unit[idxs].mean(axis=0)
        # re-normalize centroid
        centroids[c] = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-12)

    # Representative anchor per cluster (closest to centroid)
    from numpy import dot
    rep_idx = {}
    for c, idxs in members.items():
        sims = [float(dot(A_unit[i], centroids[c])) for i in idxs]
        j = int(idxs[int(np.argmax(sims))])
        rep_idx[c] = j

    # Topic terms via c-TF-IDF over anchor texts (per cluster)
    texts_per = {c: meta.iloc[idxs][args.anchor_text_col].astype(str).tolist() for c, idxs in members.items()}
    ctfidf = ctfidf_terms(texts_per, ngram_max=args.ngram_max, min_df=args.min_df, top_k=args.top_k_terms)

    # Compose super-anchors table
    rows = []
    super_ids = []
    for c in range(K):
        sid = f"S{c:04d}"
        super_ids.append(sid)
        idxs = members[c]
        mem_anchor_ids = [a_ids[i] for i in idxs]
        mem_labels = meta.iloc[idxs][args.anchor_label_col].astype(str).tolist()

        rep = rep_idx[c]
        rep_id = a_ids[rep]
        rep_label = str(meta.iloc[rep][args.anchor_label_col])

        terms = [t for t,_ in ctfidf.get(c, [])]
        top_terms = "; ".join(terms)
        # Human-friendly super label: prefer representative anchor label; add 1-2 key terms for specificity
        extra = (", ".join(terms[:2])) if terms else ""
        super_label = rep_label if not extra else f"{rep_label} — {extra}"

        # Optional text field for downstream (some tools require it)
        super_text = (rep_label + " " + " ".join(terms)).strip()

        rows.append({
            "super_id": sid,
            "size": len(idxs),
            "super_label": super_label,
            "repr_anchor_id": rep_id,
            "repr_anchor_label": rep_label,
            "top_terms": top_terms,
            "member_ids": ";".join(mem_anchor_ids),
            "member_labels": ";".join(mem_labels),
            "super_text": super_text
        })

    super_df = pd.DataFrame(rows).sort_values("size", ascending=False)
    super_df.to_csv(args.out_super_csv, index=False)

    # Mapping file
    map_rows = [{"anchor_id": a_ids[i], "super_id": f"S{labels[i]:04d}"} for i in range(len(a_ids))]
    pd.DataFrame(map_rows).to_csv(args.out_mapping, index=False)

    # Save centroid embeddings
    np.savez_compressed(args.out_super_emb, ids=np.array(super_ids), embeddings=centroids.astype(np.float32))

    print(f"Wrote {args.out_super_csv}, {args.out_mapping}, {args.out_super_emb}")

if __name__ == "__main__":
    main()
