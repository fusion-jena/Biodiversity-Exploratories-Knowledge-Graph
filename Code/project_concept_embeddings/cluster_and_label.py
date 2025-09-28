import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import hdbscan

def c_tf_idf(docs_per_topic, m, ngram_range=(1,3), min_df=5, top_k=10):
    """Compute c-TF-IDF for topic labeling.
    docs_per_topic: dict {topic_id: [texts...]}
    m: total number of documents
    Returns: dict {topic_id: [(term, score), ...]}
    """
    topics = sorted(docs_per_topic.keys())
    corpus = [" ".join(docs_per_topic[t]) if docs_per_topic[t] else "" for t in topics]
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(corpus)  # shape: (n_topics, n_terms)

    # Compute class-based TF (normalize rows by L1)
    tf = normalize(X, norm="l1", axis=1)  # per-topic term distribution
    # Compute IDF using number of topics
    # idf = log(n_topics / (1 + df_term))
    df = (X > 0).sum(axis=0).A1
    n_topics = len(topics)
    idf = np.log((n_topics) / (1 + df))
    ctfidf = tf.multiply(idf)

    terms = np.array(vectorizer.get_feature_names_out())
    top_terms = {}
    for i, t in enumerate(topics):
        row = ctfidf.getrow(i).toarray().ravel()
        if row.size == 0:
            top_terms[t] = []
            continue
        idx = np.argsort(-row)[:top_k]
        top_terms[t] = [(terms[j], float(row[j])) for j in idx]
    return top_terms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--embeddings", required=True, help=".npz with 'ids' and 'embeddings'")
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--clusters-out", required=True)
    ap.add_argument("--topics-out", required=True)
    args = ap.parse_args()

    data = pd.read_csv(args.input_csv)
    npz = np.load(args.embeddings, allow_pickle=False)
    ids = [str(x) for x in npz["ids"].tolist()]
    X = npz["embeddings"].astype(np.float32)

    # Align rows
    data = data.astype({ "id": str })
    assert len(data) == len(ids)
    assert list(data["id"]) == ids, "Order mismatch between CSV and embeddings"

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(X)  # -1 are outliers

    data["cluster_id"] = labels
    data["is_outlier"] = (labels == -1)

    # Build docs_per_topic for labeling (exclude outliers)
    docs_per_topic = {}
    for cid, grp in data[~data["is_outlier"]].groupby("cluster_id"):
        docs_per_topic[int(cid)] = grp["text"].astype(str).tolist()

    top_terms = c_tf_idf(docs_per_topic, m=len(data))

    # Save clusters file
    data.to_csv(args.clusters_out, index=False)

    # Save topics summary
    rows = []
    for cid, docs in docs_per_topic.items():
        terms = top_terms.get(cid, [])
        rows.append({
            "cluster_id": cid,
            "n_docs": len(docs),
            "top_terms": "; ".join([t for t, _ in terms])
        })
    topics_df = pd.DataFrame(rows).sort_values("n_docs", ascending=False)
    topics_df.to_csv(args.topics_out, index=False)

    # Quick summary
    n_clusters = len([c for c in set(labels) if c != -1])
    n_out = int((labels == -1).sum())
    print(f"Clusters: {n_clusters} (min_cluster_size={args.min_cluster_size}); Outliers: {n_out} / {len(labels)}")
    print(f"Wrote clusters -> {args.clusters_out}")
    print(f"Wrote topics   -> {args.topics_out}")

if __name__ == "__main__":
    main()
