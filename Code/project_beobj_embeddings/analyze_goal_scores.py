import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-embeddings", required=True)   # e.g., out_biodiv_goals/doc_embeddings.npz
    ap.add_argument("--goal-embeddings", required=True)  # e.g., out_biodiv_goals/goal_embeddings.npz
    ap.add_argument("--doc-csv", required=True)          # e.g., out_biodiv_goals/combined_inputs.csv (id,text,source)
    ap.add_argument("--goal-csv", required=True)         # e.g., out_biodiv_goals/goals.csv (id,label,text)
    ap.add_argument("--outdir", default="out_biodiv_goals/analysis")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--annot-thresholds", default="", help="comma-separated thresholds to annotate, e.g. 0.20,0.25")
    ap.add_argument("--annot-quantiles", default="",  help="comma-separated quantiles to annotate, e.g. 0.2,0.5,0.8")
    ap.add_argument("--hexbin", action="store_true", help="use hexbin instead of scatter for top1 vs margin")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load
    d = np.load(args.doc_embeddings, allow_pickle=False)
    g = np.load(args.goal_embeddings, allow_pickle=False)
    D = d["embeddings"].astype(np.float32)
    G = g["embeddings"].astype(np.float32)
    doc_ids = [str(x) for x in d["ids"].tolist()]
    goal_ids = [str(x) for x in g["ids"].tolist()]

    docs = pd.read_csv(args.doc_csv).astype({"id": str})
    goals = pd.read_csv(args.goal_csv).astype({"id": str})
    id2label = dict(zip(goals["id"], goals.get("label", goals["id"])))

    # Align doc order
    if list(docs["id"]) != doc_ids:
        docs = docs.set_index("id").reindex(doc_ids).reset_index().rename(columns={"index":"id"})

    # --- Similarities
    S = cosine_sim(D, G)                       # [N_docs, N_goals]
    top1 = S.max(axis=1)
    winners = S.argmax(axis=1)
    # second best
    top2 = np.partition(S, -2, axis=1)[:, -2]
    margin = top1 - top2

    # Save a tidy table (handy for further analysis)
    out_scores_csv = os.path.join(args.outdir, "doc_top_scores.csv")
    pd.DataFrame({
        "id": doc_ids,
        "winner_goal_id": [goal_ids[j] for j in winners],
        "winner_goal_label": [id2label.get(goal_ids[j], goal_ids[j]) for j in winners],
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "source": docs.get("source", pd.Series([""]*len(doc_ids))).fillna("")
    }).to_csv(out_scores_csv, index=False)

    # Convenience: thresholds & quantiles to annotate
    ann_th = [float(x) for x in args.annot_thresholds.split(",") if x.strip()]
    ann_q  = [float(x) for x in args.annot_quantiles.split(",") if x.strip()]

    # --- Plot 1: overall top-1 histogram
    plt.figure(figsize=(8,5))
    plt.hist(top1, bins=args.bins, alpha=0.9)
    for t in ann_th:
        plt.axvline(t, linestyle="--", linewidth=1)
    for q in ann_q:
        qcut = float(np.quantile(top1, q))
        plt.axvline(qcut, linestyle=":", linewidth=1)
    plt.xlabel("Top-1 similarity")
    plt.ylabel("Documents")
    plt.title("Distribution of top-1 similarity scores")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hist_top1.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, "hist_top1.pdf"), dpi=300)
    plt.close()

    # --- Plot 2: margin histogram
    plt.figure(figsize=(8,5))
    plt.hist(margin, bins=args.bins, alpha=0.9)
    plt.xlabel("Top-1 − Top-2 margin")
    plt.ylabel("Documents")
    plt.title("Distribution of confidence margin (disambiguation)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hist_margin.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, "hist_margin.pdf"), dpi=300)
    plt.close()

    # --- Plot 3: per-goal top-1 histograms (winner grouping)
    # Small multiples 2x2 (you have 4 goals)
    labels = [id2label.get(gid, gid) for gid in goal_ids]
    fig, axes = plt.subplots(2, 2, figsize=(10,7), sharex=True, sharey=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:len(goal_ids)]):
        mask = (winners == j)
        ax.hist(top1[mask], bins=args.bins, alpha=0.9)
        ax.set_title(labels[j], fontsize=10)
        ax.set_xlabel("Top-1 similarity"); ax.set_ylabel("Documents")
    fig.suptitle("Top-1 similarity by winning goal")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(args.outdir, "hist_top1_by_goal.png"), dpi=300)
    fig.savefig(os.path.join(args.outdir, "hist_top1_by_goal.pdf"), dpi=300)
    plt.close(fig)

    # --- Plot 4: coverage vs threshold from scores (not quantiles)
    cuts = np.linspace(0, 1, 501)
    coverage = (top1[:,None] >= cuts[None,:]).mean(axis=0)
    plt.figure(figsize=(8,5))
    plt.plot(cuts, coverage, linewidth=1.5)
    for t in ann_th:
        plt.axvline(t, linestyle="--", linewidth=1)
    plt.xlabel("Threshold on top-1 similarity")
    plt.ylabel("Coverage (fraction assigned)")
    plt.title("Coverage vs threshold (from score distribution)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coverage_vs_threshold.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, "coverage_vs_threshold.pdf"), dpi=300)
    plt.close()

    # --- Plot 5: top-1 vs margin (scatter or hexbin)
    plt.figure(figsize=(7.5,6))
    if args.hexbin:
        hb = plt.hexbin(top1, margin, gridsize=40, mincnt=1)
        plt.colorbar(hb, label="Docs")
    else:
        plt.plot(top1, margin, ".", markersize=2, alpha=0.5)
    for t in ann_th:
        plt.axvline(t, linestyle="--", linewidth=1)
    plt.xlabel("Top-1 similarity")
    plt.ylabel("Margin (top1 - top2)")
    plt.title("Confidence landscape: top-1 vs margin")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "top1_vs_margin.png"), dpi=300)
    plt.savefig(os.path.join(args.outdir, "top1_vs_margin.pdf"), dpi=300)
    plt.close()

    print("Wrote:")
    for f in ["hist_top1","hist_margin","hist_top1_by_goal",
              "coverage_vs_threshold","top1_vs_margin"]:
        print(" -", os.path.join(args.outdir, f+".png"))

if __name__ == "__main__":
    main()
