import argparse, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-emb", required=True)
    ap.add_argument("--goal-emb", required=True)
    ap.add_argument("--quantiles", default="0,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1")
    ap.add_argument("--floors", default="0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26")
    ap.add_argument("--out", default="coverage_sweep.csv")
    args = ap.parse_args()

    d=np.load(args.doc_emb); g=np.load(args.goal_emb)
    D=d["embeddings"].astype(np.float32); G=g["embeddings"].astype(np.float32)
    D/=np.linalg.norm(D,axis=1,keepdims=True)+1e-12
    G/=np.linalg.norm(G,axis=1,keepdims=True)+1e-12
    S = D @ G.T
    top1 = S.max(axis=1); top2 = np.partition(S, -2, axis=1)[:,-2]
    margin = top1 - top2

    quantiles = [float(x) for x in args.quantiles.split(",")]
    floors = [float(x) for x in args.floors.split(",")]

    rows=[]
    for q in quantiles:
        qcut = float(np.quantile(top1, q))
        for f in floors:
            accept = np.maximum(qcut, f)
            cov = float((top1 >= accept).mean())
            rows.append({"quantile": q, "floor": f, "cut": accept, "coverage": cov})
    # margin-only acceptance examples
    for m in [0.03,0.05,0.07]:
        cov = float((margin >= m).mean())
        rows.append({"quantile":"-", "floor":"-", "cut": f"margin>={m}", "coverage": cov})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
