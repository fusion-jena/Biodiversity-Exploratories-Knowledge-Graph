import argparse, json, time, sys
from typing import List
import numpy as np
import pandas as pd
import requests

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def fetch_embeddings(api_base: str, model: str, texts: List[str]) -> List[List[float]]:
    url = f"{api_base.rstrip('/')}/v1/embeddings"
    payload = {"model": model, "input": texts}
    for attempt in range(6):
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                # OpenAI-compatible format: data: [{embedding: [...]}]
                return [item["embedding"] for item in data["data"]]
            else:
                print(f"[warn] {r.status_code}: {r.text[:2000]}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Exception: {e}", file=sys.stderr)
        time.sleep(2 + attempt)
    raise RuntimeError("Failed to fetch embeddings after retries.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True, help="Path to .npz file")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise SystemExit(f"Input CSV missing columns '{args.id_col}' or '{args.text_col}'")
    ids = df[args.id_col].astype(str).tolist()
    texts = df[args.text_col].astype(str).tolist()

    embs = []
    for chunk in batched(texts, args.batch_size):
        vecs = fetch_embeddings(args.api_base, args.model, chunk)
        embs.extend(vecs)
    arr = np.array(embs, dtype=np.float32)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=arr)
    print(f"Saved embeddings to {args.out} with shape {arr.shape}")

if __name__ == "__main__":
    main()