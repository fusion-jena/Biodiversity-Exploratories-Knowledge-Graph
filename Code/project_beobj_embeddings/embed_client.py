import argparse, time, sys
import numpy as np
import pandas as pd
import requests

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise SystemExit(f"CSV must contain '{args.id_col}' and '{args.text_col}'")

    ids = df[args.id_col].astype(str).tolist()
    texts = df[args.text_col].astype(str).tolist()

    all_vecs = []
    sess = requests.Session()
    url = f"{args.api_base.rstrip('/')}/v1/embeddings"

    for chunk_ids, chunk_texts in zip(batched(ids, args.batch_size), batched(texts, args.batch_size)):
        payload = {"model": args.model, "input": chunk_texts}
        for attempt in range(5):
            try:
                r = sess.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                vecs = [d["embedding"] for d in data["data"]]
                all_vecs.extend(vecs)
                break
            except Exception as e:
                wait = 2**attempt
                print(f"[embed] retry {attempt+1}/5 after error: {e} (sleep {wait}s)", file=sys.stderr)
                time.sleep(wait)
        else:
            raise SystemExit("Failed to embed after retries")

    arr = np.array(all_vecs, dtype=np.float32)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=arr)
    print(f"Wrote {args.out}: shape={arr.shape}", file=sys.stderr)

if __name__ == "__main__":
    main()
