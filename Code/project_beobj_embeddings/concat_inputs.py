import argparse, pandas as pd, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files to merge (must contain id and merged)")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="merged")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        if args.id_col not in df.columns or args.text_col not in df.columns:
            raise SystemExit(f"{path} missing columns {args.id_col}/{args.text_col}")
        tmp = pd.DataFrame({
            "id": df[args.id_col].astype(str),
            "text": df[args.text_col].astype(str),
            "source": path
        })
        frames.append(tmp)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows.", file=sys.stderr)

if __name__ == "__main__":
    main()
