import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV files")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="merged")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if args.id_col not in df.columns or args.text_col not in df.columns:
            raise SystemExit(f"File {p} missing columns '{args.id_col}' or '{args.text_col}'")
        df = df[[args.id_col, args.text_col]].copy()
        df["source"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.rename(columns={args.id_col: "id", args.text_col: "text"}, inplace=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows from {len(frames)} files.")

if __name__ == "__main__":
    main()
