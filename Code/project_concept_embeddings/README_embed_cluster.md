# Embedding + Clustering Pipeline (SLURM)

This package serves a multilingual embedding model (default: `BAAI/bge-m3`) via a local
OpenAI-compatible server (vLLM), embeds your **datasets** and **publications** CSVs,
then clusters and labels topics with HDBSCAN + c‑TF‑IDF.

## Files
- `serve_embed_cluster.slurm` — SLURM job that (optionally) starts vLLM, then runs the pipeline.
- `concat_inputs.py` — merges your CSVs into one (`id`, `text`, `source`).
- `embed_client.py` — calls `/v1/embeddings` in batches and saves `embeddings.npz`.
- `cluster_and_label.py` — HDBSCAN clustering + c‑TF‑IDF topic terms.

## Assumptions
- Each CSV has columns: `id`, `merged` (`"Title: ...\nAbstract: ..."`).
- You have a Conda env with `vllm`, `pandas`, `numpy`, `scikit-learn`, `hdbscan`, `requests` installed.
  ```bash
  conda create -n mistral3 python=3.10 -y
  conda activate mistral3
  pip install vllm "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.2" "hdbscan>=0.8.33" requests
  ```

## Run
Edit paths or export env vars, then submit:
```bash
export DATASETS_CSV=/mnt/data/datasets_merged.csv
export PUBLICATIONS_CSV=/mnt/data/publications_merged.csv
# Optional: use a remote server instead of serving locally
# export SERVE_LOCALLY=0
# export REMOTE_BASE_URL=http://gpu006.cluster:8000
# export EMBED_MODEL=BAAI/bge-m3
# export MIN_CLUSTER_SIZE=20

sbatch /mnt/data/serve_embed_cluster.slurm
```

Artifacts land in `out_emb_cluster/` by default:
- `combined_inputs.csv`
- `embeddings.npz`
- `clusters.csv` (id, source, text, cluster_id, is_outlier)
- `topics.csv` (cluster_id, n_docs, top_terms)

## Notes
- To run **CPU-only**, set `--gres=gpu:0` and consider embedding *without* vLLM by replacing `embed_client.py` with a local `sentence-transformers` script; GPU is recommended for speed.
- If your cluster already exposes an OpenAI-compatible embeddings server, set `SERVE_LOCALLY=0` and `REMOTE_BASE_URL` to skip starting vLLM.
