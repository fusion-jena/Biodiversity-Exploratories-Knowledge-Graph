import argparse, sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import requests

EN_STOP = set("""a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split())

DE_STOP = set("""aber alle als also am an ander andere anderes anderm anderen anderr andern anderes anderm anders auch auf aus
bei bin bis bist da dadurch dafür dagegen daher darum das dass de der den des dem die dies dieser dieses deren dessen dich dir
doch dort du durch ein eine einem einen einer eines er es etwas euer eure euren eurer eures für gegen gewesen habe haben hat
hatte hatten hier hin hinter ich ihr ihre ihren ihrer ihres im in ist ja jede jedem jeden jeder jedes jener jenes jetzt kann
kannst kein keine keinem keinen keiner keines man mehr mein meine meiner meines mich mir mit muss musst mussten nach nicht
noch nur ob oder ohne sehr seid sein seine seiner seines selbst sich sie sind so solche solchen solcher solches soll sollen
sollst sollt sondern sondern um und uns unser unsere unseren unserer unseres unter vom von vor wann war waren warst was weg
weil weiter wenn wer werde werden werdet wir wird wirst wo wurde wurden zu zum zur zwar zwischen""".split())

def c_tfidf_for_topics(docs_per_topic: Dict[int, List[str]], min_df: int = 3):
    topics = sorted(docs_per_topic.keys())
    corpus = [" ".join(docs_per_topic[t]) if docs_per_topic[t] else "" for t in topics]
    vec = CountVectorizer(ngram_range=(1,1), min_df=min_df, stop_words=list(EN_STOP | DE_STOP))
    X = vec.fit_transform(corpus)
    tf = normalize(X, norm="l1", axis=1)
    df = (X > 0).sum(axis=0).A1
    n_topics = len(topics)
    idf = np.log((n_topics) / (1 + df))
    ctfidf = tf.multiply(idf)
    terms = np.array(vec.get_feature_names_out())
    top = {}
    for i, t in enumerate(topics):
        row = ctfidf.getrow(i).toarray().ravel()
        if row.size == 0:
            top[t] = []
            continue
        idx = np.argsort(-row)
        top[t] = [(terms[j], float(row[j])) for j in idx]
    return top

def pick_single_keyword(ctfidf_terms: List[Tuple[str, float]]) -> Optional[str]:
    # Choose the first non-stopword, alphabetic token of length >= 3
    for term, _ in ctfidf_terms:
        w = term.strip()
        if len(w) >= 3 and w.isalpha() and (w.lower() not in EN_STOP) and (w.lower() not in DE_STOP):
            return w
    return ctfidf_terms[0][0] if ctfidf_terms else None

def join_terms_for_sentence(terms: List[str], max_terms: int = 6) -> str:
    terms = [t for t in terms if t and t.strip()][:max_terms]
    if not terms:
        return "This cluster groups related publications and datasets."
    if len(terms) == 1:
        return f"This cluster focuses on {terms[0]}."
    if len(terms) == 2:
        return f"This cluster focuses on {terms[0]} and {terms[1]}."
    return f"This cluster focuses on {', '.join(terms[:-1])}, and {terms[-1]}."

def call_llm(api_base: str, model: str, prompt: str, max_tokens: int = 64, temperature: float = 0.2) -> str:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a scientific editor. Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters-csv", required=True)   # needs: id, text, cluster_id, is_outlier (optional)
    ap.add_argument("--topics-csv", required=False)    # optional; may contain top_terms
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--out", required=True)

    # Optional LLM to refine labels
    ap.add_argument("--llm-api-base", default="")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--sentence", action="store_true", help="Generate/refine a 1-sentence description per cluster")
    ap.add_argument("--keyword", action="store_true", help="Generate/refine a single-keyword label per cluster")
    args = ap.parse_args()

    df = pd.read_csv(args.clusters_csv)
    if not {"cluster_id", "text"}.issubset(df.columns):
        raise SystemExit("clusters.csv must contain cluster_id and text columns")

    if "is_outlier" in df.columns:
        df = df[~df["is_outlier"]]

    docs_per = {int(cid): grp["text"].astype(str).tolist() for cid, grp in df.groupby("cluster_id")}
    ctfidf = c_tfidf_for_topics(docs_per, min_df=args.min_df)

    seed_terms: Dict[int, List[str]] = {}
    if args.topics_csv:
        tdf = pd.read_csv(args.topics_csv)
        if {"cluster_id", "top_terms"}.issubset(tdf.columns):
            for _, row in tdf.iterrows():
                cid = int(row["cluster_id"])
                terms = [t.strip() for t in str(row["top_terms"]).split(";") if t.strip()]
                seed_terms[cid] = terms

    rows = []
    for cid in sorted(docs_per.keys()):
        kw = pick_single_keyword(ctfidf.get(cid, [])[:50])

        terms = seed_terms.get(cid, [t for t, _ in ctfidf.get(cid, [])[:6]])[:6]
        if kw and kw not in terms:
            terms = [kw] + terms

        sentence = join_terms_for_sentence(terms, max_terms=6)

        if args.llm_api_base and args.llm_model:
            if args.keyword:
                prompt = f"Return ONE single-word keyword (German or English) that best names this scientific topic. Terms: {', '.join(terms)}. Respond with ONE word only."
                try:
                    kw_resp = call_llm(args.llm_api_base, args.llm_model, prompt, max_tokens=6, temperature=0.0)
                    kw_resp = kw_resp.split()[0].strip().strip('.,;:!"\'()[]')
                    if kw_resp:
                        kw = kw_resp
                except Exception as e:
                    print(f"[warn] LLM keyword failed for cluster {cid}: {e}", file=sys.stderr)
            if args.sentence:
                prompt = ("Write ONE sentence (<=15 words) describing a research topic using these terms: "
                          + ", ".join(terms) + ". Keep it general, neutral, and scientific.")
                try:
                    sentence_resp = call_llm(args.llm_api_base, args.llm_model, prompt, max_tokens=64, temperature=0.2)
                    if sentence_resp:
                        sentence = sentence_resp.strip().replace('\n', ' ').strip()
                except Exception as e:
                    print(f"[warn] LLM sentence failed for cluster {cid}: {e}", file=sys.stderr)

        n_docs = len(docs_per[cid])
        rows.append({"cluster_id": cid, "n_docs": n_docs, "keyword": kw if kw else "", "sentence": sentence})

    out = pd.DataFrame(rows).sort_values("n_docs", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows.")

if __name__ == "__main__":
    main()
