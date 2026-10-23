"""Stronger encoders and a larger re-ranking pool: every embedder x retrieval mode x re-ranker x pool size.

Chunks are the notebook's sliding windows, so the saved gold chunk ids apply, and the first configuration
(MiniLM-L6 | dense | MiniLM-L6 | 10) reproduces the report's Table 2. Each embedder ranks the chunks once per
question (dense, and hybrid with the same BM25 and RRF as rag_core); each re-ranker scores every candidate any
configuration needs once, so the whole grid costs one pass per model. Meant for a GPU (Kaggle); also times each
re-ranker on CPU threads, which is what the deployed app has.

  RAG_ENCODERS=torch python eval_models.py           # GDPR (stress + hunt) and the Constitution (study + hunt)
Writes notebook/results/models_eval.json.
"""
import json
import os
import statistics
import time

os.environ.setdefault("RAG_ENCODERS", "torch")   # before rag_core is imported: everything here runs on torch

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from eval_hybrid import UPLOADS, metrics, question_sets, upload_sets  # noqa: E402
from rag_core import HYBRID_POOL, PDF_PATH, RESULTS_DIR, TOP_K_FINAL, DocIndex, GDPR_SUBJECT, Models, rrf  # noqa: E402

# name: (model id, query prefix, passage prefix), following each model card
EMBEDDERS = {
    "MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
}
RERANKERS = {
    "MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MiniLM-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-base": "BAAI/bge-reranker-base",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}
POOLS = (10, 20)
MODES = ("dense", "hybrid")
CPU_THREADS, CPU_QUERIES = 4, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def documents(models):
    gdpr = DocIndex(models, PDF_PATH, title="GDPR", subject=GDPR_SUBJECT, profile="gdpr")
    title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
    const = DocIndex(models, UPLOADS / "a914ab3a3b.pdf", title=title["a914ab3a3b"])
    return {"gdpr": (gdpr, sum(question_sets(), [])), "constitution": (const, sum(upload_sets("a914ab3a3b"), []))}


def first_stage(idx, qs, name):
    """{mode: [top-HYBRID_POOL chunk ids per question]} for one embedder."""
    model_id, qp, pp = EMBEDDERS[name]
    st = SentenceTransformer(model_id, device=DEVICE)
    emb = st.encode([pp + t for t in idx.search_texts], batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True)
    qv = st.encode([qp + q["query"] for q in qs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    out = {"dense": [], "hybrid": []}
    for q, v in zip(qs, qv):
        dense = np.argsort(-(emb @ v), kind="stable")[:HYBRID_POOL].tolist()
        bm, _ = idx.bm25.search(q["query"], HYBRID_POOL)
        out["dense"].append(dense)
        out["hybrid"].append(rrf(dense, bm)[0][:HYBRID_POOL])
    del st
    torch.cuda.empty_cache()
    return out


def rerank_scores(idx, qs, name, needed):
    """{(question index, chunk id): score} for every pair in needed, plus GPU and CPU timings for 10 pairs."""
    kw = {"model_kwargs": {"torch_dtype": torch.float16}} if name == "bge-v2-m3" and DEVICE == "cuda" else {}
    ce = CrossEncoder(RERANKERS[name], device=DEVICE, max_length=512, **kw)
    pairs = sorted(needed)
    scores = ce.predict([(qs[i]["query"], idx.search_texts[c]) for i, c in pairs], batch_size=64)
    out = dict(zip(pairs, map(float, scores)))
    timing = {}
    sample = [(q["query"], [idx.search_texts[c] for c in range(10)]) for q in qs[:20]]
    for device in ([DEVICE, "cpu"] if DEVICE == "cuda" else ["cpu"]):
        if device == "cpu":
            torch.set_num_threads(CPU_THREADS)
            ce = CrossEncoder(RERANKERS[name], device="cpu", max_length=512)
        runs = sample[:CPU_QUERIES] if device == "cpu" else sample
        ce.predict([(runs[0][0], t) for t in runs[0][1]])   # warm-up
        t = []
        for query, texts in runs:
            t0 = time.perf_counter()
            ce.predict([(query, x) for x in texts], batch_size=10)
            t.append((time.perf_counter() - t0) * 1000)
        timing["gpuMs" if device != "cpu" else "cpuMs"] = round(statistics.mean(t), 1)
    del ce
    torch.cuda.empty_cache()
    return out, timing


def main():
    models = Models()
    result = {"device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
              "cpuThreads": CPU_THREADS, "rerankers": {}, "docs": {}}
    for doc, (idx, qs) in documents(models).items():
        print(f"== {doc}: {len(idx.chunks)} chunks, {len(qs)} questions", flush=True)
        stage1 = {e: first_stage(idx, qs, e) for e in EMBEDDERS}
        needed = {(i, c) for e in stage1 for m in MODES for i, ids in enumerate(stage1[e][m]) for c in ids}
        sets = sorted({q["set"] for q in qs}, key=lambda s: s == "hunt")
        res = {s: {} for s in sets}
        for r in RERANKERS:
            print(f"   re-ranker {r}: {len(needed)} pairs", flush=True)
            scores, timing = rerank_scores(idx, qs, r, needed)
            result["rerankers"].setdefault(r, timing)
            for e in EMBEDDERS:
                for m in MODES:
                    for k in POOLS:
                        s1, rr = [], []
                        for i, q in enumerate(qs):
                            cands = stage1[e][m][i][:k]
                            order = sorted(cands, key=lambda c: -scores[(i, c)])
                            gold = set(q["gold"])
                            s1.append(next((r1 for r1, c in enumerate(cands, 1) if c in gold), None))
                            rr.append(next((r1 for r1, c in enumerate(order, 1) if c in gold), None))
                        for s in sets:
                            pick = [j for j, q in enumerate(qs) if q["set"] == s]
                            res[s][f"{e} | {m} | {r} | {k}"] = {
                                "stage1": metrics([s1[j] for j in pick]), "reranked": metrics([rr[j] for j in pick])}
        result["docs"][doc] = {"chunks": len(idx.chunks), "questions": {s: sum(q["set"] == s for q in qs) for s in sets},
                               **res}
        out = RESULTS_DIR / "models_eval.json"
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")   # after each document, in case of a crash
    best = sorted(result["docs"]["gdpr"]["hunt"].items(), key=lambda kv: -kv[1]["reranked"]["hit@3"])[:8]
    for key, v in best:
        print(f"{key:45} hit@3 {v['reranked']['hit@3']:.2f} mrr {v['reranked']['mrr@10']:.2f}")
    print(json.dumps(result["rerankers"], indent=1))
    print(f"top {TOP_K_FINAL} reach the LLM")


if __name__ == "__main__":
    main()
