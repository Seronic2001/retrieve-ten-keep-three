"""Which re-ranker should the app run on CPU? Quality and latency of ONNX re-rankers behind hybrid retrieval.

eval_models.py ranked the re-rankers on a GPU: bge-reranker-v2-m3 is the most accurate, but takes ~10 s per
question on 4 CPU threads, and the app has only a CPU. This runs each candidate as the app would (ONNX Runtime,
onnx_encoders.CrossEncoder), including int8-quantized exports, over the same 10 hybrid candidates per question.

Questions: GDPR stress + hunt, and the Constitution study + hunt (its hunt with the hand-corrected gold, see
eval_clean.correct_hunt). Latency is the median time to score one question's 10 candidates, with every CPU thread
and with 2 threads (a free Hugging Face Space).

  ../.venv/Scripts/python.exe eval_rerank_cpu.py
Writes notebook/results/rerank_cpu_eval.json.
"""
import json
import statistics
import time

import numpy as np

from eval_hybrid import UPLOADS, metrics, question_sets, upload_sets
from onnx_encoders import CrossEncoder, _cpus
from rag_core import RESULTS_DIR, DocIndex, Models, gdpr_index

# name: (repo, ONNX file)
RERANKERS = {
    "MiniLM-L6": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"),
    "MiniLM-L6-int8": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "onnx/model_quint8_avx2.onnx"),
    "MiniLM-L12": ("cross-encoder/ms-marco-MiniLM-L-12-v2", "onnx/model.onnx"),
    "MiniLM-L12-int8": ("cross-encoder/ms-marco-MiniLM-L-12-v2", "onnx/model_quint8_avx2.onnx"),
    "bge-v2-m3-int8": ("onnx-community/bge-reranker-v2-m3-ONNX", "onnx/model_int8.onnx"),
}
FEW_THREADS, LATENCY_QUESTIONS = 2, 20


def timed_scores(ce, items):
    """Each question's candidate scores, and the median time to score one question in ms."""
    scores, times = [], []
    for it in items:
        t0 = time.perf_counter()
        scores.append(ce.predict(it[4], batch_size=len(it[4])))
        times.append(time.perf_counter() - t0)
    return scores, round(statistics.median(times) * 1000)


def main():
    models = Models()
    title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
    stress, hunt = question_sets()
    docs = {"gdpr": (gdpr_index(models, cache=UPLOADS / "gdpr.emb.npz"), stress + hunt),
            "constitution": (DocIndex(models, UPLOADS / "a914ab3a3b.pdf", title=title["a914ab3a3b"],
                                      cache=UPLOADS / "a914ab3a3b.emb.npz"), sum(upload_sets("a914ab3a3b"), []))}
    # the hybrid candidates, once: (doc, set, gold, [chunk ids], [(query, text)])
    items = []
    for doc, (idx, qs) in docs.items():
        for q in qs:
            cands, _ = idx.retrieve_and_rerank(q["query"], "hybrid")
            ids = [c["chunk"] for c in sorted(cands, key=lambda c: c["biRank"])]
            items.append((doc, q["set"], set(q["gold"]), ids, [(q["query"], idx.search_texts[i]) for i in ids]))
    print(f"{len(items)} questions, {_cpus()} CPU threads", flush=True)

    groups = list(dict.fromkeys(it[:2] for it in items))
    by_group = lambda ranks: {f"{d}/{s}": metrics([r for r, it in zip(ranks, items) if it[:2] == (d, s)])   # noqa: E731
                              for d, s in groups}
    first_gold = lambda order, gold: next((r for r, c in enumerate(order, 1) if c in gold), None)   # noqa: E731
    summary = {"cpuThreads": _cpus(), "questions": {f"{d}/{s}": sum(it[:2] == (d, s) for it in items) for d, s in groups},
               "fused": by_group([first_gold(it[3], it[2]) for it in items]), "rerankers": {}}
    for name, (repo, file) in RERANKERS.items():
        ce, few = CrossEncoder(repo, file=file), CrossEncoder(repo, file=file, threads=FEW_THREADS)
        ce.predict(items[0][4])   # warm-up
        few.predict(items[0][4])
        scores, ms = timed_scores(ce, items)
        ranks = [first_gold([it[3][j] for j in np.argsort(-sc)], it[2]) for sc, it in zip(scores, items)]
        res = {"msPerQuestion": ms, f"msPerQuestion{FEW_THREADS}Threads": timed_scores(few, items[:LATENCY_QUESTIONS])[1],
               **by_group(ranks)}
        summary["rerankers"][name] = res
        print(name, json.dumps(res), flush=True)
        del ce, few
    (RESULTS_DIR / "rerank_cpu_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
