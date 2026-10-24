"""Exports retrieval results to ui/public/data.json for the presentation app.

Re-runs only the deterministic retrieval steps through rag_core (same settings as rag_rerank.ipynb).
LLM answers and timings are read from notebook/results/, so no API calls are made.
Run from notebook/ (it reads rag_rerank.ipynb there):  ../.venv/Scripts/python.exe ../app/export_ui_data.py
"""
import ast
import json
import re

import pandas as pd

from rag_core import (CHUNK_WORDS, DEVICE, OVERLAP_WORDS, RESULTS_DIR, ROOT, Pipeline)

RESULTS = RESULTS_DIR
OUT = ROOT / "ui" / "public" / "data.json"

rag = Pipeline()


def norm(s):
    s = s.lower().translate(str.maketrans("‘’“”", "''\"\""))
    return re.sub(r"\s+", " ", s)


NT = [norm(t) for t in rag.texts]

# Query list and gold snippets, read from the notebook's CANDIDATE_QUERIES cell.
nb = json.load(open("rag_rerank.ipynb", encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"] if "CANDIDATE_QUERIES = [" in "".join(c["source"]))
queries = ast.literal_eval(src[src.index("["):src.index("\n]\n") + 2])

# The latency figures come from the notebook run, so label them with the device that run printed,
# not the device this export happens to use.
nb_text = "".join(o.get("text", "") if isinstance(o.get("text"), str) else "".join(o.get("text", []))
                  for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
NOTEBOOK_DEVICE = (re.search(r"Device: (\w+)", nb_text) or [None, DEVICE])[1]

hunt = pd.read_csv(RESULTS / "failure_hunt.csv")
answers = pd.read_csv(RESULTS / "answers.csv").set_index("query")

out_queries = []
for cat, q, snippets in queries:
    gold = sorted(i for i, t in enumerate(NT) if any(norm(s) in t for s in snippets))
    candidates, _ = rag.retrieve_and_rerank(q)
    for c in candidates:
        c["gold"] = c["chunk"] in gold
    row = hunt[hunt["query"] == q].iloc[0]
    b, r = row["baseline_rank"], row["reranked_rank"]
    status = ("restored" if row["restored"] else "degraded" if row["degraded"]
              else "both-miss" if pd.isna(b) else "both-hit")
    item = {"category": cat, "query": q, "goldSnippets": snippets, "gold": gold,
            "baselineRank": None if pd.isna(b) else int(b),
            "rerankedRank": None if pd.isna(r) else int(r),
            "status": status, "candidates": candidates}
    if q in answers.index:
        a = answers.loc[q]
        item["answers"] = {"baseline": a["answer_1_baseline"], "reranked": a["answer_2_reranked"]}
    out_queries.append(item)

# Three consecutive chunks (around the Art. 83 fines) to illustrate the overlapping windows.
mid = next(i for i, t in enumerate(NT) if "20 000 000 eur" in t)
example = [{"chunk": c["id"], "page": c["page"], "cite": c["cite"], "text": c["text"]}
           for c in rag.chunks[mid - 1:mid + 2]]



def experiments():
    """The follow-up experiments on both test documents (GDPR, Constitution of India), from their saved results:
    re-rankers behind hybrid retrieval on a CPU (eval_rerank_cpu.py) and judged answer quality with the corrected
    gold (eval_clean.py, "fixed" stage)."""
    cpu = json.loads((RESULTS / "rerank_cpu_eval.json").read_text(encoding="utf-8"))
    answers = json.loads((RESULTS / "clean_eval.json").read_text(encoding="utf-8"))["answers"]["fixed"]

    def hit3(res, doc):   # pooled over the document's question sets
        sets = [k for k in cpu["questions"] if k.startswith(f"{doc}/")]
        return round(sum(res[k]["hit@3"] * res[k]["n"] for k in sets) / sum(res[k]["n"] for k in sets), 2)

    shown = {"MiniLM-L6": "MiniLM-L6", "MiniLM-L12-int8": "MiniLM-L12 (int8)", "bge-v2-m3-int8": "bge-reranker-v2-m3 (int8)"}
    rerankers = [{"name": "None (fused order)", "ms": 0, "ms2": 0,
                  "gdpr": hit3(cpu["fused"], "gdpr"), "constitution": hit3(cpu["fused"], "constitution")}]
    rerankers += [{"name": label, "ms": cpu["rerankers"][k]["msPerQuestion"], "ms2": cpu["rerankers"][k]["msPerQuestion2Threads"],
                   "gdpr": hit3(cpu["rerankers"][k], "gdpr"), "constitution": hit3(cpu["rerankers"][k], "constitution")}
                  for k, label in shown.items()]
    return {"questions": {d: sum(n for k, n in cpu["questions"].items() if k.startswith(f"{d}/")) for d in ("gdpr", "constitution")},
            "cpuThreads": cpu["cpuThreads"], "rerankers": rerankers,
            "answers": [{"doc": key.split("/")[0], "pipeline": key.split("/")[1],
                         **{k: v[k] for k in ("n", "correct", "partial", "wrong", "abstained", "goldInTop3")}}
                        for key, v in answers.items()]}


lat = pd.read_csv(RESULTS / "latency_stages.csv")
scaling = pd.read_csv(RESULTS / "scaling.csv")
data = {
    "document": {"title": "General Data Protection Regulation (EU) 2016/679",
                 "source": "Official Journal of the European Union, L 119, 4.5.2016",
                 "pages": len(rag.pages), "chunks": len(rag.chunks), "firstArticleChunk": rag.first_article,
                 "chunkWords": CHUNK_WORDS, "overlapWords": OVERLAP_WORDS},
    "chunkIndex": [{"page": c["page"], "cite": c["cite"]} for c in rag.chunks],
    "chunkingExample": example,
    "models": {"bi": "all-MiniLM-L6-v2", "cross": "ms-marco-MiniLM-L-6-v2",
               "llm": "gemini-3.7-flash", "device": NOTEBOOK_DEVICE},
    "queries": out_queries,
    "latency": {"retrieveMs": float(lat.mean_ms[0]), "rerankMs": float(lat.mean_ms[1]),
                "llmMs": float(lat.mean_ms[2])},
    "scaling": {"corpus": int(scaling.corpus_size_chunks[0]),
                "fullScanMs": float(scaling.cross_encoder_full_scan_s[0] * 1000),
                "perPairMs": float(scaling.cross_encoder_full_scan_s[0] * 1000 / scaling.corpus_size_chunks[0])},
    # From the question-generation sweep (612 Gemini-written questions, one source chunk each).
    "sweep": {"questions": 612, "restored": 93, "degraded": 11},
    "experiments": experiments(),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), {len(out_queries)} queries")
for q in out_queries:
    print(f"  {q['status']:9} {q['query'][:60]:60} gold: {[c['cite'] for c in q['candidates'] if c['gold']]}")
