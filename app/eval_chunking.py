"""Sliding windows vs structure-aware chunks, with dense and hybrid retrieval, each followed by the cross-encoder.

Gold chunk ids belong to one chunking, so gold is located in the text instead: each question's evidence snippet
is found inside its saved gold chunks (window chunking) and kept as a word span in the document. Under any
chunking, a chunk is gold when it holds at least half of such a span. This is the notebook's rule ("the chunk
contains the snippet") made independent of where the chunks are cut. Questions whose snippet cannot be found
are left out and counted.

Variants:
  window              the notebook's 220-word windows with 40-word overlap (the current pipeline)
  structure           one provision per chunk, packed or split at paragraphs up to 220 words (structure_chunks.py)
  structure+heading   the same chunks, embedded and re-ranked under their provision headings

  ../.venv/Scripts/python.exe eval_chunking.py                  # the GDPR
  ../.venv/Scripts/python.exe eval_chunking.py --doc a914ab3a3b # an upload (its case study and failure hunt)
  RAG_ENCODERS=torch python eval_chunking.py --reranker bge-v2-m3   # another re-ranker (see eval_models.py)
Writes notebook/results/chunking_eval.json and chunking_ranks.csv (an upload's start with its id; another
re-ranker's end with its name, e.g. chunking_eval_bge-v2-m3.json).
"""
import argparse
import ast
import bisect
import csv
import json
import re
import statistics

from case_study import norm
from eval_hybrid import UPLOADS, first_gold, metrics, top3
from rag_core import PDF_PATH, PROJECT, RESULTS_DIR, TOP_K_FINAL, DocIndex, GDPR_SUBJECT, Models

VARIANTS = {"window": {}, "structure": {"chunking": "structure"},
            "structure+heading": {"chunking": "structure", "headings": True}}
MODES = ("dense", "hybrid")


def notebook_snippets():
    """{query: [snippets]} for the hand-labelled GDPR questions, from the notebook's CANDIDATE_QUERIES."""
    nb = json.loads((PROJECT / "notebook" / "rag_rerank.ipynb").read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"] if "CANDIDATE_QUERIES = [" in "".join(c["source"]))
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CANDIDATE_QUERIES")
    return {q: snippets for _, q, snippets in ast.literal_eval(node.value)}


def questions(doc):
    if doc:
        study = json.loads((UPLOADS / f"{doc}.study.json").read_text(encoding="utf-8"))["queries"]
        hunt = json.loads((UPLOADS / f"{doc}.hunt.json").read_text(encoding="utf-8"))["questions"]
        first = [{"set": "study", "query": q["query"], "gold": q["gold"], "snippets": q["goldSnippets"]} for q in study]
    else:
        snippets = notebook_snippets()
        with open(RESULTS_DIR / "failure_hunt.csv", newline="", encoding="utf-8") as f:
            first = [{"set": "stress", "query": r["query"], "gold": ast.literal_eval(r["gold_chunks"]),
                      "snippets": snippets[r["query"]]} for r in csv.DictReader(f)]
        hunt = json.loads((RESULTS_DIR / "gdpr_hunt.json").read_text(encoding="utf-8"))["questions"]
    return first + [{"set": "hunt", "query": q["query"], "gold": q["gold"], "snippets": q["goldSnippets"]}
                    for q in hunt]


def anchors(q, window_idx, text, offsets):
    """Word spans [(start, end)] of the question's snippets inside its saved gold chunks."""
    spans = []
    ranges = [(window_idx.chunks[g]["start"], window_idx.chunks[g]["end"]) for g in q["gold"]]
    for snip in q["snippets"]:
        snip = norm(snip)
        for m in re.finditer(re.escape(snip), text):
            s = _word_at(offsets, m.start())
            e = _word_at(offsets, m.end() - 1) + 1
            if any(a <= s and e <= b for a, b in ranges):
                spans.append((s, e))
    return spans


def _word_at(offsets, ch):
    """Index of the word containing character ch of the joined text."""
    return bisect.bisect_right(offsets, ch) - 1


def gold_chunks(idx, spans):
    out = set()
    for c in idx.chunks:
        for s, e in spans:
            if min(c["end"], e) - max(c["start"], s) >= (e - s) / 2:
                out.add(c["id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="id of an uploaded document; default: the GDPR")
    ap.add_argument("--reranker", help="a re-ranker from eval_models.RERANKERS instead of the app's (needs torch); "
                                       "its name is added to the output files")
    args = ap.parse_args()
    suffix = f"_{args.reranker}" if args.reranker else ""
    out = lambda name: RESULTS_DIR / ((f"{args.doc}_" if args.doc else "") + name.replace(".", f"{suffix}.", 1))  # noqa: E731

    models = Models()
    if args.reranker:
        import torch
        from sentence_transformers import CrossEncoder

        from eval_models import RERANKERS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if args.reranker == "bge-v2-m3" and device == "cuda" else {}
        models.ce = CrossEncoder(RERANKERS[args.reranker], device=device, max_length=512, **kw)
    if args.doc:
        title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
        build = lambda **kw: DocIndex(models, UPLOADS / f"{args.doc}.pdf", title=title[args.doc], **kw)   # noqa: E731
    else:
        build = lambda **kw: DocIndex(models, PDF_PATH, title="GDPR", subject=GDPR_SUBJECT, profile="gdpr", **kw)  # noqa: E731
    indexes = {name: build(**kw) for name, kw in VARIANTS.items()}
    window = indexes["window"]
    window.retrieve_and_rerank("warm-up")

    text = norm(" ".join(w for w, _ in window.words))
    offsets, pos = [], 0
    for w, _ in window.words:
        offsets.append(pos)
        pos += len(w) + 1
    qs, skipped = [], 0
    for q in questions(args.doc):
        spans = anchors(q, window, text, offsets)
        if spans:
            qs.append({**q, "spans": spans})
        else:
            skipped += 1

    rows = []
    for i, q in enumerate(qs, 1):
        if i % 20 == 0:
            print(f"question {i}/{len(qs)}", flush=True)
        row = {"set": q["set"], "query": q["query"]}
        for name, idx in indexes.items():
            gold = gold_chunks(idx, q["spans"])
            for mode in MODES:
                cands, _ = idx.retrieve_and_rerank(q["query"], mode)
                row[f"{name}/{mode}"] = {
                    "stage1": first_gold(cands, gold, "biRank"), "reranked": first_gold(cands, gold, "ceRank"),
                    "contextWords": sum(len(idx.texts[c].split()) for c in top3(cands, "ceRank"))}
        rows.append(row)

    sets = sorted({q["set"] for q in qs}, key=lambda s: s == "hunt")
    summary = {"chunks": {n: {"count": len(ix.chunks),
                              "meanWords": round(statistics.mean(c["end"] - c["start"] for c in ix.chunks))}
                          for n, ix in indexes.items()},
               "questions": {s: sum(r["set"] == s for r in rows) for s in sets}, "skippedNoSnippet": skipped}
    for s in sets:
        rs = [r for r in rows if r["set"] == s]
        summary[s] = {}
        for key in rows[0]:
            if "/" not in key:
                continue
            summary[s][key] = {stage: metrics([r[key][stage] for r in rs]) for stage in ("stage1", "reranked")}
            summary[s][key]["contextWords"] = round(statistics.mean(r[key]["contextWords"] for r in rs))
        in3 = lambda r, k: r[k]["reranked"] is not None and r[k]["reranked"] <= TOP_K_FINAL   # noqa: E731
        for mode in MODES:
            for name in ("structure", "structure+heading"):
                k_new, k_old = f"{name}/{mode}", f"window/{mode}"
                summary[s][f"{k_new} vs {k_old}"] = {"restored": sum(in3(r, k_new) and not in3(r, k_old) for r in rs),
                                                     "degraded": sum(in3(r, k_old) and not in3(r, k_new) for r in rs)}

    keys = [k for k in rows[0] if "/" in k]
    with open(out("chunking_ranks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "query"] + [f"{k} {st}" for k in keys for st in ("stage1", "reranked")])
        for r in rows:
            w.writerow([r["set"], r["query"]] + [r[k][st] for k in keys for st in ("stage1", "reranked")])
    out("chunking_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
