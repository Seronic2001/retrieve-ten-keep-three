"""Dense vs hybrid (dense + BM25, RRF) retrieval on the GDPR, each followed by the same cross-encoder.

Question sets, with their gold chunks, come from the notebook's saved results:
  stress  notebook/results/failure_hunt.csv  the 13 hand-labelled stress-test questions
  hunt    notebook/results/gdpr_hunt.json    the 100 generated questions (natural and wording traps)
The dense rows should reproduce the report's Table 2, which checks that nothing else changed.

  ../.venv/Scripts/python.exe eval_hybrid.py              # retrieval metrics and latency only
  ../.venv/Scripts/python.exe eval_hybrid.py --answers 8  # plus answers from the local model (start_local_llm.sh)
                                                          # for up to 8 questions whose top 3 changed
  ../.venv/Scripts/python.exe eval_hybrid.py --hyde       # adds "hyde": hybrid plus a passage the local model writes
                                                          # in the regulation's wording (cached in hyde_passages.json)
Writes notebook/results/hybrid_eval.json, hybrid_ranks.csv and (with --answers) hybrid_answers.csv.
"""
import argparse
import ast
import csv
import json
import statistics

from rag_core import RESULTS_DIR, TOP_K_FINAL, Models, gdpr_index, generate_local, local_status

MODES = ("dense", "hybrid")
LATENCY_RUNS = 5
HYDE_PROMPT = """Write the passage of {subject} that answers the question below, in the regulation's own legal
wording (for example "shall", "the controller", "the data subject"). Write one or two sentences and nothing else.

Question: {question}

Passage:"""


def question_sets():
    with open(RESULTS_DIR / "failure_hunt.csv", newline="", encoding="utf-8") as f:
        stress = [{"set": "stress", "category": r["category"], "query": r["query"],
                   "gold": ast.literal_eval(r["gold_chunks"])} for r in csv.DictReader(f)]
    hunt = json.loads((RESULTS_DIR / "gdpr_hunt.json").read_text(encoding="utf-8"))["questions"]
    hunt = [{"set": "hunt", "category": "trap" if q["category"] == "trap" else "natural", "query": q["query"],
             "gold": q["gold"], "evidence": (q["goldSnippets"] or [""])[0]} for q in hunt]
    return stress, hunt


def first_gold(candidates, gold, key):
    """Rank of the best-placed gold chunk under key ('biRank' or 'ceRank'), or None if none was retrieved."""
    return min((c[key] for c in candidates if c["chunk"] in gold), default=None)


def metrics(ranks):
    n = len(ranks)
    hit = lambda k: round(sum(r is not None and r <= k for r in ranks) / n, 2)   # noqa: E731
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@10": hit(10),
            "mrr@10": round(sum(1 / r for r in ranks if r is not None) / n, 2)}


def top3(candidates, key):
    return [c["chunk"] for c in sorted(candidates, key=lambda c: c[key])[:TOP_K_FINAL]]


def hyde_passages(idx, questions):
    """One local-model passage per question, cached by question so a rerun costs no generation."""
    path = RESULTS_DIR / "hyde_passages.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    model = None
    for i, q in enumerate(questions, 1):
        if q["query"] in cache:
            continue
        model = model or local_status(timeout=3)
        if not model:
            raise SystemExit("The local model is not reachable; start app/start_local_llm.sh in WSL first")
        print(f"hyde {i}/{len(questions)}: {q['query'][:70]}", flush=True)
        text, used, ms = generate_local(HYDE_PROMPT.format(subject=idx.subject, question=q["query"]), max_tokens=120)
        cache[q["query"]] = {"text": text, "model": used, "ms": ms}
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=int, default=0, help="answer up to N changed questions with the local model")
    ap.add_argument("--hyde", action="store_true", help="also evaluate hybrid with a local-model HyDE passage")
    args = ap.parse_args()
    modes = MODES + ("hyde",) if args.hyde else MODES

    idx = gdpr_index(Models())
    idx.retrieve_and_rerank("warm-up")
    stress, hunt = question_sets()
    hyde = hyde_passages(idx, stress + hunt) if args.hyde else {}
    rows = []
    for q in stress + hunt:
        row = dict(q)
        for mode in modes:
            cands, _ = (idx.retrieve_and_rerank(q["query"], "hybrid", hyde[q["query"]]["text"]) if mode == "hyde"
                        else idx.retrieve_and_rerank(q["query"], mode))
            row[mode] = {"stage1": first_gold(cands, q["gold"], "biRank"),
                         "reranked": first_gold(cands, q["gold"], "ceRank"),
                         "top3": top3(cands, "ceRank")}
        rows.append(row)

    groups = {"stress": [r for r in rows if r["set"] == "stress"],
              "hunt": [r for r in rows if r["set"] == "hunt"],
              "hunt/natural": [r for r in rows if r["set"] == "hunt" and r["category"] == "natural"],
              "hunt/trap": [r for r in rows if r["set"] == "hunt" and r["category"] == "trap"]}
    summary = {}
    for name, rs in groups.items():
        summary[name] = {f"{mode}/{stage}": metrics([r[mode][stage] for r in rs])
                         for mode in modes for stage in ("stage1", "reranked")}
        in3 = lambda r, m: r[m]["reranked"] is not None and r[m]["reranked"] <= TOP_K_FINAL   # noqa: E731
        for new, old in [("hybrid", "dense")] + ([("hyde", "dense"), ("hyde", "hybrid")] if args.hyde else []):
            summary[name][f"{new}Vs{old.title()}"] = {
                "restored": sum(in3(r, new) and not in3(r, old) for r in rs),
                "degraded": sum(in3(r, old) and not in3(r, new) for r in rs),
                "recovered@10": sum(r[old]["stage1"] is None and r[new]["stage1"] is not None for r in rs),
                "lost@10": sum(r[old]["stage1"] is not None and r[new]["stage1"] is None for r in rs)}

    latency = {}
    for mode in MODES:
        runs = [idx.retrieve_and_rerank(q["query"], mode)[1] for _ in range(LATENCY_RUNS) for q in stress]
        latency[mode] = {k: round(statistics.mean(t.get(k, 0) for t in runs), 2)
                         for k in ("embedMs", "searchMs", "fuseMs", "rerankMs")}
        latency[mode]["totalMs"] = round(sum(latency[mode].values()), 2)

    if args.hyde:
        latency["hyde"] = {"generateMs": round(statistics.mean(h["ms"] for h in hyde.values()))}

    with open(RESULTS_DIR / "hybrid_ranks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "category", "query", "gold"] + [f"{m}_{k}" for m in modes for k in ("stage1", "reranked")]
                   + [f"{m}_top3" for m in modes])
        for r in rows:
            w.writerow([r["set"], r["category"], r["query"], r["gold"]]
                       + [r[m][k] for m in modes for k in ("stage1", "reranked")] + [r[m]["top3"] for m in modes])

    answers = []
    if args.answers:
        model = local_status(timeout=3)
        if not model:
            raise SystemExit("The local model is not reachable; start app/start_local_llm.sh in WSL first")
        in3 = lambda r, m: r[m]["reranked"] is not None and r[m]["reranked"] <= TOP_K_FINAL   # noqa: E731
        changed = [r for r in rows if in3(r, "dense") != in3(r, "hybrid")][:args.answers]
        for i, r in enumerate(changed, 1):
            print(f"answering {i}/{len(changed)} with {model}: {r['query'][:70]}", flush=True)
            out = {"set": r["set"], "category": r["category"], "query": r["query"], "gold": r["gold"],
                   "evidence": r.get("evidence", "")}
            for mode in MODES:
                text, used, ms = generate_local(idx.build_prompt(r["query"], r[mode]["top3"]))
                out.update({f"{mode}_gold_in_top3": in3(r, mode), f"{mode}_answer": text, f"{mode}_ms": ms,
                            "model": used})
            answers.append(out)
        with open(RESULTS_DIR / "hybrid_answers.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(answers[0]) if answers else ["query"])
            w.writeheader()
            w.writerows(answers)

    (RESULTS_DIR / "hybrid_eval.json").write_text(json.dumps(
        {"summary": summary, "latency": latency, "answered": len(answers)}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "latency": latency}, indent=2))


if __name__ == "__main__":
    main()
