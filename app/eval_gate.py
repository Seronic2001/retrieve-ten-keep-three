"""Confidence gate: would refusing to answer when the cross-encoder's top score is low remove the wrong answers?

For every GDPR and Constitution question, dense and hybrid retrieval with the app's re-ranker (MiniLM-L6) give
the top-3 scores. Each question is then joined to two outcomes:
  goldInTop3  whether the gold chunk reached the LLM (retrieval-level, every question)
  label       the judge's verdict on the answer from that top 3 (answers_judged.csv, from eval_answers.py);
              left empty when the local top 3 differs from the one judged (Kaggle ran the encoders in PyTorch)
A gate at threshold t abstains when the signal is below t. For each t the script reports how many questions it
still answers, Hit@3 among them, and which share of the retrieval misses, wrong answers and correct answers it
turns into abstentions. Signals: the top score, and the margin between the top two scores. Raw cross-encoder
logits need not be comparable across documents, so the threshold picked on one document is also checked on the
other.

  ../.venv/Scripts/python.exe eval_gate.py
Writes notebook/results/gate_eval.json and gate_scores.csv.
"""
import csv
import json

import numpy as np

from eval_hybrid import MODES, UPLOADS, question_sets, upload_sets
from rag_core import RESULTS_DIR, TOP_K_FINAL, DocIndex, Models, gdpr_index

SIGNALS = {"top": lambda s: s[0], "margin": lambda s: s[0] - s[1]}
MAX_CORRECT_LOST = (0.05, 0.10, 0.20)   # operating points: the gate may abstain on at most this share of correct


def auroc(pos, neg):
    """P(a random positive scores above a random negative), ties counted half."""
    if not pos or not neg:
        return None
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return round(float((p > n).mean() + 0.5 * (p == n).mean()), 3)


def gate_stats(rows, signal, t):
    kept = [r for r in rows if r[signal] >= t]
    share = lambda sel, pred: round(sum(map(pred, sel)) / len(sel), 2) if sel else None   # noqa: E731
    out = {"threshold": round(float(t), 3), "answered": share(rows, lambda r: r[signal] >= t),
           "hit@3Answered": share(kept, lambda r: r["goldInTop3"]),
           "missesCaught": share([r for r in rows if not r["goldInTop3"]], lambda r: r[signal] < t)}
    judged = [r for r in rows if r["label"]]
    if judged:
        out["wrongCaught"] = share([r for r in judged if r["label"] == "wrong"], lambda r: r[signal] < t)
        out["correctLost"] = share([r for r in judged if r["label"] == "correct"], lambda r: r[signal] < t)
        out["wrongAnswered"] = share([r for r in judged if r[signal] >= t], lambda r: r["label"] == "wrong")
    return out


def operating_points(rows, signal):
    """The highest threshold that loses at most each MAX_CORRECT_LOST share of correct answers."""
    correct = sorted(r[signal] for r in rows if r["label"] == "correct")
    if not correct:
        return {}
    return {f"correctLost<={m}": gate_stats(rows, signal, correct[int(m * len(correct))]) for m in MAX_CORRECT_LOST}


def main():
    models = Models()
    title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
    stress, hunt = question_sets()
    docs = {"gdpr": (gdpr_index(models), stress + hunt),
            "constitution": (DocIndex(models, UPLOADS / "a914ab3a3b.pdf", title=title["a914ab3a3b"],
                                      cache=UPLOADS / "a914ab3a3b.emb.npz"), sum(upload_sets("a914ab3a3b"), []))}
    labels = {}
    judged = RESULTS_DIR / "answers_judged.csv"
    if judged.exists():
        with open(judged, newline="", encoding="utf-8") as f:
            labels = {(r["doc"], r["query"], r["pipeline"]): (r["label"], json.loads(r["top3"]))
                      for r in csv.DictReader(f)}

    rows = []
    for doc, (idx, qs) in docs.items():
        for q in qs:
            for mode in MODES:
                cands, _ = idx.retrieve_and_rerank(q["query"], mode)
                ranked = sorted(cands, key=lambda c: c["ceRank"])
                scores = [c["ceScore"] for c in ranked]
                top = [c["chunk"] for c in ranked[:TOP_K_FINAL]]
                label, judged_top = labels.get((doc, q["query"], f"{mode}+MiniLM-L6"), ("", None))
                rows.append({"doc": doc, "set": q["set"], "mode": mode, "query": q["query"],
                             "goldInTop3": bool(set(top) & set(q["gold"])),
                             "label": label if judged_top == top else "", "top3Mismatch": judged_top not in (None, top),
                             **{name: round(f(scores), 3) for name, f in SIGNALS.items()}})

    summary = {"reranker": "MiniLM-L6", "judgedRows": sum(bool(r["label"]) for r in rows),
               "top3Mismatches": sum(r["top3Mismatch"] for r in rows), "groups": {}}
    for doc in docs:
        for mode in MODES:
            rs = [r for r in rows if r["doc"] == doc and r["mode"] == mode]
            g = {"n": len(rs), "hit@3": round(sum(r["goldInTop3"] for r in rs) / len(rs), 2)}
            for signal in SIGNALS:
                g[signal] = {
                    "aurocGoldInTop3": auroc([r[signal] for r in rs if r["goldInTop3"]],
                                             [r[signal] for r in rs if not r["goldInTop3"]]),
                    "aurocCorrectVsWrong": auroc([r[signal] for r in rs if r["label"] == "correct"],
                                                 [r[signal] for r in rs if r["label"] == "wrong"]),
                    "sweep": [gate_stats(rs, signal, t) for t in np.percentile([r[signal] for r in rs],
                                                                               [0, 10, 20, 30, 40, 50])],
                    "operatingPoints": operating_points(rs, signal)}
            summary["groups"][f"{doc}/{mode}"] = g

    # a threshold chosen on one document, applied to the other
    transfer = {}
    for src, dst in (("gdpr", "constitution"), ("constitution", "gdpr")):
        for mode in MODES:
            src_rows = [r for r in rows if r["doc"] == src and r["mode"] == mode]
            dst_rows = [r for r in rows if r["doc"] == dst and r["mode"] == mode]
            for name, point in operating_points(src_rows, "top").items():
                transfer[f"{src}->{dst}/{mode}/{name}"] = gate_stats(dst_rows, "top", point["threshold"])
    summary["transferTopScore"] = transfer

    (RESULTS_DIR / "gate_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RESULTS_DIR / "gate_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, g in summary["groups"].items():
        print(name, {k: v for k, v in g.items() if k in ("n", "hit@3")},
              {s: (g[s]["aurocGoldInTop3"], g[s]["aurocCorrectVsWrong"]) for s in SIGNALS})
        for op, p in g["top"]["operatingPoints"].items():
            print("   top", op, p)
    for k, v in transfer.items():
        print("transfer", k, v)


if __name__ == "__main__":
    main()
