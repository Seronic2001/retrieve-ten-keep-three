"""Recompute the answer, support-check and gate results without the test questions whose gold label is suspect.

A hunt or case-study question's gold chunk is the passage the question writer said it used. When the writer's
evidence quote appears nowhere in that passage, the passage id is probably wrong: on the Constitution, 11 of 124
hunt questions are like that (e.g. questions on Art. 124, Supreme Court judges, with gold in Art. 217), and the
judge, grading against that gold, marks right answers from the real provision wrong. Those questions are dropped
here; every other number is recomputed from the saved results, so no model runs.

Hand corrections (gold_fixes.json) are applied on top: questions with a broken premise are dropped, corrected
gold chunks replace the old ones in goldInTop3, and once eval_regrade.py has run, its labels (regraded.csv)
replace the judge's labels for those questions' answers.

  ../.venv/Scripts/python.exe eval_clean.py
Reads answers_judged.csv, support_checked.csv and gate_scores.csv; writes notebook/results/clean_eval.json with
"all" (as judged), "clean" (auto-detected suspects dropped) and "fixed" (hand corrections too).
"""
import ast
import csv
import json

from eval_gate import auroc
from eval_hybrid import UPLOADS
from rag_core import RESULTS_DIR

LABELS = ("correct", "partial", "wrong", "abstained")
QUESTION_FILES = {"gdpr": [RESULTS_DIR / "gdpr_hunt.json"],
                  "constitution": [UPLOADS / "a914ab3a3b.hunt.json", UPLOADS / "a914ab3a3b.study.json"]}


NO_EVIDENCE = "evidence not in its gold chunk"


def suspect_questions():
    """{(doc, query)} whose evidence quote was not found in their gold chunk, including those correct_hunt() has
    already removed from a hunt file."""
    out = set()
    for doc, paths in QUESTION_FILES.items():
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            out |= {(doc, q["query"]) for q in data.get("questions") or data["queries"] if not q["goldSnippets"]}
            out |= {(doc, d["query"]) for d in data.get("dropped", []) if d["reason"] == NO_EVIDENCE}
    return out


def read(name):
    with open(RESULTS_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_fixes(rows, answer_of):
    """rows with gold_fixes.json applied: dropped questions removed, goldInTop3 recomputed from corrected gold (where
    rows carry top3), and labels replaced by regraded.csv. answer_of(row) gives the row's answer, or None."""
    spec = json.loads((RESULTS_DIR / "gold_fixes.json").read_text(encoding="utf-8"))
    fixes = {(spec["doc"], f["query"]): f for f in spec["fixes"]}
    regraded = ({(r["doc"], r["query"], r["answer"]): r["label"] for r in read("regraded.csv")}
                if (RESULTS_DIR / "regraded.csv").exists() else {})
    out = []
    for r in rows:
        fix = fixes.get((r["doc"], r["query"]))
        if fix and fix.get("drop"):
            continue
        r = dict(r)
        if fix and "top3" in r:
            r["goldInTop3"] = str(bool(set(ast.literal_eval(r["top3"])) & set(fix["gold"])))
        r["label"] = regraded.get((r["doc"], r["query"], answer_of(r)), r["label"])
        out.append(r)
    return out


def share(rows, pred):
    return round(sum(map(pred, rows)) / len(rows), 2) if rows else None


def answers(rows):
    out = {}
    for key in sorted({(r["doc"], r["pipeline"]) for r in rows}):
        rs = [r for r in rows if (r["doc"], r["pipeline"]) == key]
        out["/".join(key)] = {"n": len(rs), **{lab: share(rs, lambda r, lab=lab: r["label"] == lab) for lab in LABELS},
                              "goldInTop3": share(rs, lambda r: r["goldInTop3"] == "True")}
    return out


def support(rows):
    """Per checker and document, pooled over pipelines; abstentions were not checked."""
    out = {}
    checkers = [c.removesuffix("/flagged") for c in rows[0] if c.endswith("/flagged")]
    for checker in checkers:
        for doc in sorted({r["doc"] for r in rows}):
            rs = [r for r in rows if r["doc"] == doc and r["label"] != "abstained"]
            flagged = lambda r: r[f"{checker}/flagged"] == "True"   # noqa: E731
            wrong, correct = [r for r in rs if r["label"] == "wrong"], [r for r in rs if r["label"] == "correct"]
            out[f"{checker}/{doc}"] = {"n": len(rs), "wrong": len(wrong), "wrongFlagged": share(wrong, flagged),
                                       "correctFlagged": share(correct, flagged),
                                       "wrongAmongUnflagged": share([r for r in rs if not flagged(r)],
                                                                    lambda r: r["label"] == "wrong")}
    return out


def gate(rows):
    out = {}
    for key in sorted({(r["doc"], r["mode"]) for r in rows}):
        rs = [r for r in rows if (r["doc"], r["mode"]) == key]
        out["/".join(key)] = {"n": len(rs), "aurocCorrectVsWrong": auroc(
            [float(r["top"]) for r in rs if r["label"] == "correct"], [float(r["top"]) for r in rs if r["label"] == "wrong"])}
    return out


def correct_hunt():
    """Applies the same corrections to the Constitution failure hunt the UI shows (app/uploads/<id>.hunt.json and
    .csv): suspect and broken questions dropped, corrected gold re-scored. Running it again changes nothing."""
    from case_study import _status, _tally, hunt_csv
    from rag_core import chunk_pages, load_pages
    path = UPLOADS / "a914ab3a3b.hunt.json"
    hunt = json.loads(path.read_text(encoding="utf-8"))
    spec = json.loads((RESULTS_DIR / "gold_fixes.json").read_text(encoding="utf-8"))
    fixes = {f["query"]: f for f in spec["fixes"]}
    suspect = suspect_questions()
    texts = None
    questions, dropped = [], hunt.get("dropped", [])
    for q in hunt["questions"]:
        fix = fixes.get(q["query"])
        if (spec["doc"], q["query"]) in suspect or (fix and fix.get("drop")):
            dropped.append({"query": q["query"], "reason": fix["note"] if fix else NO_EVIDENCE})
            continue
        if fix and q["gold"] != fix["gold"]:
            gold = set(fix["gold"])
            for c in q["candidates"]:
                c["gold"] = c["chunk"] in gold
            rank = lambda key: min((c[key] for c in q["candidates"] if c["gold"]), default=None)   # noqa: E731
            q.update(gold=fix["gold"], goldSnippets=[fix["evidence"]], baselineRank=rank("biRank"),
                     rerankedRank=rank("ceRank"))
            q["status"] = _status(q["baselineRank"], q["rerankedRank"])
            if q["status"] in ("restored", "degraded") and "text" not in q["candidates"][0]:   # the UI shows these
                texts = texts or [c["text"] for c in chunk_pages(load_pages(UPLOADS / "a914ab3a3b.pdf", "generic"))[0]]
                for c in q["candidates"]:
                    c["text"] = texts[c["chunk"]]
        questions.append(q)
    hunt["questions"], hunt["dropped"] = questions, dropped
    hunt["summary"] = {"all": _tally(questions),
                       "natural": _tally([q for q in questions if q["category"] != "trap"]),
                       "trap": _tally([q for q in questions if q["category"] == "trap"])}
    hunt["generated"]["goldCorrected"] = (
        f"Gold checked by hand: questions whose quoted evidence is missing from their answer chunk or whose premise is "
        f"broken are dropped, and {sum('gold' in f for f in spec['fixes'])} answer chunks are corrected "
        f"(notebook/results/gold_fixes.json).")
    path.write_text(json.dumps(hunt, ensure_ascii=False), encoding="utf-8")
    hunt_csv(hunt, path.with_suffix(".csv"))
    return hunt["summary"]


def main():
    print("Constitution hunt corrected:", correct_hunt()["all"])
    suspect = suspect_questions()
    keep = lambda rows: [r for r in rows if (r["doc"], r["query"]) not in suspect]   # noqa: E731
    judged = read("answers_judged.csv")
    # gate rows carry no answer: theirs is the MiniLM-L6 answer for the same question and retrieval mode
    gate_answer = {(r["doc"], r["query"], r["pipeline"].removesuffix("+MiniLM-L6")): r["answer"] for r in judged}
    sources = {"answers": (answers, judged, lambda r: r["answer"]),
               "support": (support, read("support_checked.csv"), lambda r: r["answer"]),
               "gate": (gate, read("gate_scores.csv"), lambda r: gate_answer.get((r["doc"], r["query"], r["mode"])))}
    summary = {"suspectQuestions": sorted(f"{d}: {q}" for d, q in suspect),
               "regraded": (RESULTS_DIR / "regraded.csv").exists()}
    for name, (fn, rows, answer_of) in sources.items():
        summary[name] = {"all": fn(rows), "clean": fn(keep(rows)), "fixed": fn(apply_fixes(keep(rows), answer_of))}
    (RESULTS_DIR / "clean_eval.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(suspect), "suspect questions dropped; regraded labels applied:", summary["regraded"])
    for name in sources:
        for key, before in summary[name]["all"].items():
            print(f"{name:8} {key:40}", before)
            for stage in ("clean", "fixed"):
                print(" " * 43, f"{stage:5}", summary[name][stage][key])


if __name__ == "__main__":
    main()
