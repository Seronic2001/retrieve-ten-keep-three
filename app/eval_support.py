"""Support check: can a model that sees only the question, the three retrieved passages and the answer (never the
gold passage) flag the answers the judge marked wrong? This is the check the app could run after answering.

The checker answers two questions per answer:
  passagesAnswer   do the passages answer this question, or only something near it (a neighbouring provision,
                   a different article)? yes / partly / no
  answerSupported  is everything the answer asserts stated in the passages? yes / no
An answer is flagged when passagesAnswer is "no" or answerSupported is "no"; a flagged answer would be replaced
by "not sure" or trigger another retrieval. Answers the judge labelled abstained are skipped: they already say so.
The flags are scored against the judge's labels in answers_judged.csv (from eval_answers.py): the share of wrong
answers flagged, and the share of correct answers flagged by mistake.

Checkers: the app's answer model (what would run in deployment) and the 14B judge (an upper bound).
Needs a GPU (Kaggle T4 x2):  RAG_ENCODERS=torch python eval_support.py
Writes notebook/results/support_eval.json and support_checked.csv; raw replies go to support_raw.json.
"""
import ast
import csv
import json
import re

from eval_answers import GENERATOR, GEN_BATCH, JUDGES, LABELS, chat_all, documents, load_llm, \
    release_gpu
from rag_core import RESULTS_DIR, Models

CHECKERS = {GENERATOR: False, JUDGES[0]: True}   # model: load in 4-bit
# the 14B ran a T4 out of memory on a pair of long prompts (question + 3 passages + answer) even in batches of 2;
# after a CUDA OOM the device is unusable, so it gets one prompt at a time rather than a retry
CHECK_BATCH = {False: GEN_BATCH, True: 1}
CHECK_TOKENS = 100

CHECK_PROMPT = """You are checking an answer about {subject} before it is shown to a user. The answer was written
from the passages below and nothing else.

Question: {question}

{passages}

Answer:
{answer}

1. Do the passages answer this question, or only something near it, for example a neighbouring provision or a
   different article than the one asked about? "yes", "partly" or "no".
2. Is everything the answer asserts stated in the passages? "yes" or "no".

Reply with JSON only: {{"passagesAnswer": "yes|partly|no", "answerSupported": "yes|no", "reason": "<one sentence>"}}"""


def parse_check(reply):
    get = lambda key, opts: next((m.group(1).lower() for m in [re.search(rf'"{key}"\s*:\s*"({opts})"', reply, re.I)]
                                  if m), "unparsed")   # noqa: E731
    return get("passagesAnswer", "yes|partly|no"), get("answerSupported", "yes|no")


def main():
    docs = documents(Models())
    with open(RESULTS_DIR / "answers_judged.csv", newline="", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    keys = sorted({(it["doc"], it["query"], it["top3"], it["answer"]) for it in items if it["label"] != "abstained"})

    def prompt(doc, query, top3, answer):
        idx = docs[doc][0]
        passages = "\n\n".join(f"[Passage {i + 1}]\n{idx.texts[c]}" for i, c in enumerate(ast.literal_eval(top3)))
        return CHECK_PROMPT.format(subject=idx.subject, question=query, passages=passages, answer=answer)

    # replies are saved after each checker, so a crash in the second keeps the first; a rerun reuses them
    raw = RESULTS_DIR / "support_raw.json"
    saved = json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else {}
    checks = {name: {tuple(r["key"]): r["reply"] for r in saved.get(name, [])} for name in CHECKERS}
    for name, four_bit in CHECKERS.items():
        todo = [k for k in keys if k not in checks[name]]
        if not todo:
            continue
        copies = load_llm(name, four_bit=four_bit)
        print(f"checking {len(todo)} answers with {name} on {len(copies)} GPU(s) "
              f"({len(keys) - len(todo)} saved)", flush=True)
        replies = chat_all(copies, [prompt(*k) for k in todo], CHECK_TOKENS, CHECK_BATCH[four_bit])
        checks[name].update(zip(todo, replies))
        raw.write_text(json.dumps({n: [{"key": list(k), "reply": r} for k, r in c.items()] for n, c in checks.items()},
                                  ensure_ascii=False), encoding="utf-8")
        del copies
        release_gpu()

    rows = []
    for it in items:
        row = {k: it[k] for k in ("doc", "set", "pipeline", "query", "top3", "goldInTop3", "label", "answer")}
        key = (it["doc"], it["query"], it["top3"], it["answer"])
        for name in CHECKERS:
            short = name.split("/")[-1]
            reply = checks[name].get(key, "")
            pa, sup = parse_check(reply) if reply else ("", "")
            row.update({f"{short}/passagesAnswer": pa, f"{short}/answerSupported": sup,
                        f"{short}/flagged": bool(reply) and (pa == "no" or sup == "no"), f"{short}/reply": reply})
        rows.append(row)

    summary = {"checkers": list(CHECKERS), "checked": len(keys), "results": {}}
    for name in CHECKERS:
        short = name.split("/")[-1]
        for doc in docs:
            for pipe in sorted({r["pipeline"] for r in rows}):
                rs = [r for r in rows if r["doc"] == doc and r["pipeline"] == pipe and r["label"] != "abstained"]
                share = lambda sel: round(sum(r[f"{short}/flagged"] for r in sel) / len(sel), 2) if sel else None  # noqa: E731
                kept = [r for r in rs if not r[f"{short}/flagged"]]
                summary["results"][f"{short}/{doc}/{pipe}"] = {
                    "n": len(rs), "flagged": share(rs),
                    **{f"{lab}Flagged": share([r for r in rs if r["label"] == lab]) for lab in LABELS[:3]},
                    "goldMissingFlagged": share([r for r in rs if r["goldInTop3"] == "False"]),
                    "wrongAmongUnflagged": round(sum(r["label"] == "wrong" for r in kept) / len(kept), 2) if kept else None,
                    "wrongBefore": round(sum(r["label"] == "wrong" for r in rs) / len(rs), 2),
                    "unparsed": sum(r[f"{short}/passagesAnswer"] == "unparsed" for r in rs)}

    (RESULTS_DIR / "support_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RESULTS_DIR / "support_checked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
