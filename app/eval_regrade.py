"""Re-judge the answers to the Constitution questions whose gold was corrected by hand (gold_fixes.json).

The same judge and prompt as eval_answers.py, shown the corrected gold chunks and evidence. Only answers the
judge did not label abstained are re-graded: the abstention check never sees the gold, so it still stands.

Needs a GPU and transformers (Kaggle):  RAG_ENCODERS=torch python eval_regrade.py
Reads answers_judged.csv; writes notebook/results/regraded.csv, which eval_clean.py applies.
"""
import csv
import json
import os

os.environ.setdefault("RAG_ENCODERS", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from eval_answers import JUDGE_BATCH, JUDGE_PROMPT, JUDGE_TOKENS, JUDGES, chat_all, load_llm, parse_label  # noqa: E402
from eval_hybrid import UPLOADS  # noqa: E402
from rag_core import RESULTS_DIR, chunk_pages, load_pages  # noqa: E402


def main():
    spec = json.loads((RESULTS_DIR / "gold_fixes.json").read_text(encoding="utf-8"))
    fixes = {f["query"]: f for f in spec["fixes"] if "gold" in f}
    title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
    texts = [c["text"] for c in chunk_pages(load_pages(UPLOADS / "a914ab3a3b.pdf", "generic"))[0]]
    with open(RESULTS_DIR / "answers_judged.csv", newline="", encoding="utf-8") as f:
        keys = sorted({(r["query"], r["answer"]) for r in csv.DictReader(f)
                       if r["doc"] == spec["doc"] and r["query"] in fixes and r["label"] != "abstained"})
    prompts = [JUDGE_PROMPT.format(subject=f'the document "{title["a914ab3a3b"]}"', question=q,
                                   snippet=fixes[q]["evidence"],
                                   gold="\n\n".join(texts[g] for g in fixes[q]["gold"][:2]), answer=a)
               for q, a in keys]
    print(f"re-judging {len(keys)} answers to {len(fixes)} questions with {JUDGES[0]}", flush=True)
    replies = chat_all(load_llm(JUDGES[0], four_bit=True), prompts, JUDGE_TOKENS, JUDGE_BATCH)
    with open(RESULTS_DIR / "regraded.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "query", "answer", "label", "judgeReply"])
        w.writeheader()
        w.writerows({"doc": spec["doc"], "query": q, "answer": a, "label": parse_label(r), "judgeReply": r}
                    for (q, a), r in zip(keys, replies))
    for (q, _), r in zip(keys, replies):
        print(parse_label(r), "|", q[:80])


if __name__ == "__main__":
    main()
