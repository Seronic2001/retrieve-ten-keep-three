"""Answer-level evaluation: generate an answer for every question under several pipelines, then have a larger
LLM judge each answer against the gold evidence as correct / partial / wrong / abstained.

Pipelines (sliding-window chunks, so the saved gold chunk ids apply):
  dense+MiniLM-L6    the notebook's Pipeline 2
  hybrid+MiniLM-L6   the app's Hybrid toggle
  hybrid+bge-v2-m3   the strongest re-ranker in eval_models.py
Answers come from GENERATOR with the app's prompt (DocIndex.build_prompt), greedy and without thinking, as in
start_local_llm.sh; an answer is generated once per distinct top 3. The judge is a different, larger model.
It is calibrated first: it also labels the answers graded by hand in hand_labels.json, and its agreement with
those labels is reported next to the results.

Needs a GPU and transformers (Kaggle):  RAG_ENCODERS=torch python eval_answers.py
Writes notebook/results/answers_eval.json (summary and calibration) and answers_judged.csv (every answer).
The generated answers are saved to answers_raw.json before judging and reused on a rerun.
"""
import csv
import gc
import json
import os
import re
import time

os.environ.setdefault("RAG_ENCODERS", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from eval_chunking import notebook_snippets  # noqa: E402
from eval_hybrid import UPLOADS, question_sets, upload_sets  # noqa: E402
from eval_models import RERANKERS  # noqa: E402
from rag_core import PDF_PATH, RESULTS_DIR, TOP_K_FINAL, DocIndex, GDPR_SUBJECT, Models  # noqa: E402

GENERATOR = "Qwen/Qwen3.5-4B"
JUDGES = ("Qwen/Qwen3-14B", "Qwen/Qwen2.5-14B-Instruct")   # 4-bit; the first that loads
PIPELINES = {"dense+MiniLM-L6": ("dense", "MiniLM-L6"), "hybrid+MiniLM-L6": ("hybrid", "MiniLM-L6"),
             "hybrid+bge-v2-m3": ("hybrid", "bge-v2-m3")}
LABELS = ("correct", "partial", "wrong", "abstained")
GEN_TOKENS, JUDGE_TOKENS = 512, 120
GEN_BATCH, JUDGE_BATCH = 8, 2   # 14B judge prompts are long; batches of 8 ran a T4 out of memory

JUDGE_PROMPT = """You are grading an answer to a question about {subject}. The gold evidence below is the passage
that contains the true answer; judge the answer only against it.

Question: {question}

Gold evidence (key phrase): {snippet}

Gold passage:
{gold}

Answer to grade:
{answer}

Labels:
- correct: the answer states the fact in the gold evidence, and nothing it asserts contradicts it.
- partial: it gives only part of that fact, or the right content with a wrong article or a material slip.
- wrong: it asserts something the evidence contradicts, gives a different rule as the answer, or claims the rule
  does not exist.
- abstained: it says the context does not contain the answer, without asserting a wrong answer.

Reply with JSON only: {{"label": "correct|partial|wrong|abstained", "reason": "<one sentence>"}}"""


def documents(models):
    snippets = notebook_snippets()
    stress, hunt = question_sets()
    for q in stress:
        q["evidence"] = "; ".join(snippets[q["query"]])
    gdpr = DocIndex(models, PDF_PATH, title="GDPR", subject=GDPR_SUBJECT, profile="gdpr")
    title = {d["id"]: d["title"] for d in json.loads((UPLOADS / "library.json").read_text(encoding="utf-8"))}
    const = DocIndex(models, UPLOADS / "a914ab3a3b.pdf", title=title["a914ab3a3b"])
    return {"gdpr": (gdpr, stress + hunt), "constitution": (const, sum(upload_sets("a914ab3a3b"), []))}


def load_llm(name, four_bit=False):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    kw = ({"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.float16)}
          if four_bit else {"torch_dtype": torch.float16})
    return tok, AutoModelForCausalLM.from_pretrained(name, device_map="cuda", **kw)


def chat(tok, model, prompts, max_new, batch):
    """Greedy replies to prompts, batched, with thinking switched off where the chat template supports it."""
    def template(p):
        msgs = [{"role": "user", "content": p}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok([template(p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        print(f"   {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)
    return out


def parse_label(reply):
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        label = json.loads(m.group(0))["label"].strip().lower() if m else ""
    except (ValueError, KeyError, AttributeError):
        label = ""
    if label not in LABELS:
        label = next((lab for lab in LABELS if lab in reply.lower()), "unparsed")
    return label


def release_gpu():
    """Return freed GPU memory to the allocator; callers drop their references first."""
    gc.collect()
    torch.cuda.empty_cache()


def main():
    models = Models()
    docs = documents(models)

    # 1. top 3 per question and pipeline
    ce_cache = {"MiniLM-L6": models.ce}
    items = []   # one per (doc, question, pipeline)
    for doc, (idx, qs) in docs.items():
        for pname, (mode, reranker) in PIPELINES.items():
            if reranker not in ce_cache:
                kw = {"model_kwargs": {"torch_dtype": torch.float16}} if reranker == "bge-v2-m3" else {}
                ce_cache[reranker] = CrossEncoder(RERANKERS[reranker], device="cuda", max_length=512, **kw)
            models.ce = ce_cache[reranker]
            for q in qs:
                cands, _ = idx.retrieve_and_rerank(q["query"], mode)
                top = [c["chunk"] for c in sorted(cands, key=lambda c: c["ceRank"])[:TOP_K_FINAL]]
                items.append({"doc": doc, "set": q["set"], "category": q["category"], "query": q["query"],
                              "pipeline": pname, "top3": top, "goldInTop3": bool(set(top) & set(q["gold"])),
                              "gold": q["gold"], "evidence": q.get("evidence", "")})
        models.ce = ce_cache["MiniLM-L6"]
    ce_cache.clear()
    release_gpu()

    # 2. answers, once per distinct (doc, question, top 3)
    keys = sorted({(it["doc"], it["query"], tuple(it["top3"])) for it in items})
    raw = RESULTS_DIR / "answers_raw.json"
    saved_raw = json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else {}
    answers = {(r["doc"], r["query"], tuple(r["top3"])): r["answer"] for r in saved_raw.get("answers", [])}
    gen_minutes = saved_raw.get("generationMinutes")
    todo = [k for k in keys if k not in answers]
    if todo:
        tok, llm = load_llm(GENERATOR)
        print(f"generating {len(todo)} answers with {GENERATOR} ({len(keys) - len(todo)} saved)", flush=True)
        t0 = time.time()
        replies = chat(tok, llm, [docs[d][0].build_prompt(q, list(t)) for d, q, t in todo], GEN_TOKENS, GEN_BATCH)
        gen_minutes = round((time.time() - t0) / 60, 1)
        answers.update(zip(todo, replies))
        raw.write_text(json.dumps({"generator": GENERATOR, "generationMinutes": gen_minutes,
                                   "answers": [{"doc": d, "query": q, "top3": list(t), "answer": answers[(d, q, t)]}
                                               for d, q, t in keys]}, ensure_ascii=False), encoding="utf-8")
        del llm, tok
        release_gpu()
    for it in items:
        it["answer"] = answers[(it["doc"], it["query"], tuple(it["top3"]))]

    # 3. the judge: hand-labelled answers first (calibration), then every distinct answer
    calib = []
    hand = json.loads((RESULTS_DIR / "hand_labels.json").read_text(encoding="utf-8"))["labels"]
    saved = {}
    for doc, f in (("gdpr", "hybrid_answers.csv"), ("constitution", "a914ab3a3b_hybrid_answers.csv")):
        with open(RESULTS_DIR / f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                saved[(doc, r["query"])] = r
    gold_of = {(d, q["query"]): q for d, (_, qs) in docs.items() for q in qs}
    for h in hand:
        for pipe in ("dense", "hybrid"):
            calib.append({"doc": h["doc"], "query": h["query"], "pipe": pipe, "human": h[pipe],
                          "answer": saved[(h["doc"], h["query"])][f"{pipe}_answer"]})

    def judge_prompt(doc, query, answer):
        idx, q = docs[doc][0], gold_of[(doc, query)]
        gold = "\n\n".join(idx.texts[g] for g in q["gold"][:2])
        return JUDGE_PROMPT.format(subject=idx.subject, question=query, snippet=q.get("evidence") or "(see passage)",
                                   gold=gold, answer=answer)

    judge_name = None
    for name in JUDGES:
        try:
            jtok, judge = load_llm(name, four_bit=True)
            judge_name = name
            break
        except Exception as e:  # noqa: BLE001 - try the next judge
            print(f"could not load {name}: {e}", flush=True)
    print(f"judging with {judge_name}", flush=True)
    c_replies = chat(jtok, judge, [judge_prompt(c["doc"], c["query"], c["answer"]) for c in calib], JUDGE_TOKENS,
                     JUDGE_BATCH)
    for c, r in zip(calib, c_replies):
        c["judge"], c["judgeReply"] = parse_label(r), r
    akeys = sorted({(it["doc"], it["query"], it["answer"]) for it in items})
    a_replies = chat(jtok, judge, [judge_prompt(d, q, a) for d, q, a in akeys], JUDGE_TOKENS, JUDGE_BATCH)
    verdict = {k: (parse_label(r), r) for k, r in zip(akeys, a_replies)}
    for it in items:
        it["label"], it["judgeReply"] = verdict[(it["doc"], it["query"], it["answer"])]

    # 4. summary
    agree = sum(c["judge"] == c["human"] for c in calib)
    binary = sum((c["judge"] == "correct") == (c["human"] == "correct") for c in calib)
    confusion = {h: {j: sum(c["human"] == h and c["judge"] == j for c in calib) for j in LABELS + ("unparsed",)}
                 for h in LABELS}
    summary = {"generator": GENERATOR, "judge": judge_name, "generationMinutes": gen_minutes,
               "distinctAnswers": len(keys),
               "calibration": {"answers": len(calib), "exactAgreement": round(agree / len(calib), 2),
                               "correctVsNotAgreement": round(binary / len(calib), 2),
                               "confusion_humanRows_judgeColumns": confusion},
               "results": {}}
    for doc in docs:
        for s in sorted({it["set"] for it in items if it["doc"] == doc}, key=lambda s: s == "hunt"):
            for pname in PIPELINES:
                rs = [it for it in items if it["doc"] == doc and it["set"] == s and it["pipeline"] == pname]
                share = lambda pred: round(sum(map(pred, rs)) / len(rs), 2)   # noqa: E731
                summary["results"][f"{doc}/{s}/{pname}"] = {
                    "n": len(rs), **{lab: share(lambda it, lab=lab: it["label"] == lab) for lab in LABELS},
                    "goldInTop3": share(lambda it: it["goldInTop3"]),
                    "correctWhenGoldInTop3": round(sum(it["label"] == "correct" for it in rs if it["goldInTop3"])
                                                   / max(1, sum(it["goldInTop3"] for it in rs)), 2),
                    "wrongWhenGoldMissing": round(sum(it["label"] == "wrong" for it in rs if not it["goldInTop3"])
                                                  / max(1, sum(not it["goldInTop3"] for it in rs)), 2)}
    (RESULTS_DIR / "answers_eval.json").write_text(json.dumps({**summary, "calibrationItems": calib}, indent=1,
                                                              ensure_ascii=False), encoding="utf-8")
    with open(RESULTS_DIR / "answers_judged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "set", "category", "pipeline", "query", "gold", "top3",
                                          "goldInTop3", "label", "answer", "judgeReply", "evidence"])
        w.writeheader()
        w.writerows(items)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
