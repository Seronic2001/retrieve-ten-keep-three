"""Turns any indexed document into a case study like the GDPR one, without hand labelling.

An LLM writes test questions from passages spread across the document, each in one of the GDPR study's
failure styles, and copies the words that answer it. The source chunk, plus a neighbour whose overlap holds
those words, counts as the answer, so both pipelines are scored as in the notebook: where the answer
sat in the bi-encoder top 10, where re-ranking moved it, and whether each top 3 holds it. Answers are
generated only where the two top-3 contexts differ on that, and latency is measured on this machine.
The result has the same shape as ui/public/data.json and is saved next to the PDF.

The failure hunt runs the same scoring over a few hundred generated questions, to find the rare cases where
re-ranking pushes the answer out of the top 3 (and the commoner ones where it rescues it).
"""
import csv
import json
import re
import threading
import time
from datetime import datetime, timezone

import numpy as np

from rag_core import (BI_ENCODER_NAME, CHUNK_WORDS, CROSS_ENCODER_NAME, OVERLAP_WORDS, RERANK_DEVICE, TOP_K_FINAL,
                      LocalUnavailable, QuotaExhausted, generate, generate_local)

N_QUESTIONS = 12
MAX_ANSWERED = 4        # questions that get both answers; each costs two LLM calls
FULL_SCAN_SAMPLE = 256  # chunks scored by the cross-encoder to time a full scan
WRITING_TOKENS = 8192   # Gemini reply budget for a batch of questions (an answer needs far less)

STYLES = {
    "keyword_trap": "uses everyday words instead of the passage's own terms, so matching keywords would not find it",
    "exception": "asks about a condition, limit or exception that the passage states",
    "negation": "asks when something does NOT apply, is NOT allowed or is NOT required",
    "buried": "asks for one specific detail (a number, deadline, amount or name) mentioned in passing",
    "control": "asks plainly, using the passage's own words",
}

PROMPT = """You are writing test questions for a search system over {subject}.

{task}
Each question must be answerable from its passage alone and must make sense to a reader who has not seen the
passage: never write "this passage", "the text" or "the above". Question styles:
{styles}

For each question, copy into "evidence" the words from the passage that answer it: 4 to 15 consecutive words,
exactly as they appear.
Reply with JSON only, a list of objects:
[{{"passage": <passage number>, "style": "<style>", "question": "...", "evidence": "..."}}]

{passages}"""

BATCH_TASK = ("Below are numbered passages from the document. Write one question for each of {n} different "
              "passages. Choose passages that state a clear, specific fact; skip tables of contents, indexes, "
              "lists of headings and boilerplate. Use every style at least twice.")
SINGLE_TASK = 'Below is one passage from the document. Write one question about it in the style "{style}".'


def norm(s):
    s = s.lower().translate(str.maketrans("‘’“”", "''\"\""))
    return re.sub(r"\s+", " ", s).strip()


def _eligible(idx):
    """Chunks worth asking about: mostly prose, not a table of contents, and past the cover pages."""
    skip = len(idx.chunks) // 30 if len(idx.chunks) > 30 else 0
    out = []
    for c in idx.chunks[skip:]:
        words = c["text"].split()
        alpha = sum(w.isalpha() for w in words) / max(1, len(words))
        if alpha >= 0.6 and c["text"].count("....") < 3 and len(words) >= 80:
            out.append(c["id"])
    return out or [c["id"] for c in idx.chunks]


def _spread(ids, n):
    """n ids evenly spaced through the list."""
    if len(ids) <= n:
        return list(ids)
    return [ids[round(i * (len(ids) - 1) / (n - 1))] for i in range(n)]


def _parse(text):
    """The JSON list in an LLM reply, tolerating code fences, a single object or text around it."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_, close in (("[", "]"), ("{", "}")):
        a, b = text.find(open_), text.rfind(close)
        if a >= 0 and b > a:
            try:
                got = json.loads(text[a:b + 1])
                return got if isinstance(got, list) else [got]
            except json.JSONDecodeError:
                continue
    # A reply cut off at the output limit is not valid JSON as a whole; keep every complete object in it.
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def _snippet(evidence, source):
    """The evidence as it appears in the source chunk; if the model paraphrased, its longest run of
    4+ consecutive words that does appear. None when nothing matches."""
    ev = norm(evidence).strip(" .,;:\"'")
    if len(ev.split()) >= 3 and ev in source:
        return ev
    words = ev.split()
    for size in range(min(len(words), 15), 3, -1):
        for start in range(len(words) - size + 1):
            piece = " ".join(words[start:start + size]).strip(" .,;:\"'")
            if piece in source:
                return piece
    return None


def write_questions(idx, llm, report):
    """Asks the LLM for test questions. Gemini gets all passages in one call; the local model (2k context)
    gets one passage per call. Returns ([(style, question, source_chunk, evidence)], model)."""
    styles = "\n".join(f"- {k}: {v}" for k, v in STYLES.items())
    block = lambda cid: f"[Passage {cid}]\n{idx.texts[cid]}"   # noqa: E731
    model, raw = None, []
    if llm is generate:
        pool = _spread(_eligible(idx), 2 * N_QUESTIONS)
        report("writing", 0, 1)
        text, model, _ = llm(PROMPT.format(subject=idx.subject, task=BATCH_TASK.format(n=N_QUESTIONS), styles=styles,
                                           passages="\n\n".join(block(c) for c in pool)),
                             max_output_tokens=WRITING_TOKENS)
        raw = [(item, set(pool)) for item in _parse(text)]
    else:
        pool = _spread(_eligible(idx), N_QUESTIONS)
        names = list(STYLES)
        for i, cid in enumerate(pool):
            report("writing", i, len(pool))
            text, model, _ = llm(PROMPT.format(subject=idx.subject, task=SINGLE_TASK.format(style=names[i % len(names)]),
                                               styles=styles, passages=block(cid)))
            raw += [({**item, "passage": cid}, {cid}) for item in _parse(text)[:1]]
    out, seen = [], set()
    for item, allowed in raw:
        try:
            cid, question = int(item["passage"]), str(item["question"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        style = item.get("style") if item.get("style") in STYLES else "control"
        if cid not in allowed or len(question) < 8 or norm(question) in seen:
            continue
        seen.add(norm(question))
        out.append((style, question, cid, str(item.get("evidence") or "")))
    return out[:N_QUESTIONS], model


def _status(bi, ce):
    in_b, in_c = bi is not None and bi <= TOP_K_FINAL, ce is not None and ce <= TOP_K_FINAL
    return "restored" if in_c and not in_b else "degraded" if in_b and not in_c else "both-hit" if in_b else "both-miss"


def score(idx, nt, style, question, source, evidence):
    """Runs one test question through both pipelines and records where its answer chunk landed.
    Returns (question record, timings)."""
    snippet = _snippet(evidence, nt[source])
    # Overlap repeats text only in the adjacent chunks; the same words further away (a stock phrase such as
    # "not less than two-thirds of the members") are a different passage, not the answer.
    gold = sorted({source} | ({j for j in (source - 1, source + 1) if 0 <= j < len(nt) and snippet in nt[j]}
                              if snippet else set()))
    candidates, t = idx.retrieve_and_rerank(question)
    for c in candidates:
        c["gold"] = c["chunk"] in gold
    ranks = lambda key: min((c[key] for c in candidates if c["gold"]), default=None)   # noqa: E731
    bi, ce = ranks("biRank"), ranks("ceRank")
    return {"category": style, "query": question, "goldSnippets": [snippet] if snippet else [],
            "gold": gold, "source": source, "baselineRank": bi, "rerankedRank": ce,
            "status": _status(bi, ce), "candidates": candidates}, t


def answer_both(idx, q, llm):
    """Answer 1 (bi-encoder top 3) and Answer 2 (cross-encoder top 3), same prompt. Returns (answers, model, [ms])."""
    top = lambda key: [c["chunk"] for c in sorted(q["candidates"], key=lambda c: c[key])[:TOP_K_FINAL]]   # noqa: E731
    answers, model, times = {}, None, []
    for name, key in (("baseline", "biRank"), ("reranked", "ceRank")):
        answers[name], model, ms = llm(idx.build_prompt(q["query"], top(key)))
        times.append(ms)
    return answers, model, times


def build(idx, *, backend="gemini", report=lambda stage, done=0, total=0: None):
    """The full case study for one document, in the shape of data.json."""
    llm = generate_local if backend == "local" else generate
    written, question_model = write_questions(idx, llm, report)
    if not written:
        raise ValueError("The language model did not return any usable questions. Try again, or use the other model.")

    nt = [norm(t) for t in idx.texts]
    idx.retrieve_and_rerank("warm-up")
    queries, timings = [], []
    for i, (style, question, source, evidence) in enumerate(written):
        report("retrieving", i, len(written))
        q, t = score(idx, nt, style, question, source, evidence)
        queries.append(q)
        timings.append(t)
    order = {"restored": 0, "degraded": 1, "both-miss": 2, "both-hit": 3}
    queries.sort(key=lambda q: order[q["status"]])

    # Both answers where the two contexts differ on holding the answer; at least one question, to time the LLM.
    targets = [q for q in queries if q["status"] in ("restored", "degraded")][:MAX_ANSWERED] or queries[:1]
    llm_ms, answer_model, note = [], None, None
    for i, q in enumerate(targets):
        report("answering", i, len(targets))
        try:
            q["answers"], answer_model, ms = answer_both(idx, q, llm)
            llm_ms += ms
        except (QuotaExhausted, LocalUnavailable) as e:
            note = f"Answers stopped early: {e}"
            break

    report("timing", 0, 1)
    sample = idx.search_texts[:FULL_SCAN_SAMPLE]
    t0 = time.perf_counter()
    idx.models.ce.predict([(queries[0]["query"], t) for t in sample], batch_size=32)
    per_pair = (time.perf_counter() - t0) * 1000 / len(sample)
    n = len(idx.chunks)
    mid = max(1, min(n - 2, n // 2))
    return {
        "document": {"title": idx.title, "source": None, "pages": len(idx.pages), "chunks": n,
                     "firstArticleChunk": idx.first_article, "chunkWords": CHUNK_WORDS, "overlapWords": OVERLAP_WORDS},
        "chunkIndex": [{"page": c["page"], "cite": c["cite"]} for c in idx.chunks],
        "chunkingExample": [{"chunk": c["id"], "page": c["page"], "cite": c["cite"], "text": c["text"]}
                            for c in idx.chunks[mid - 1:mid + 2]],
        "models": {"bi": BI_ENCODER_NAME.split("/")[-1], "cross": CROSS_ENCODER_NAME.split("/")[-1],
                   "llm": answer_model or question_model, "device": RERANK_DEVICE},
        "queries": queries,
        "latency": {"retrieveMs": float(np.mean([t["embedMs"] + t["searchMs"] for t in timings])),
                    "rerankMs": float(np.mean([t["rerankMs"] for t in timings])),
                    "llmMs": float(np.mean(llm_ms)) if llm_ms else None,
                    "runs": len(timings)},
        "scaling": {"corpus": n, "fullScanMs": per_pair * n, "perPairMs": per_pair},
        "sweep": None,
        "generated": {"questionsBy": question_model, "backend": backend, "note": note,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


# ── failure hunt ─────────────────────────────────────────────────────────────

HUNT_PASSAGES = {"gemini": 120, "local": 20}   # main passages; two questions each
HUNT_GROUP = 10   # main passages per Gemini call, each sent with its similar passage (about 20 questions per reply)
TRAP = ("borrows a distinctive phrase from the SIMILAR passage (words the main passage does not use) while asking "
        "about a fact that only the main passage states")

HUNT_PROMPT = """You are writing test questions for a search system over {subject}.

Below, each main passage is followed by a similar passage from elsewhere in the document. For EVERY main passage,
write two questions that only the main passage answers:
1. one in a style of your choice from keyword_trap, exception, negation, buried and control (vary them);
2. one in the style "trap".
Questions must make sense to a reader who has not seen the passages: never write "this passage" or "the text".
Never ask about the similar passages. Styles:
{styles}

For each question, copy into "evidence" the words from the MAIN passage that answer it: 4 to 15 consecutive
words, exactly as they appear.
Reply with JSON only, a list of objects:
[{{"passage": <main passage number>, "style": "<style>", "question": "...", "evidence": "..."}}]

{passages}"""


def _tally(qs):
    count = lambda s: sum(q["status"] == s for q in qs)   # noqa: E731
    return {"questions": len(qs), "restored": count("restored"), "degraded": count("degraded"),
            "bothHit": count("both-hit"), "bothMiss": count("both-miss")}


def hunt(idx, *, backend="gemini", report=lambda stage, done=0, total=0: None):
    """A large question sweep looking for the cases where re-ranking helps or hurts.
    Each passage gets an ordinary question and a "trap" question that borrows wording from its most similar
    non-overlapping passage: the pattern behind the GDPR's broken case, where the cross-encoder preferred a
    passage repeating the question's phrase. The two kinds are tallied apart, so the ordinary questions still
    give an unbiased failure rate. Only rescued and broken questions keep the chunk texts."""
    llm = generate_local if backend == "local" else generate
    pool = _spread(_eligible(idx), HUNT_PASSAGES[backend])
    emb = idx.index.reconstruct_n(0, idx.index.ntotal)

    def similar(cid):
        sims = emb @ emb[cid]
        sims[max(0, cid - 1):cid + 2] = -np.inf   # itself and its overlapping neighbours
        return int(np.argmax(sims))

    group = HUNT_GROUP if backend == "gemini" else 1   # the local model has a 2k context
    batches = [pool[i:i + group] for i in range(0, len(pool), group)]
    styles = "\n".join(f"- {k}: {v}" for k, v in STYLES.items()) + f"\n- trap: {TRAP}"
    written, seen, model, note, empty = [], set(), None, None, 0
    for b, batch in enumerate(batches):
        report("writing", b, len(batches))
        passages = "\n\n".join(f"[Passage {c}]\n{idx.texts[c]}\n\n[Similar passage, do not ask about it]\n"
                               f"{idx.texts[similar(c)]}" for c in batch)
        try:
            text, model, _ = llm(HUNT_PROMPT.format(subject=idx.subject, styles=styles, passages=passages),
                                 **({"max_output_tokens": WRITING_TOKENS} if llm is generate else {}))
        except (QuotaExhausted, LocalUnavailable) as e:
            if not written:
                raise
            note = f"Question writing stopped after {b} of {len(batches)} batches ({e})."
            break
        before = len(written)
        for item in _parse(text):
            try:
                cid = batch[0] if len(batch) == 1 else int(item["passage"])
                question = str(item["question"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            style = item.get("style") if item.get("style") in (*STYLES, "trap") else "control"
            if cid not in batch or len(question) < 8 or norm(question) in seen:
                continue
            seen.add(norm(question))
            written.append((style, question, cid, str(item.get("evidence") or "")))
        empty += len(written) == before
    if not written:
        raise ValueError("The language model did not return any usable questions. Try again, or use the other model.")
    if empty:
        note = " ".join(filter(None, [note, f"{empty} of {len(batches)} batches returned no usable questions."]))

    nt = [norm(t) for t in idx.texts]
    idx.retrieve_and_rerank("warm-up")
    questions = []
    for i, (style, question, source, evidence) in enumerate(written):
        report("retrieving", i, len(written))
        q, _ = score(idx, nt, style, question, source, evidence)
        if q["status"] not in ("restored", "degraded"):
            q["candidates"] = [{k: v for k, v in c.items() if k != "text"} for c in q["candidates"]]
        questions.append(q)
    return {
        "summary": {"all": _tally(questions),
                    "natural": _tally([q for q in questions if q["category"] != "trap"]),
                    "trap": _tally([q for q in questions if q["category"] == "trap"])},
        "questions": questions,
        "generated": {"questionsBy": model, "backend": backend, "passages": len(pool), "note": note,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


def _top3(q, key):
    return [c["chunk"] for c in sorted(q["candidates"], key=lambda c: c[key])[:TOP_K_FINAL]]


def hunt_csv(result, path):
    """One row per hunted question, for the report and for re-checking without the UI."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["style", "query", "source_chunk", "gold_chunks", "evidence", "baseline_rank", "reranked_rank",
                    "status", "baseline_top3", "reranked_top3"])
        for q in result["questions"]:
            w.writerow([q["category"], q["query"], q["source"], q["gold"], (q["goldSnippets"] or [""])[0],
                        q["baselineRank"], q["rerankedRank"], q["status"], _top3(q, "biRank"), _top3(q, "ceRank")])


# ── jobs and files ───────────────────────────────────────────────────────────

class Studies:
    """Generated case studies and failure hunts, built on background threads and saved as files:
    <id>.study.json and <id>.hunt.json (+ .csv) next to uploaded PDFs. The built-in GDPR's study is the saved
    notebook run in the UI; its hunt and the questions added from it go to notebook/results/ as gdpr_hunt.json/.csv and
    gdpr_added.json."""

    def __init__(self, folder, results, builtin):
        self.folder, self.results, self.builtin = folder, results, builtin
        self.lock = threading.Lock()
        self.jobs = {}   # (doc_id, kind) -> {"state": "building" | "error", "stage", "done", "total", "error"}

    def _path(self, doc_id, kind, ext="json"):
        if doc_id == self.builtin:
            return self.results / f"{doc_id}_{kind}.{ext}"
        return self.folder / f"{doc_id}.{kind}.{ext}"

    def _load(self, doc_id, kind):
        path = self._path(doc_id, kind)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _save(self, doc_id, kind, value):
        self._path(doc_id, kind).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def status(self, doc_id, kind="study"):
        with self.lock:
            job = self.jobs.get((doc_id, kind))
        if job:
            return dict(job)
        if kind == "study" and doc_id == self.builtin:
            return {"state": "ready"}
        return {"state": "ready" if self._path(doc_id, kind).exists() else "none"}

    def get(self, doc_id, kind="study"):
        out = self.status(doc_id, kind)
        if kind == "study" and doc_id == self.builtin:
            out["added"] = self._load(doc_id, "added") or []
        else:
            saved = self._load(doc_id, kind)   # also while a rebuild runs or after it failed
            if saved is not None:
                out[kind] = saved
        return out

    def start(self, doc_id, idx, backend, kind="study"):
        key = (doc_id, kind)
        with self.lock:
            if self.jobs.get(key, {}).get("state") == "building":
                return False
            self.jobs[key] = {"state": "building", "stage": "writing", "done": 0, "total": 1, "backend": backend}

        def report(stage, done=0, total=0):
            with self.lock:
                if key in self.jobs:
                    self.jobs[key].update(stage=stage, done=done, total=total)

        def run():
            try:
                result = (build if kind == "study" else hunt)(idx, backend=backend, report=report)
                with self.lock:
                    self._save(doc_id, kind, result)
                    if kind == "hunt":
                        hunt_csv(result, self._path(doc_id, kind, "csv"))
                    self.jobs.pop(key, None)
            except Exception as e:  # noqa: BLE001 - reported to the UI instead of lost on the thread
                msg = str(e) if isinstance(e, (ValueError, QuotaExhausted, LocalUnavailable)) else f"Build failed ({e})"
                with self.lock:
                    if key in self.jobs:
                        self.jobs[key] = {"state": "error", "error": msg, "backend": backend}

        threading.Thread(target=run, daemon=True).start()
        return True

    def add_from_hunt(self, doc_id, idx, i):
        """Moves hunted question i into the case study, with both answers, so it can be replayed.
        Returns (question, note); note says why answers are missing, if they are."""
        hunted = self._load(doc_id, "hunt") or {"questions": []}
        if not 0 <= i < len(hunted["questions"]):
            raise KeyError(i)
        if doc_id != self.builtin and not self._path(doc_id, "study").exists():
            raise ValueError("Build this document's case study first")
        src = hunted["questions"][i]
        nt = [norm(t) for t in idx.texts]
        q, _ = score(idx, nt, src["category"], src["query"], src["source"], (src["goldSnippets"] or [""])[0])
        q["fromHunt"] = True
        backend = (hunted.get("generated") or {}).get("backend", "gemini")
        note = None
        try:
            q["answers"], _, _ = answer_both(idx, q, generate_local if backend == "local" else generate)
        except (QuotaExhausted, LocalUnavailable) as e:
            note = f"Added without answers: {e}"
        same = lambda a: norm(a["query"]) == norm(q["query"])   # noqa: E731
        with self.lock:
            if doc_id == self.builtin:
                self._save(doc_id, "added", [a for a in self._load(doc_id, "added") or [] if not same(a)] + [q])
            else:
                study = self._load(doc_id, "study")
                study["queries"] = [a for a in study["queries"] if not same(a)] + [q]
                self._save(doc_id, "study", study)
            hunted = self._load(doc_id, "hunt")
            hunted["questions"][i]["added"] = True
            self._save(doc_id, "hunt", hunted)
        return q, note

    def forget(self, doc_id):
        with self.lock:
            for key in [k for k in self.jobs if k[0] == doc_id]:
                self.jobs.pop(key)
        for kind, ext in (("study", "json"), ("hunt", "json"), ("hunt", "csv")):
            self._path(doc_id, kind, ext).unlink(missing_ok=True)
