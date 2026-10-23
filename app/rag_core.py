"""Shared retrieval pipeline used by server.py and export_ui_data.py.

Mirrors notebook/rag_rerank.ipynb exactly (same cleaning, chunking, models, prompt), so the live chat
and the exported UI data match the notebook's results. The GDPR PDF and the saved results live with the
notebook; .env sits at the project root and is shared by both.
"""
import hashlib
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from dotenv import load_dotenv

from onnx_encoders import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).parent          # app/
PROJECT = ROOT.parent                  # the repository root
load_dotenv(PROJECT / ".env")

PDF_PATH = PROJECT / "notebook" / "data" / "gdpr.pdf"
RESULTS_DIR = PROJECT / "notebook" / "results"
BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS, OVERLAP_WORDS = 220, 40
TOP_K_RETRIEVE, TOP_K_FINAL = 10, 3
# Hybrid retrieval: dense and BM25 each return their top HYBRID_POOL, Reciprocal Rank Fusion merges the two
# lists (score = sum of 1 / (RRF_K + rank)), and the fused top TOP_K_RETRIEVE go to the cross-encoder, so
# re-ranking costs the same as in the dense pipeline. RRF_K = 60 is the constant from Cormack et al. (2009).
HYBRID_POOL, RRF_K = 20, 60
BM25_K1, BM25_B = 1.5, 0.75
# The encoders run on ONNX Runtime on the CPU (see onnx_encoders.py); the notebook used PyTorch, and the
# scores agree to ~1e-6. Re-ranking 10 candidates takes ~0.8 s on 8 CPU threads.
DEVICE = RERANK_DEVICE = "cpu"

# Tried in order; each Gemini model has its own free-tier daily quota.
GEMINI_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-3.7-flash", "gemini-3.6-flash",
                             "gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"] if m]

# {subject} is filled per document. For the GDPR it reproduces the notebook's prompt exactly.
PROMPT_TEMPLATE = """You are answering questions about {subject}.
Use ONLY the context passages below. If the context does not contain the answer, say so explicitly.
Quote or cite the relevant {unit} where possible. Mention any exceptions or conditions that apply.

Context:
{context}

Question: {question}

Answer:"""
GDPR_SUBJECT = "the EU General Data Protection Regulation (GDPR)"

# GDPR profile: the Official Journal's running header/footer, exactly as in the notebook.
NOISE_LINE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{4}|EN|Official Journal of the European Union|L\s?\d+/\d+)\s*$")
# Generic profile: bare page numbers such as "12", "Page 12", "12 / 40", "- 12 -".
PAGE_NUMBER_LINE = re.compile(r"^\s*[-–]?\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*[-–]?\s*$", re.I)


class EmptyDocument(ValueError):
    """The PDF has no extractable text, e.g. a scan without OCR."""


def load_pages(path=PDF_PATH, profile="gdpr"):
    """Returns [(page_number, cleaned_text)].
    profile="gdpr" strips the Official Journal header/footer (the notebook's cleaning).
    profile="generic" strips page-number lines and any short line repeated on at least 40% of pages,
    which catches running headers and footers in most documents."""
    with pymupdf.open(path) as doc:
        raw = [page.get_text().splitlines() for page in doc]
    if profile == "gdpr":
        keep = lambda line: not NOISE_LINE.match(line)   # noqa: E731
    else:
        counts = {}
        for lines in raw:
            for l in {l.strip() for l in lines if 0 < len(l.strip()) < 90}:
                counts[l] = counts.get(l, 0) + 1
        threshold = max(3, int(0.4 * len(raw)))
        repeated = {l for l, n in counts.items() if n >= threshold} if len(raw) >= 4 else set()
        keep = lambda line: not PAGE_NUMBER_LINE.match(line) and line.strip() not in repeated   # noqa: E731
    pages = []
    for i, lines in enumerate(raw, start=1):
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", " ".join(l for l in lines if keep(l)))
        pages.append((i, re.sub(r"\s+", " ", text).strip()))
    if sum(len(t) for _, t in pages) < 200:
        raise EmptyDocument("No extractable text found. Scanned PDFs need OCR before they can be indexed.")
    return pages


def chunk_pages(pages):
    """Sliding word windows with overlap; each chunk keeps its start/end page and word offset."""
    words = [(w, p) for p, t in pages for w in t.split()]
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + CHUNK_WORDS]
        if len(window) < 30 and chunks:
            break
        chunks.append({"id": len(chunks), "start": start, "page": window[0][1], "endPage": window[-1][1],
                       "text": " ".join(w for w, _ in window)})
    return chunks, words


def article_heads(words):
    """Word offsets of 'Article N Title' headings (not cross-references like 'Article 6(1)')."""
    return [(i + 1, int(words[i + 1][0])) for i in range(len(words) - 2)
            if words[i][0] == "Article" and words[i + 1][0].isdigit() and words[i + 2][0][:1].isupper()]


# Statute-style headings used in India and other Commonwealth laws: "21. Protection of life and personal
# liberty.—No person shall ...". The em dash after the title is what separates a heading from a list item.
# Amended headings start with a footnote marker: "368. 1[Power of Parliament to amend ...].—".
NUMBERED_HEAD = re.compile(r"(?:^|\s)(\d{1,3}[A-Z]{0,3})\.\s+(?:\d{1,2}\[)?[A-Z][^.]{2,150}?\.\s?[—–-]")
SCHEDULE_HEAD = re.compile(r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)"
                           r" SCHEDULE\s*\[\s*Art")   # real headings cite their articles ("FIRST SCHEDULE
                                                        # [Articles 1 and 4]"); footnotes quoting old wording do not


def numbered_citations(chunks, words):
    """Labels chunks of documents whose provisions are headed '21. Title.—'. Returns True if applied.
    The prefix is 'Art.' or 'Sec.' depending on which word the document uses more. Once the Schedules
    begin, their numbered paragraphs are not provisions, so those chunks are labelled by schedule."""
    found = [[m.group(1) for m in NUMBERED_HEAD.finditer(c["text"])] for c in chunks]
    if sum(map(len, found)) < 10:
        return False
    lower = " ".join(w for w, _ in words).lower()
    unit, many = ("Art.", "Arts.") if lower.count("article") >= lower.count("section") else ("Sec.", "Secs.")
    last, schedule = None, None
    for c, heads in zip(chunks, found):
        sched = SCHEDULE_HEAD.findall(c["text"])
        if last is not None and sched:   # headings before any article are the table of contents
            schedule = sched[-1].title()
        if schedule:
            c["cite"] = f"Sch. {schedule}"
            c["refs"] = [f"{schedule} Schedule"]
            continue
        first = NUMBERED_HEAD.search(c["text"])
        # A chunk that starts mid-provision also belongs to the provision before its first heading.
        labels = list(dict.fromkeys(([last] if last and (not heads or first.start() > 30) else []) + heads))
        if labels:
            c["cite"] = f"{unit} {labels[0]}" if len(labels) == 1 else f"{many} {labels[0]}–{labels[-1]}"
            c["refs"] = [f"{'Article' if unit == 'Art.' else 'Section'} {n}" for n in dict.fromkeys(labels)]
            last = heads[-1] if heads else last
    return True


def add_citations(chunks, words):
    """Labels each chunk for display.
    Documents with numbered 'Article N' headings (EU-style laws) get article labels, and recitals before
    an EU enacting formula get recital labels. Documents with '21. Title.—' headings (Indian and other
    Commonwealth statutes) get article or section labels. Everything else gets page labels ('p. 4').
    Returns the first chunk of the enacting terms, or None when the document has none."""
    heads = article_heads(words)
    enacting = next((c["id"] for c in chunks if "HAVE ADOPTED THIS REGULATION" in c["text"]), None)
    use_articles = len(heads) >= 5
    if not use_articles and enacting is None:
        for c in chunks:
            c["cite"] = f"p. {c['page']}" if c["page"] == c["endPage"] else f"pp. {c['page']}–{c['endPage']}"
        numbered_citations(chunks, words)
        return None
    for c in chunks:
        if enacting is not None and c["id"] < enacting:
            nums = [int(n) for n in re.findall(r"(?:^|\s)\((\d{1,3})\)\s+[A-Z]", c["text"])]
            c["cite"] = ("Recitals" if not nums else f"Recital {nums[0]}" if len(nums) == 1
                         else f"Recitals {nums[0]}–{nums[-1]}")
            continue
        if use_articles:
            end = c["start"] + CHUNK_WORDS
            arts = [a for w, a in heads if w <= c["start"]][-1:] + [a for w, a in heads if c["start"] < w < end]
            if arts:
                c["cite"] = f"Art. {arts[0]}" if len(arts) == 1 else f"Arts. {arts[0]}–{arts[-1]}"
                continue
        c["cite"] = f"p. {c['page']}" if c["page"] == c["endPage"] else f"pp. {c['page']}–{c['endPage']}"
    return enacting


TOKEN = re.compile(r"\w+")


def tokenize(text):
    """Lower-cased word and number tokens. Numbers are kept: 'Article 83' and '20 000 000' are what a dense
    encoder blurs and a keyword match catches."""
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over the chunks, as a sparse term-by-chunk weight matrix so a query is one sum per term."""

    def __init__(self, texts):
        docs = [tokenize(t) for t in texts]
        self.vocab = {}
        rows, cols, tfs = [], [], []
        for d, toks in enumerate(docs):
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, n in counts.items():
                rows.append(self.vocab.setdefault(t, len(self.vocab)))
                cols.append(d)
                tfs.append(n)
        rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, dtype="float32")
        lengths = np.array([len(t) for t in docs], dtype="float32")
        df = np.bincount(rows, minlength=len(self.vocab))
        idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype("float32")
        norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths / lengths.mean())
        weights = idf[rows] * tfs * (BM25_K1 + 1) / (tfs + norm[cols])
        # One row per term: a query's scores are the sum of its terms' rows.
        self.weights = [dict() for _ in range(len(self.vocab))]
        for r, c, w in zip(rows.tolist(), cols.tolist(), weights.tolist()):
            self.weights[r][c] = w
        self.n = len(docs)

    def scores(self, query):
        out = np.zeros(self.n, dtype="float32")
        for t in tokenize(query):
            for c, w in self.weights[self.vocab[t]].items() if t in self.vocab else ():
                out[c] += w
        return out

    def search(self, query, k):
        s = self.scores(query)
        ids = np.argsort(-s, kind="stable")[:k]
        return ids.tolist(), s[ids].tolist()


def rrf(*rankings):
    """Reciprocal Rank Fusion of ranked id lists; returns ids by fused score, best first, with the scores."""
    fused = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + r)
    order = sorted(fused, key=lambda cid: -fused[cid])
    return order, [fused[cid] for cid in order]


class Models:
    """The two encoders, loaded once and shared by every document."""

    def __init__(self):
        self.bi = SentenceTransformer(BI_ENCODER_NAME)
        self.ce = CrossEncoder(CROSS_ENCODER_NAME)


class DocIndex:
    """One PDF: cleaned pages, chunks with citations, and a FAISS index of chunk embeddings."""

    def _cache_key(self):
        """Changes whenever the embedded texts or the bi-encoder change, so a stale cache is never used."""
        h = hashlib.sha256(BI_ENCODER_NAME.encode())
        for t in self.search_texts:
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    def _load_cache(self, cache):
        if not cache or not Path(cache).exists():
            return None
        try:
            with np.load(cache) as saved:
                return saved["emb"] if str(saved["key"]) == self._cache_key() else None
        except (OSError, ValueError, KeyError):
            return None

    def __init__(self, models, path, *, title, subject=None, profile="generic", on_progress=None, cache=None):
        """cache: optional .npz path for the chunk embeddings, so a restart does not re-embed the document."""
        self.models, self.title, self.profile = models, title, profile
        self.subject = subject or f'the document "{title}"'
        report = on_progress or (lambda stage, done=0, total=0: None)
        report("reading")
        self.pages = load_pages(path, profile)
        self.chunks, words = chunk_pages(self.pages)
        self.first_article = add_citations(self.chunks, words)
        self.unit = "Article" if any(c["cite"].startswith("Art") for c in self.chunks) else "page"
        self.texts = [c["text"] for c in self.chunks]
        # Uploaded documents: embed and re-rank each chunk with a header naming the provisions it covers
        # ("Article 20 | Article 21"), because the text itself says "21. Protection of life ..." and neither
        # encoder links that to a question about "Article 21". What is shown and sent to the LLM is the
        # plain text. The GDPR case study keeps the notebook's exact setup (no headers).
        self.search_texts = self.texts if profile == "gdpr" else [
            (" | ".join(c["refs"]) + "\n" + c["text"]) if c.get("refs") else c["text"] for c in self.chunks]
        emb = self._load_cache(cache)
        if emb is None:
            parts, batch = [], 16
            for i in range(0, len(self.texts), batch):
                report("embedding", i, len(self.texts))
                parts.append(models.bi.encode(self.search_texts[i:i + batch], batch_size=batch,
                                              normalize_embeddings=True, convert_to_numpy=True).astype("float32"))
            emb = np.vstack(parts)
            if cache:
                np.savez(cache, emb=emb, key=self._cache_key())
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.bm25 = BM25(self.search_texts)
        report("ready", len(self.texts), len(self.texts))

    def retrieve_and_rerank(self, query, mode="dense"):
        """Stage 1 (top 10 candidates) and stage 2 (cross-encoder re-sort), with per-step timings.
        mode="dense": the bi-encoder's top 10, as in the notebook.
        mode="hybrid": the bi-encoder's and BM25's top 20 fused by RRF, cut to 10. biRank and biScore are then
        the fused rank and score; denseRank and bm25Rank say where each list put the chunk (None: not in its top 20)."""
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"Unknown retrieval mode {mode!r}")
        k = min(TOP_K_RETRIEVE, len(self.texts))
        pool = min(HYBRID_POOL, len(self.texts)) if mode == "hybrid" else k
        t0 = time.perf_counter()
        q = self.models.bi.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        t1 = time.perf_counter()
        scores, ids = self.index.search(q, pool)
        ids, scores = ids[0].tolist(), scores[0].tolist()
        t2 = time.perf_counter()
        extra = {}
        if mode == "hybrid":
            bm_ids, _ = self.bm25.search(query, pool)
            dense_rank = {cid: r for r, cid in enumerate(ids, start=1)}
            bm_rank = {cid: r for r, cid in enumerate(bm_ids, start=1)}
            ids, scores = rrf(ids, bm_ids)
            ids, scores = ids[:k], scores[:k]
            extra = {cid: {"denseRank": dense_rank.get(cid), "bm25Rank": bm_rank.get(cid)} for cid in ids}
        t3 = time.perf_counter()
        ce_scores = self.models.ce.predict([(query, self.search_texts[i]) for i in ids], batch_size=len(ids))
        t4 = time.perf_counter()
        ce_rank = {ids[j]: r for r, j in enumerate(np.argsort(-ce_scores), start=1)}
        candidates = [{"chunk": cid, "page": self.chunks[cid]["page"], "cite": self.chunks[cid]["cite"],
                       "biRank": r, "biScore": round(float(s), 4),
                       "ceRank": ce_rank[cid], "ceScore": round(float(ce_scores[r - 1]), 3),
                       **extra.get(cid, {}), "text": self.texts[cid]}
                      for r, (cid, s) in enumerate(zip(ids, scores), start=1)]
        ms = lambda a, b: round((b - a) * 1000, 2)   # noqa: E731
        timings = {"embedMs": ms(t0, t1), "searchMs": ms(t1, t2), "rerankMs": ms(t3, t4)}
        if mode == "hybrid":
            timings["fuseMs"] = ms(t2, t3)
        return candidates, timings

    def build_prompt(self, question, chunk_ids):
        # Uploaded documents also name each passage's provisions, so answers can cite them; the GDPR
        # prompt stays identical to the notebook's.
        label = (lambda cid: "") if self.profile == "gdpr" else (lambda cid: f" | {self.chunks[cid]['cite']}")
        context = "\n\n".join(
            f"[Passage {i + 1} | chunk {cid} | page {self.chunks[cid]['page']}{label(cid)}]\n{self.texts[cid]}"
            for i, cid in enumerate(chunk_ids))
        return PROMPT_TEMPLATE.format(subject=self.subject, unit=self.unit, context=context, question=question)


def gdpr_index(models, cache=None):
    """The built-in case-study document, processed exactly as in the notebook."""
    return DocIndex(models, PDF_PATH, title="General Data Protection Regulation (EU) 2016/679",
                    subject=GDPR_SUBJECT, profile="gdpr", cache=cache)


def Pipeline():   # noqa: N802 - kept for export_ui_data.py
    rag = gdpr_index(Models())
    rag.retrieve_and_rerank("warm-up")   # first call pays kernel start-up
    return rag


class QuotaExhausted(Exception):
    pass


class LocalUnavailable(Exception):
    pass


# llama.cpp's llama-server (running in WSL) exposes an OpenAI-compatible API on this address.
# The "local" backend is any OpenAI-compatible chat API. Unset, it is llama-server in WSL (start_local_llm.sh).
# For a hosted open model, set LOCAL_LLM_URL to the provider's base URL (e.g. https://api.groq.com/openai),
# LOCAL_LLM_KEY to its API key and LOCAL_LLM_MODEL to the model id; a trailing /v1 on the URL is fine.
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080").rstrip("/").removesuffix("/v1")
LOCAL_LLM_KEY = os.getenv("LOCAL_LLM_KEY")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
LOCAL_LLM_HOSTED = bool(LOCAL_LLM_KEY)
_local_seen = {"at": 0.0, "name": None}   # hosted status is cached: the UI polls /api/health every few seconds


def _model_name(model_id):
    """Display name: 'qwen/qwen3-32b' -> 'qwen3-32b', '/home/x/Qwen3.5-4B-Q4_K_M.gguf' -> 'Qwen3.5-4B-Q4_K_M'."""
    name = str(model_id).replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".gguf") or "local"


def _local_request(**body):
    """Headers and JSON body for a chat call; the key and model are only sent when configured."""
    headers = {"Authorization": f"Bearer {LOCAL_LLM_KEY}"} if LOCAL_LLM_KEY else {}
    if LOCAL_LLM_MODEL:
        body["model"] = LOCAL_LLM_MODEL
    return {"headers": headers, "json": body}


def local_status(timeout=0.5):
    """Returns the model's display name, or None if the endpoint is not reachable."""
    import httpx
    if LOCAL_LLM_HOSTED and time.monotonic() - _local_seen["at"] < 60:
        return _local_seen["name"]
    try:
        r = httpx.get(f"{LOCAL_LLM_URL}/v1/models", timeout=3 if LOCAL_LLM_HOSTED else timeout,
                      headers=_local_request()["headers"])
        r.raise_for_status()
        # A hosted provider lists many models; llama-server lists only the one it loaded.
        name = _model_name(LOCAL_LLM_MODEL or r.json()["data"][0]["id"])
    except Exception:  # noqa: BLE001 - any failure means "not available"
        name = None
    _local_seen.update(at=time.monotonic(), name=name)
    return name


def stream_local(prompt):
    """Streams the local model's answer. Yields ("model", name) once, then ("delta", text) pieces."""
    import json

    import httpx
    try:
        with httpx.stream("POST", f"{LOCAL_LLM_URL}/v1/chat/completions", timeout=300, **_local_request(
            messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=700, stream=True,
        )) as r:
            r.raise_for_status()
            sent_model = False
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if not sent_model:
                    sent_model = True
                    yield "model", _model_name(chunk.get("model", "local"))
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield "delta", piece
    except httpx.HTTPError as e:
        raise LocalUnavailable(f"The open-model endpoint at {LOCAL_LLM_URL} failed ({e})") from e


def stream_gemini(prompt, min_interval_s=13):
    """Streams a Gemini answer. Falls back to the next model on 404/429, but only before the first
    piece of text has been sent. Yields ("model", name) once, then ("delta", text) pieces."""
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise QuotaExhausted("No GEMINI_API_KEY in .env")
    client = genai.Client(api_key=key)
    last_error = None
    for model in GEMINI_MODELS:
        time.sleep(max(0.0, _last_call[0] + min_interval_s - time.time()))   # stay under 5 requests/min
        _last_call[0] = time.time()
        started = False
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048))
            for chunk in stream:
                if not started:
                    started = True
                    yield "model", model
                if chunk.text:
                    yield "delta", chunk.text
            if not started:
                yield "model", model
            return
        except Exception as e:  # noqa: BLE001 - the SDK raises several error types
            if started:
                raise
            last_error = e
            if getattr(e, "code", None) in (404, 429, 500, 503):
                continue
            raise
    raise QuotaExhausted(f"All Gemini models are unavailable or out of quota ({last_error})")


def generate_local(prompt):
    """Same prompt and temperature as the Gemini path, sent to the local llama.cpp server.
    Returns (text, model, ms)."""
    import httpx
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{LOCAL_LLM_URL}/v1/chat/completions", timeout=300, **_local_request(
            messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=700,
        ))
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise LocalUnavailable(f"The open-model endpoint at {LOCAL_LLM_URL} failed ({e})") from e
    body = r.json()
    text = (body["choices"][0]["message"].get("content") or "[empty response]").strip()
    return text, _model_name(body.get("model", "local")), round((time.perf_counter() - t0) * 1000)


_last_call = [0.0]


def generate(prompt, min_interval_s=13, max_output_tokens=2048):
    """Calls Gemini at temperature 0, moving to the next model when one is out of quota.
    Returns (text, model, ms) where ms covers only the successful call."""
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise QuotaExhausted("No GEMINI_API_KEY in .env")
    client = genai.Client(api_key=key)
    last_error = None
    for model in GEMINI_MODELS:
        for attempt in range(3):
            time.sleep(max(0.0, _last_call[0] + min_interval_s - time.time()))   # stay under 5 requests/min
            _last_call[0] = time.time()
            t0 = time.perf_counter()
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_output_tokens))
                return (r.text or "[empty response]").strip(), model, round((time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001 - the SDK raises several error types
                code = getattr(e, "code", None)
                last_error = e
                if code in (404, 429):      # model unavailable or out of quota: try the next model
                    break
                if code in (500, 503) and attempt < 2:
                    time.sleep(5 * 2 ** attempt)
                    continue
                raise
    raise QuotaExhausted(f"All Gemini models are unavailable or out of quota ({last_error})")
