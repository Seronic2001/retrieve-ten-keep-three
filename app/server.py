"""Live backend for the UI: a library of PDFs, each with its own index, and the two RAG pipelines.

Run from app/:  ../.venv/Scripts/python.exe -m uvicorn server:app --port 8000
The Vite dev server proxies /api to this port.
Optional local LLM: run start_local_llm.sh inside WSL (llama-server on port 8080).
"""
import json
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import case_study
import chunk_space

from rag_core import (BI_ENCODER_NAME, CHUNK_WORDS, CROSS_ENCODER_NAME, DEVICE, LOCAL_LLM_HOSTED, OVERLAP_WORDS,
                      PDF_PATH, RERANK_DEVICE, RESULTS_DIR, ROOT, TOP_K_FINAL, DocIndex, EmptyDocument,
                      LocalUnavailable, Models,
                      QuotaExhausted, gdpr_index, generate, generate_local, local_status, stream_gemini,
                      stream_local)

UPLOADS = ROOT / "uploads"
LIBRARY_FILE = UPLOADS / "library.json"
MAX_UPLOAD_MB = 50
GDPR_ID = "gdpr"

app = FastAPI(title="Retrieve 10, keep 3")
models = Models()   # both encoders, loaded once and shared by every document
studies = case_study.Studies(UPLOADS, RESULTS_DIR, GDPR_ID)


class Library:
    """Document metadata plus the in-memory index of each ready document. Uploads are indexed one at a
    time on a background thread; metadata is saved so uploads come back after a restart."""

    def __init__(self):
        self.lock = threading.Lock()
        self.meta, self.indexes = {}, {}
        self.jobs = queue.Queue()
        UPLOADS.mkdir(exist_ok=True)

        gdpr = gdpr_index(models, cache=UPLOADS / "gdpr.emb.npz")   # the Docker image ships this cache
        gdpr.retrieve_and_rerank("warm-up")   # first call pays kernel start-up
        self.indexes[GDPR_ID] = gdpr
        self.meta[GDPR_ID] = {"id": GDPR_ID, "title": gdpr.title, "filename": PDF_PATH.name, "builtin": True,
                              "status": "ready", "stage": "ready", "progress": 1.0, "pages": len(gdpr.pages),
                              "chunks": len(gdpr.chunks), "error": None, "created": None}
        if LIBRARY_FILE.exists():
            for m in json.loads(LIBRARY_FILE.read_text(encoding="utf-8")):
                if (UPLOADS / f"{m['id']}.pdf").exists():
                    self.meta[m["id"]] = {**m, "builtin": False, "status": "queued", "stage": "queued",
                                          "progress": 0.0, "pages": None, "chunks": None, "error": None}
                    self.jobs.put(m["id"])
        threading.Thread(target=self._worker, daemon=True).start()

    def _save(self):
        uploads = [{k: m[k] for k in ("id", "title", "filename", "created")}
                   for m in self.meta.values() if not m["builtin"]]
        LIBRARY_FILE.write_text(json.dumps(uploads, indent=1), encoding="utf-8")

    def _update(self, doc_id, **change):
        with self.lock:
            if doc_id in self.meta:
                self.meta[doc_id].update(change)

    def _worker(self):
        while True:
            doc_id = self.jobs.get()
            if doc_id not in self.meta:
                continue   # deleted while queued

            def progress(stage, done=0, total=0, doc_id=doc_id):
                frac = 0.05 if stage == "reading" else 1.0 if stage == "ready" else 0.1 + 0.9 * done / max(total, 1)
                self._update(doc_id, status="indexing", stage=stage, progress=round(frac, 3))
            try:
                idx = DocIndex(models, UPLOADS / f"{doc_id}.pdf", title=self.meta[doc_id]["title"],
                               on_progress=progress, cache=UPLOADS / f"{doc_id}.emb.npz")
                with self.lock:
                    if doc_id in self.meta:
                        self.indexes[doc_id] = idx
                        self.meta[doc_id].update(status="ready", stage="ready", progress=1.0,
                                                 pages=len(idx.pages), chunks=len(idx.chunks))
            except EmptyDocument as e:
                self._update(doc_id, status="error", error=str(e))
            except Exception as e:  # noqa: BLE001 - a broken PDF must not stop the worker
                self._update(doc_id, status="error", error=f"Could not read this PDF ({e})")

    def add(self, filename, data):
        doc_id = uuid.uuid4().hex[:10]
        path = UPLOADS / f"{doc_id}.pdf"
        path.write_bytes(data)
        title = re.sub(r"[_\-]+", " ", re.sub(r"\.pdf$", "", filename, flags=re.I)).strip() or "Untitled"
        try:   # prefer the PDF's own title when it has a sensible one
            with pymupdf.open(path) as doc:
                meta_title = ((doc.metadata or {}).get("title") or "").strip()
                pages = doc.page_count
        except Exception as e:  # noqa: BLE001
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"This file is not a readable PDF ({e})") from e
        if 4 <= len(meta_title) <= 120 and not meta_title.lower().startswith(("microsoft", "untitled")):
            title = meta_title
        with self.lock:
            self.meta[doc_id] = {"id": doc_id, "title": title, "filename": filename, "builtin": False,
                                 "status": "queued", "stage": "queued", "progress": 0.0, "pages": pages,
                                 "chunks": None, "error": None,
                                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self._save()
            out = dict(self.meta[doc_id])
        self.jobs.put(doc_id)
        return out

    def delete(self, doc_id):
        with self.lock:
            m = self.meta.get(doc_id)
            if not m:
                raise HTTPException(404, "Unknown document")
            if m["builtin"]:
                raise HTTPException(400, "The built-in case study cannot be deleted")
            del self.meta[doc_id]
            self.indexes.pop(doc_id, None)
            self._save()
        (UPLOADS / f"{doc_id}.pdf").unlink(missing_ok=True)
        (UPLOADS / f"{doc_id}.emb.npz").unlink(missing_ok=True)
        chunk_space.forget(doc_id)
        studies.forget(doc_id)

    def index(self, doc_id):
        with self.lock:
            m = self.meta.get(doc_id)
            idx = self.indexes.get(doc_id)
        if not m:
            raise HTTPException(404, "Unknown document")
        if not idx:
            raise HTTPException(409, f"This document is not ready yet (status: {m['status']})")
        return idx

    def listing(self):
        with self.lock:
            out = [dict(m) for m in self.meta.values()]
        for m in out:
            m["study"] = studies.status(m["id"])["state"]
        return out


library = Library()


class Question(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    doc: str = GDPR_ID
    # "hybrid" fuses the bi-encoder's and BM25's top 20 before re-ranking (see rag_core.retrieve_and_rerank)
    retrieval: Literal["dense", "hybrid"] = "dense"


class AnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    doc: str = GDPR_ID
    baseline: list[int] = Field(min_length=1, max_length=TOP_K_FINAL)
    reranked: list[int] = Field(min_length=1, max_length=TOP_K_FINAL)
    backend: Literal["gemini", "local"] = "gemini"
    # which answers to generate, in this order; the UI asks for the re-ranked one first and the
    # baseline only when the reader wants to compare
    pipelines: list[Literal["reranked", "baseline"]] = Field(default=["reranked", "baseline"], min_length=1, max_length=2)


@app.get("/api/health")
def health():
    return {"ok": True, "device": DEVICE, "rerankDevice": RERANK_DEVICE,
            "models": {"bi": BI_ENCODER_NAME, "cross": CROSS_ENCODER_NAME},
            "chunkWords": CHUNK_WORDS, "overlapWords": OVERLAP_WORDS, "local": local_status(),
            "localHosted": LOCAL_LLM_HOSTED,
            "maxUploadMb": MAX_UPLOAD_MB}


@app.get("/api/documents")
def list_documents():
    return library.listing()


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Only PDF files can be added")
    return library.add(file.filename or "document.pdf", data)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    library.delete(doc_id)
    return {"ok": True}


@app.get("/api/documents/{doc_id}/map")
def document_map(doc_id: str):
    """What the chunk-flow page needs: every chunk's page and citation, and three consecutive
    chunks from the middle of the document to illustrate the overlapping windows."""
    idx = library.index(doc_id)
    mid = max(1, min(len(idx.chunks) - 2, len(idx.chunks) // 2))
    example = [{"chunk": c["id"], "page": c["page"], "cite": c["cite"], "text": c["text"]}
               for c in idx.chunks[mid - 1:mid + 2]]
    return {"chunks": [{"page": c["page"], "cite": c["cite"]} for c in idx.chunks], "example": example,
            "firstArticleChunk": idx.first_article, "pages": len(idx.pages),
            "chunkWords": CHUNK_WORDS, "overlapWords": OVERLAP_WORDS}


class Locate(BaseModel):
    query: str = Field(min_length=3, max_length=500)


@app.get("/api/documents/{doc_id}/space")
def document_space(doc_id: str):
    """Every chunk on two 2D maps of the embedding space (t-SNE and PCA) plus named topic groups."""
    idx = library.index(doc_id)
    return chunk_space.payload(chunk_space.build(doc_id, idx), idx)


@app.post("/api/documents/{doc_id}/space/locate")
def locate_in_space(doc_id: str, q: Locate):
    """Where a question lands on both maps, its similarity to every chunk, and both pipelines' rankings."""
    idx = library.index(doc_id)
    space = chunk_space.build(doc_id, idx)
    candidates, timings = idx.retrieve_and_rerank(q.query.strip())
    return {**chunk_space.locate(space, idx, q.query.strip()), "candidates": candidates, "timings": timings}


@app.get("/api/documents/{doc_id}/chunks/{chunk}/neighbours")
def chunk_neighbours(doc_id: str, chunk: int):
    """One chunk's full text and its nearest chunks in the full embedding space."""
    idx = library.index(doc_id)
    if not 0 <= chunk < len(idx.chunks):
        raise HTTPException(404, "Unknown chunk")
    space = chunk_space.build(doc_id, idx)
    c = idx.chunks[chunk]
    return {"chunk": chunk, "cite": c["cite"], "page": c["page"], "text": idx.texts[chunk],
            "topic": int(space["labels"][chunk]), "neighbours": chunk_space.neighbours(space, idx, chunk)}


class BuildStudy(BaseModel):
    backend: Literal["gemini", "local"] = "gemini"


@app.get("/api/documents/{doc_id}/study")
def get_study(doc_id: str):
    """This document's generated case study when built, else its build state and progress.
    For the GDPR (whose study is the saved notebook run in the UI): the questions added from its failure hunt."""
    library.index(doc_id)
    return studies.get(doc_id)


@app.get("/api/documents/{doc_id}/hunt")
def get_hunt(doc_id: str):
    """This document's failure hunt when run, else its state and progress."""
    library.index(doc_id)
    return studies.get(doc_id, "hunt")


@app.post("/api/documents/{doc_id}/hunt")
def run_hunt(doc_id: str, req: BuildStudy):
    """Starts (or redoes) a failure hunt: a few hundred generated questions scored on both pipelines."""
    idx = library.index(doc_id)
    if not studies.start(doc_id, idx, req.backend, "hunt"):
        raise HTTPException(409, "A failure hunt is already running for this document")
    return studies.status(doc_id, "hunt")


class AddFromHunt(BaseModel):
    index: int = Field(ge=0)


@app.post("/api/documents/{doc_id}/study/add")
def add_from_hunt(doc_id: str, req: AddFromHunt):
    """Adds one hunted question to the case study, with both answers, so it appears in the replay."""
    idx = library.index(doc_id)
    try:
        q, note = studies.add_from_hunt(doc_id, idx, req.index)
    except KeyError as e:
        raise HTTPException(404, "Unknown hunted question") from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {"question": q, "note": note}


@app.post("/api/documents/{doc_id}/study")
def build_study(doc_id: str, req: BuildStudy):
    """Starts (or redoes) the case study build: LLM-written test questions scored on both pipelines."""
    if doc_id == GDPR_ID:
        raise HTTPException(400, "The GDPR case study uses the hand-labelled notebook questions")
    idx = library.index(doc_id)
    if not studies.start(doc_id, idx, req.backend):
        raise HTTPException(409, "This case study is already being built")
    return studies.status(doc_id)


@app.post("/api/retrieve")
def retrieve(q: Question):
    """Both retrieval stages; fast, no LLM."""
    candidates, timings = library.index(q.doc).retrieve_and_rerank(q.query.strip(), q.retrieval)
    return {"query": q.query.strip(), "doc": q.doc, "retrieval": q.retrieval, "candidates": candidates,
            "timings": timings}


def _checked(req):
    idx = library.index(req.doc)
    for cid in req.baseline + req.reranked:
        if not 0 <= cid < len(idx.chunks):
            raise HTTPException(400, f"Unknown chunk id {cid}")
    return idx


@app.post("/api/answer")
def answer(req: AnswerRequest):
    """Answer 1 from the baseline top 3 and Answer 2 from the re-ranked top 3, same prompt."""
    idx = _checked(req)
    llm = generate_local if req.backend == "local" else generate
    out = {}
    try:
        for name, ids in (("baseline", req.baseline), ("reranked", req.reranked)):
            text, model, ms = llm(idx.build_prompt(req.query.strip(), ids))
            out[name] = {"text": text, "model": model, "ms": ms}
    except QuotaExhausted as e:
        raise HTTPException(429, str(e)) from e
    except LocalUnavailable as e:
        raise HTTPException(503, str(e)) from e
    return out


@app.post("/api/answer/stream")
def answer_stream(req: AnswerRequest):
    """Streams the requested answers (default: both, re-ranked first) as newline-delimited JSON events:
    {"type": "start" | "delta" | "end", "pipeline": ..., ...}, then {"type": "done"}.
    Failures arrive as {"type": "error", "status": 429 | 503 | 500, "message": ...}."""
    idx = _checked(req)
    streamer = stream_local if req.backend == "local" else stream_gemini

    def events():
        event = lambda **e: json.dumps(e) + "\n"   # noqa: E731
        try:
            for name in dict.fromkeys(req.pipelines):
                ids = req.reranked if name == "reranked" else req.baseline
                t0, first = time.perf_counter(), None
                for kind, value in streamer(idx.build_prompt(req.query.strip(), ids)):
                    if kind == "model":
                        yield event(type="start", pipeline=name, model=value)
                    else:
                        first = first or time.perf_counter()
                        yield event(type="delta", pipeline=name, text=value)
                end = time.perf_counter()
                yield event(type="end", pipeline=name, ms=round((end - t0) * 1000),
                            firstTokenMs=round(((first or end) - t0) * 1000))
            yield event(type="done")
        except QuotaExhausted as e:
            yield event(type="error", status=429, message=str(e))
        except LocalUnavailable as e:
            yield event(type="error", status=503, message=str(e))
        except Exception as e:  # noqa: BLE001 - report anything else to the client instead of a cut stream
            yield event(type="error", status=500, message=str(e))

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# A production build of the UI (npm run build, as in the Hugging Face Space image) is served from the same
# origin, so its relative api/ calls reach the routes above. In development Vite serves the UI instead.
UI_DIST = ROOT / "ui" / "dist"
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
