"""Structure-aware chunking: chunks follow the document's provisions instead of fixed word windows.

Each provision (an EU-style 'Article N Title', a recital '(N)', or a Commonwealth-style '21. Title.—') is a unit.
A unit that fits in CHUNK_WORDS is one chunk; consecutive small units are packed together up to that size, never
splitting one. A longer unit is split at its numbered paragraphs ('1.', '(1)'), packing paragraphs up to the
size, and a single paragraph longer than that falls back to overlapping windows. Chunk sizes therefore stay
comparable to the sliding windows in rag_core.chunk_pages, and the comparison measures boundaries, not size.

Each chunk carries 'head': its provision headings (e.g. 'Article 30 Records of processing activities'), which
DocIndex can put in front of the text that is embedded and re-ranked.
"""
import bisect
import re

from rag_core import CHUNK_WORDS, NUMBERED_HEAD, OVERLAP_WORDS, SCHEDULE_HEAD, article_heads

RECITAL = re.compile(r"^\((\d{1,3})\)$")
PARAGRAPH = re.compile(r"^(\d{1,2}\.|\(\d{1,2}[A-Z]?\))$")   # '1.' (EU paragraphs) or '(1)' (Indian clauses)
TITLE_WORDS = 12


def _word_offsets(words):
    """Character offset of each word in ' '.join(words), for mapping regex matches back to words."""
    offsets, pos = [], 0
    for w, _ in words:
        offsets.append(pos)
        pos += len(w) + 1
    return offsets


def provision_heads(words):
    """[(word_index, heading)] for every provision start, or [] when the document has no recognisable structure."""
    heads = []
    enacting = next((i for i in range(len(words) - 3) if words[i][0] == "HAVE" and words[i + 1][0] == "ADOPTED"),
                    None)
    arts = article_heads(words)
    if len(arts) >= 5:   # EU style: recitals, then 'Article N Title'
        for i in range(enacting or 0):
            if RECITAL.match(words[i][0]) and words[i + 1][0][:1].isupper():
                heads.append((i, f"Recital {RECITAL.match(words[i][0]).group(1)}"))
        if enacting is not None:
            heads.append((enacting, "Enacting terms"))
        for pos, num in arts:
            title = []
            for w, _ in words[pos + 1:pos + 1 + TITLE_WORDS]:
                if PARAGRAPH.match(w):
                    break
                title.append(w)
            heads.append((pos - 1, f"Article {num} {' '.join(title)}".strip()))
        return sorted(heads)
    text = " ".join(w for w, _ in words)
    found = list(NUMBERED_HEAD.finditer(text))
    if len(found) < 10:
        return []
    offsets = _word_offsets(words)
    to_word = lambda ch: bisect.bisect_left(offsets, ch)   # noqa: E731
    lower = text.lower()
    unit = "Article" if lower.count("article") >= lower.count("section") else "Section"
    for m in found:
        title = re.sub(r"\s+", " ", m.group(0)).strip().rstrip("—–-").strip()
        heads.append((to_word(m.start(1)), f"{unit} {title}"))
    for m in SCHEDULE_HEAD.finditer(text):
        heads.append((to_word(m.start()), f"{m.group(1).title()} Schedule"))
    return sorted(set(heads))


def _windows(start, end):
    """Overlapping windows over [start, end), as in chunk_pages."""
    step = CHUNK_WORDS - OVERLAP_WORDS
    out = []
    for s in range(start, end, step):
        e = min(s + CHUNK_WORDS, end)
        out.append((s, e))
        if e == end:
            break
    return out


def _split_unit(words, start, end):
    """[(start, end)] pieces of one provision, each at most CHUNK_WORDS, cut at paragraph markers when possible."""
    if end - start <= CHUNK_WORDS:
        return [(start, end)]
    marks = [i for i in range(start + 1, end - 1) if PARAGRAPH.match(words[i][0]) and words[i + 1][0][:1].isupper()]
    paras = list(zip([start] + marks, marks + [end]))
    pieces, cur = [], None
    for s, e in paras:
        if e - s > CHUNK_WORDS:
            if cur:
                pieces.append(cur)
                cur = None
            pieces.extend(_windows(s, e))
        elif cur and e - cur[0] <= CHUNK_WORDS:
            cur = (cur[0], e)
        else:
            if cur:
                pieces.append(cur)
            cur = (s, e)
    if cur:
        pieces.append(cur)
    return pieces


def structure_chunks(words):
    """Chunks with the same fields as chunk_pages ('id', 'start', 'end', 'page', 'endPage', 'text') plus 'head'.
    Returns None when the document has no recognisable provisions (the caller keeps sliding windows)."""
    heads = provision_heads(words)
    if not heads:
        return None
    if heads[0][0] > 0:
        heads.insert(0, (0, ""))   # title page and preamble
    bounds = [h[0] for h in heads] + [len(words)]
    spans = []   # (start, end, [headings], whole unit?)
    for (start, head), end in zip(heads, bounds[1:]):
        if end <= start:
            continue
        pieces = _split_unit(words, start, end)
        whole = len(pieces) == 1
        for s, e in pieces:
            prev = spans[-1] if spans else None
            if whole and prev and prev[3] and e - prev[0] <= CHUNK_WORDS:
                spans[-1] = (prev[0], e, prev[2] + [head], True)   # pack small provisions together
            else:
                spans.append((s, e, [head], whole))
    chunks = []
    for s, e, hs, _ in spans:
        hs = [h for h in hs if h]
        chunks.append({"id": len(chunks), "start": s, "end": e, "page": words[s][1], "endPage": words[e - 1][1],
                       "text": " ".join(w for w, _ in words[s:e]),
                       "head": " | ".join(hs if len(hs) <= 3 else [hs[0], "…", hs[-1]])})
    return chunks
