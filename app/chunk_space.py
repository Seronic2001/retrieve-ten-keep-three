"""2D maps of a document's chunk embeddings for the UI's Chunk space page.

Two projections of the same 384-d bi-encoder vectors:
- t-SNE ("meaning map"): keeps neighbourhoods, so topics form islands; distances between islands mean little.
- PCA ("linear view"): the two directions of greatest variance; faithful globally but crowded.
Topic groups come from k-means in the full 384-d space (not the 2D picture) and are named by their most
distinctive words (TF-IDF). Results are cached per document; t-SNE takes a few seconds per document.
"""
import threading

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

MAX_TOPICS = 6   # the UI's validated categorical palette has 6 slots; blue and orange are reserved for the pipelines
_locks, _cache = {}, {}


def _unit(coords):
    """Scales 2D coordinates into [0, 1] on both axes, keeping the min/max for later projections."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return (coords - lo) / span, lo, span


def _topic_names(texts, labels, k):
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
                          max_features=6000, min_df=2, sublinear_tf=True)
    try:
        tf = vec.fit_transform(texts)
    except ValueError:   # very small documents: too few repeated words
        return [f"Group {i + 1}" for i in range(k)]
    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(tf.mean(axis=0)).ravel()
    names = []
    for i in range(k):
        rows = labels == i
        lift = np.asarray(tf[rows].mean(axis=0)).ravel() - overall   # distinctive, not just frequent
        names.append(" · ".join(terms[np.argsort(-lift)[:3]]))
    return names


def build(doc_id, idx):
    """Both projections, topic groups and their names for one document (computed once, then cached)."""
    lock = _locks.setdefault(doc_id, threading.Lock())
    with lock:
        if doc_id in _cache and _cache[doc_id]["n"] == idx.index.ntotal:
            return _cache[doc_id]
        emb = idx.index.reconstruct_n(0, idx.index.ntotal)
        n = len(emb)
        pca = PCA(n_components=2, random_state=0).fit(emb)
        pca_xy, pca_lo, pca_span = _unit(pca.transform(emb))
        if n >= 5:
            tsne = TSNE(n_components=2, perplexity=max(2.0, min(30.0, (n - 1) / 3)), init="pca",
                        metric="cosine", learning_rate="auto", random_state=0)
            tsne_xy, _, _ = _unit(tsne.fit_transform(emb))
        else:
            tsne_xy = pca_xy
        k = int(np.clip(round(np.sqrt(n / 8)), 2, MAX_TOPICS)) if n >= 6 else 1
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb) if k > 1 else np.zeros(n, int)
        names = _topic_names(idx.texts, labels, k) if k > 1 else ["All chunks"]
        space = {"n": n, "emb": emb, "labels": labels, "names": names, "tsne": tsne_xy, "pca": pca_xy,
                 "pca_model": pca, "pca_lo": pca_lo, "pca_span": pca_span}
        _cache[doc_id] = space
        return space


def forget(doc_id):
    _cache.pop(doc_id, None)


def payload(space, idx):
    """What the map draws: every chunk's position in both projections, its topic, and its citation."""
    return {
        "points": [{"chunk": i, "tsne": [round(float(a), 4) for a in space["tsne"][i]],
                    "pca": [round(float(a), 4) for a in space["pca"][i]], "topic": int(space["labels"][i]),
                    "cite": idx.chunks[i]["cite"], "page": idx.chunks[i]["page"],
                    "preview": idx.texts[i][:160]}
                   for i in range(space["n"])],
        "topics": [{"id": t, "name": name, "size": int((space["labels"] == t).sum())}
                   for t, name in enumerate(space["names"])],
    }


def locate(space, idx, query):
    """Places a question on both maps and scores every chunk against it.
    PCA is a linear map, so the question's position is exact. t-SNE cannot place new points, so the question
    goes at the similarity-weighted centre of its 5 nearest chunks (marked approximate in the UI)."""
    q = idx.models.bi.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
    sims = space["emb"] @ q
    pca_xy = (space["pca_model"].transform(q[None])[0] - space["pca_lo"]) / space["pca_span"]
    near = np.argsort(-sims)[:5]
    w = np.exp((sims[near] - sims[near].max()) * 20)
    tsne_xy = (space["tsne"][near] * w[:, None]).sum(axis=0) / w.sum()
    return {"pca": [round(float(a), 4) for a in pca_xy], "tsne": [round(float(a), 4) for a in tsne_xy],
            "sims": [round(float(s), 3) for s in sims]}


def neighbours(space, idx, chunk, k=6):
    """The k chunks closest to one chunk in the full 384-d space (cosine)."""
    sims = space["emb"] @ space["emb"][chunk]
    order = [int(i) for i in np.argsort(-sims) if i != chunk][:k]
    return [{"chunk": i, "sim": round(float(sims[i]), 3), "cite": idx.chunks[i]["cite"], "page": idx.chunks[i]["page"],
             "preview": idx.texts[i][:160], "topic": int(space["labels"][i])} for i in order]
