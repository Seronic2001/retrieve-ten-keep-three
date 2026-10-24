"""The two encoders on ONNX Runtime instead of PyTorch, with the slice of the sentence-transformers API the
app uses (SentenceTransformer.encode, CrossEncoder.predict). Both model repos publish an official ONNX
export (onnx/model.onnx), so the scores match the PyTorch models to float precision while the deployed
image drops torch (~1 GB) and loads faster.
"""
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def _cpus():
    """CPUs this process may use: a container's CPU quota (cgroup v2), not the host's core count, which
    can be far larger and makes ONNX Runtime's threads fight over a slice of a CPU."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def _load(repo, max_length, file="onnx/model.onnx", threads=None):
    """file: the ONNX export to run, e.g. a quantized one. threads: ONNX Runtime threads, by default every CPU
    the process may use."""
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    pad = "[PAD]" if tok.token_to_id("[PAD]") is not None else "<pad>"   # BERT vocabularies vs XLM-RoBERTa's
    tok.enable_padding(pad_id=tok.token_to_id(pad), pad_token=pad)
    opts = ort.SessionOptions()
    # No memory arena: it keeps the largest batch's buffers for good, which pushed start-up past 1.4 GB.
    opts.intra_op_num_threads = threads or _cpus()
    opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(hf_hub_download(repo, file), opts, providers=["CPUExecutionProvider"])
    return tok, sess


def _feeds(sess, encodings):
    arrays = {"input_ids": [e.ids for e in encodings], "attention_mask": [e.attention_mask for e in encodings],
              "token_type_ids": [e.type_ids for e in encodings]}
    names = {i.name for i in sess.get_inputs()}
    return {k: np.asarray(v, dtype=np.int64) for k, v in arrays.items() if k in names}


class SentenceTransformer:
    """Bi-encoder: mean pooling over the token embeddings, as all-MiniLM-L6-v2's pooling config specifies."""

    def __init__(self, repo, max_length=256):
        self.tok, self.sess = _load(repo, max_length)

    def encode(self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True):
        out = []
        for i in range(0, len(texts), batch_size):
            feeds = _feeds(self.sess, self.tok.encode_batch(texts[i:i + batch_size]))
            tokens = self.sess.run(None, feeds)[0]
            mask = feeds["attention_mask"][..., None].astype(np.float32)
            emb = (tokens * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            if normalize_embeddings:
                emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            out.append(emb)
        return np.vstack(out)


class CrossEncoder:
    """Cross-encoder: one raw logit per (query, passage) pair; the model's default activation is identity."""

    def __init__(self, repo, max_length=512, file="onnx/model.onnx", threads=None):
        self.tok, self.sess = _load(repo, max_length, file, threads)

    def predict(self, pairs, batch_size=32):
        out = []
        for i in range(0, len(pairs), batch_size):
            feeds = _feeds(self.sess, self.tok.encode_batch([tuple(p) for p in pairs[i:i + batch_size]]))
            out.append(self.sess.run(None, feeds)[0][:, 0])
        return np.concatenate(out)
