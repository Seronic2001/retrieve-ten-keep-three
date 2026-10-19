"""Deploys the UI alone to a free static Hugging Face Space, running on saved results (no backend).

Builds the UI with VITE_STATIC_DEMO=1 (no server polling, hosted-demo wording), adds a Space card, and
uploads the build. No secrets are involved: a static site cannot keep one hidden. For the full live app
(retrieval, re-ranking and Gemini answers) use deploy.py, which needs a Docker Space.

Run from the project root, logged in with `hf auth login`:
    .venv/Scripts/python.exe app/space/deploy_static.py
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = os.getenv("SPACE_ID", "Seronic/retrieve-ten-keep-three")
UI = Path(__file__).parent.parent / "ui"

CARD = """---
title: Retrieve Ten, Keep Three
emoji: 🔎
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Bi-encoder vs cross-encoder re-ranking RAG on the GDPR
---

# Retrieve Ten, Keep Three

Two RAG pipelines over the GDPR, side by side: a bi-encoder (`all-MiniLM-L6-v2` + FAISS) that sends its top 3
chunks to the LLM, and the same bi-encoder's top 10 re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
before the top 3 are sent.

Shubhadeep Mandal, 2025201056.

This hosted demo runs on saved results: the 13 hand-labelled GDPR stress-test questions with both pipelines'
retrievals, rank shifts and LLM answers, plus the failure hunt. Asking new questions, uploading PDFs and
building case studies need the full app with its Python server.
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "site"
        npx = shutil.which("npx") or "npx"
        subprocess.run([npx, "vite", "build", "--outDir", str(out), "--emptyOutDir"], cwd=UI, check=True,
                       env={**os.environ, "VITE_STATIC_DEMO": "1"})
        (out / "README.md").write_text(CARD, encoding="utf-8")
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to {REPO_ID}: {', '.join(files)}")

        api = HfApi()
        api.create_repo(REPO_ID, repo_type="space", space_sdk="static", private=False, exist_ok=True)
        api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=out,
                          commit_message="Deploy static demo from Rag-assignment")
    print(f"Done: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
