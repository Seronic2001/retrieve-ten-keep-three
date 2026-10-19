"""Deploys the app to a Hugging Face Docker Space.

Assembles only what the Space needs (backend code, UI source, the GDPR PDF and saved results) in a temporary
folder, sets GEMINI_API_KEY from the project-root .env as an encrypted Space secret, and uploads the folder.
Never uploads .env, .venv, node_modules or app/uploads.

Run from the project root, logged in with `hf auth login`:
    .venv/Scripts/python.exe app/space/deploy.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

REPO_ID = os.getenv("SPACE_ID", "Seronic/retrieve-ten-keep-three")
SPACE = Path(__file__).parent          # app/space/
APP = SPACE.parent                     # app/
PROJECT = APP.parent


def stage(dest):
    """Copies the Space's files into dest, laid out like the repository (app/, notebook/)."""
    # The Space wants its Dockerfile and card at the root; the Dockerfile reads app/space/requirements.txt.
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(SPACE / name, dest / name)
    (dest / "app" / "space").mkdir(parents=True)
    shutil.copy2(SPACE / "requirements.txt", dest / "app" / "space" / "requirements.txt")
    for py in APP.glob("*.py"):
        shutil.copy2(py, dest / "app" / py.name)
    shutil.copytree(APP / "ui", dest / "app" / "ui", ignore=shutil.ignore_patterns("node_modules", "dist"))
    for sub in ("data", "results"):
        shutil.copytree(PROJECT / "notebook" / sub, dest / "notebook" / sub)


def main():
    api = HfApi()
    key = dotenv_values(PROJECT / ".env").get("GEMINI_API_KEY")
    if not key:
        sys.exit("No GEMINI_API_KEY in .env; add it there first (or set the secret in the Space settings).")

    api.create_repo(REPO_ID, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(REPO_ID, "GEMINI_API_KEY", key)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        files = sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to {REPO_ID}:")
        for f in files:
            print("  ", f)
        api.upload_folder(repo_id=REPO_ID, repo_type="space", folder_path=tmp,
                          commit_message="Deploy app from Rag-assignment")
    print(f"Done: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
