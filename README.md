# Re-ranking RAG on the GDPR

Bi-encoder retrieval vs bi-encoder + cross-encoder re-ranking, evaluated on the GDPR.
Shubhadeep Mandal, 2025201056.

```
notebook/   rag_rerank.ipynb, the experiment
            data/gdpr.pdf       the source document
            results/            saved outputs (ranks, answers, latency, scaling, failure hunts)
app/        live demo: FastAPI backend (server.py, rag_core.py, ...) and React UI (ui/)
            uploads/            PDFs uploaded through the UI and their indexes (created at runtime)
report/     report.tex and the compiled report.pdf
```

`.env` (copy `.env.example`) and the Python virtual environment `.venv/` sit at the root and are shared by
the notebook and the app.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cd app/ui && npm install
```

## Run

- **Notebook:** open `notebook/rag_rerank.ipynb` with the `.venv` kernel and run all cells.
- **App backend**, from `app/`: `../.venv/Scripts/python.exe -m uvicorn server:app --port 8000`
- **App UI**, from `app/ui/`: `npm run dev`, then open the printed URL.
- **Open model (optional):** `bash app/start_local_llm.sh` inside WSL, or point `LOCAL_LLM_URL`,
  `LOCAL_LLM_KEY` and `LOCAL_LLM_MODEL` at a hosted OpenAI-compatible provider (see `.env.example`).
- **Report**, from `report/`: `pdflatex report.tex` (twice).

## Hosted demo

https://huggingface.co/spaces/Seronic/retrieve-ten-keep-three runs the UI on saved results (no backend).
Redeploy it with `.venv/Scripts/python.exe app/space/deploy_static.py`. The full live app (retrieval,
re-ranking, Gemini answers) deploys with `app/space/deploy.py` to a Docker Space, which needs Hugging Face PRO.

## Full app on Railway (or any Docker host)

Live at https://app-production-a144.up.railway.app. `app/space/Dockerfile` builds the whole app from the
repository root; the encoders run on ONNX Runtime (no torch), so the app needs about 0.5 GB of RAM.
`.railwayignore` keeps `.venv` and `node_modules` out of the upload (the folder is not a git repository, so
`.gitignore` does not apply). Railway ignored `railway.json`'s builder setting, so the service also sets
`RAILWAY_DOCKERFILE_PATH=app/space/Dockerfile` and `PORT=7860` (the domain targets port 7860).
The image also ships the Constitution of India upload (`app/uploads/a914ab3a3b.*` and `library.json`, with its
embeddings, case study and failure hunt); other uploads stay out of the image.

```
npm install -g @railway/cli
railway login
railway init                                  # once: creates the project
railway variables --set GEMINI_API_KEY=... --set RAILWAY_DOCKERFILE_PATH=app/space/Dockerfile --set PORT=7860
railway up                                    # uploads the folder and builds the image
railway domain --port 7860                    # gives the service a public URL
```

The same Dockerfile works on Google Cloud Run or Fly.io. Uploaded PDFs are lost on redeploy unless a volume
is mounted at `/home/user/project/app/uploads`.
