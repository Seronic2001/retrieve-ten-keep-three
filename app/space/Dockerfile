# Image for the full live app. Build context = the repository root (Railway, Cloud Run, Fly.io) or the
# folder deploy.py assembles for a Hugging Face Docker Space, which has the same layout (app/, notebook/).
# Listens on $PORT when the host sets it (Railway, Cloud Run), else on 7860 (Hugging Face).

# Stage 1: build the React UI
FROM node:20-slim AS ui
WORKDIR /ui
COPY app/ui/package.json app/ui/package-lock.json ./
RUN npm ci
COPY app/ui/ ./
RUN npm run build

# Stage 2: FastAPI backend serving the API and the built UI, CPU only
FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/project

COPY --chown=user app/space/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake both encoders' ONNX exports and tokenizers into the image so the first request does not wait for a download
RUN python -c "from huggingface_hub import hf_hub_download as d; [d(r, f) for r in ('sentence-transformers/all-MiniLM-L6-v2', 'cross-encoder/ms-marco-MiniLM-L-6-v2')  for f in ('tokenizer.json', 'onnx/model.onnx')]"
ENV HF_HUB_OFFLINE=1

COPY --chown=user notebook/data notebook/data
COPY --chown=user notebook/results notebook/results
COPY --chown=user app/*.py app/
# Pre-loaded upload: the Constitution of India with its embeddings, case study and failure hunt
COPY --chown=user app/uploads app/uploads
COPY --chown=user --from=ui /ui/dist app/ui/dist

WORKDIR $HOME/project/app
# Embed the GDPR at build time (app/uploads/gdpr.emb.npz) so the server starts in seconds
RUN python -c "import server"
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
