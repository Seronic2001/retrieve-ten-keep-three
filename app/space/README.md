---
title: Retrieve Ten, Keep Three
emoji: 🔎
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Bi-encoder vs cross-encoder re-ranking RAG on the GDPR
---

# Retrieve Ten, Keep Three

Two RAG pipelines over the GDPR, side by side: a bi-encoder (`all-MiniLM-L6-v2` + FAISS) that sends its top 3
chunks to the LLM, and the same bi-encoder's top 10 re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
before the top 3 are sent. Ask your own questions, replay the 13 hand-labelled stress tests, or upload a PDF.

Shubhadeep Mandal, 2025201056.

- Retrieval and re-ranking run live on CPU (re-ranking 10 candidates takes about a second).
- Answers come from Gemini on a shared free-tier quota, so live answers can run out for the day; the saved
  case-study answers always load.
- Uploaded PDFs are visible to every visitor and are lost when the Space restarts.
