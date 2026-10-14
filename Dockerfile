# Production image for the public showcase (Render free web service: 512 MB RAM, no disk).
#
# The two local models (embeddings + cross-encoder, ~150 MB on disk) are downloaded at BUILD
# time into the image, because a free host starts every instance with an empty disk: without
# this, each cold start would fetch them from Hugging Face before answering a question.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/srv/models \
    # The app user has no home directory: give the Hugging Face client a cache it can write.
    HF_HOME=/srv/models/.hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # glibc keeps one malloc arena per thread by default, which inflates resident memory.
    MALLOC_ARENA_MAX=2

# The service runs as an unprivileged user: the PDF parser processes untrusted input. Only the
# model cache belongs to it; the code stays owned by root, so the process cannot rewrite it.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /srv/models \
    && chown app:app /srv/models

WORKDIR /srv

COPY requirements.txt requirements-local-models.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-local-models.txt

USER app

# Bake the model weights into the image (written by the app user, so no "chown -R" layer
# duplicates them). The names must match FASTEMBED_MODEL and FASTEMBED_RERANK_MODEL, which
# render.yaml sets explicitly.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# From here on the models are read from the image only: no call to Hugging Face at run time.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY demo ./demo

EXPOSE 8000

# /livez answers without touching the database, so the health check never wakes Neon.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/livez').status == 200 else 1)"

# Render injects PORT. One worker on purpose: the budget lives in this process's memory, and a
# second worker would load a second copy of both models.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
