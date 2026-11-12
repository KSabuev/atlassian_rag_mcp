FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY build/wheels/ /wheels/
RUN pip install --no-cache-dir --no-index --find-links /wheels --no-deps \
        lancedb==0.37.1 pyarrow==20.0.0 tantivy==0.26.0 simsimd==6.5.16 \
        tokenizers==0.23.1 huggingface_hub==1.27.0 hf_xet==1.6.0 filelock==3.32.3 \
        fsspec==2026.7.0 packaging==26.3.0 pyyaml==6.0.3 tqdm==4.70.0 typing_extensions==4.16.0 \
    && pip install --no-cache-dir --retries 5 --timeout 60 -e . \
    && rm -rf /wheels

FROM python:3.12-slim AS final

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

RUN pip install --no-cache-dir -e . --no-deps

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/bin/sh", "-c", "curl --fail http://localhost:${PORT}/health || exit 1"]

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "-m", "atlassian_rag_mcp"]
