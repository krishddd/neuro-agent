# Build context = neuro_agent/ directory
# docker build -t neuro-agent .
# docker run -p 8000:8000 -v $(pwd)/credentials:/app/neuro_agent/credentials neuro-agent

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

RUN useradd -m -u 1000 neuro
# Place the package at /app/neuro_agent so `import neuro_agent` works
WORKDIR /app/neuro_agent

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy entire neuro_agent/ into /app/neuro_agent/
COPY . .

# Mount credentials at runtime — never bake them into the image
VOLUME ["/app/neuro_agent/credentials"]
VOLUME ["/app/neuro_agent/outputs"]
VOLUME ["/app/neuro_agent/chroma_db"]

ENV PYTHONPATH=/app
ENV PORT=8000
ENV RELOAD=0

USER neuro
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["python", "run_server.py"]
