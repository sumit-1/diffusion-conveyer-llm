FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=7860

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy pyyaml tiktoken

COPY pyproject.toml README.md ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/tinystories/tiktoken_compact_4096.json ./data/tinystories/tiktoken_compact_4096.json
COPY data/tinystories/val_sample_256.txt ./data/tinystories/val_sample_256.txt
COPY runs/tinystories/benchmark_24h.json ./runs/tinystories/benchmark_24h.json

EXPOSE 7860

CMD ["python", "-m", "conveyer.web_sim", "--host", "0.0.0.0", "--port", "7860"]
