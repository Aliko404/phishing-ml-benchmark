FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# Install pinned dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY data/sample/ data/sample/

RUN mkdir -p results logs

# Default: smoke test on the micro sample
CMD ["python", "-m", "src.main", "--config", "configs/config.yaml"]
