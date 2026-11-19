FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ configs/
COPY src/ src/
COPY data/sample/ data/sample/
RUN mkdir -p results logs
CMD ["python", "-m", "src.main", "--config", "configs/config.yaml"]
