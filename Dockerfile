FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY src ./src
COPY data ./data
COPY artifacts ./artifacts
COPY .streamlit/config.toml ./.streamlit/config.toml
RUN useradd --uid 1000 --create-home appuser && \
    mkdir -p /app/runtime && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
