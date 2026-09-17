# Lean Python CPU base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://api:8000 \
    MODEL_DIR=/models

WORKDIR /app

# Install dependencies
COPY evaluator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and test files
COPY data/ /app/data/
COPY evaluator/ /app/evaluator/

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app/evaluator

CMD ["python", "evaluate.py"]
