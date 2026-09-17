# Lean Python CPU base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY trainer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset and trainer source code
COPY data/ /app/data/
COPY trainer/ /app/trainer/

WORKDIR /app/trainer

# Run training pipeline and output to /models
CMD ["python", "train.py"]
