FROM python:3.11-slim AS base

# Avoid writing .pyc files and ensure logs are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for psycopg2 and clean up afterwards
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Default port for local run; Railway will inject PORT
ENV PORT=8000

# Expose for local (optional, Railway maps automatically)
EXPOSE 8000

# Start FastAPI with Uvicorn, binding to 0.0.0.0 and PORT env
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]


