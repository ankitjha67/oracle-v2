FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -e ".[all]"

# Copy application code
COPY . .

# Create directories for data and cache
RUN mkdir -p cricsheet_data football_data .cache

# Default: run all predictions
ENTRYPOINT ["python", "run.py"]

# Override with specific sports: docker run oracle-v2 --cricket
CMD []
