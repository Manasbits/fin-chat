# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN set -ex \
    && mkdir -p /var/lib/apt/lists/partial \
    && chmod 755 /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN useradd -m tara

# Create necessary directories
RUN mkdir -p /app/tmp \
    && chown -R tara:tara /app

# Copy requirements first to leverage Docker cache
COPY --chown=tara:tara requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=tara:tara . .

# Switch to non-root user
USER tara

# Default command (can be overridden in docker-compose)
CMD ["python", "agent.py", "--whatsapp"]
