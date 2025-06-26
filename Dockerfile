# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user and switch to it
RUN useradd -m tara && chown -R tara:tara /app
USER tara

# Default command (can be overridden in docker-compose)
CMD ["python", "agent.py"]

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p tmp

# Set environment variables (these should be set in Railway dashboard)
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# Other required environment variables from .env

# Command to run the application
CMD ["python", "agent.py", "--whatsapp"]
