# Smart AI Telegram Bot - Production Dockerfile v3.1
# FIXED: Properly loads .env file

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL: Copy .env file BEFORE copying Python files
# This ensures environment variables are available when config.py is imported
COPY .env .

# Copy application code
COPY config.py .
COPY database.py .
COPY web_search.py .
COPY ai_engine.py .
COPY bot.py .

# Create sessions directory
RUN mkdir -p /app/sessions

# Verify .env file exists
RUN test -f /app/.env && echo "✓ .env file loaded" || echo "⚠ WARNING: .env file not found!"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/bot_session.session') else 1)"

# Run bot
CMD ["python", "-u", "bot.py"]
