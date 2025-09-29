# Use Python 3.11 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies directly (no requirements.txt)
RUN pip install --no-cache-dir \
    telethon==1.36.0 \
    asyncpg==0.29.0 \
    httpx==0.27.0 \
    cryptg==0.4.0

# Copy bot file
COPY bot.py .

# Create directory for session
RUN mkdir -p /app/sessions

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/bot_session.session') else 1)"

# Run bot
CMD ["python", "-u", "bot.py"]
