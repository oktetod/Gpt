# Gunakan image Python 3.12 slim
FROM python:3.12-slim

# Atur direktori kerja
WORKDIR /app

# Atur environment variables untuk efisiensi
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies (Cerebras SDK ditambahkan)
RUN pip install --no-cache-dir \
    telethon==1.36.0 \
    asyncpg==0.29.0 \
    httpx==0.27.0 \
    cryptg==0.4.0 \
    cerebras-cloud-sdk==1.2.0

# Salin file bot ke dalam container
COPY bot.py .

# Health check untuk memverifikasi bot berhasil login (session file ada)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/bot_session.session') else 1)"

# Jalankan bot
CMD ["python", "bot.py"]
