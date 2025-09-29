# Gunakan image Python 3.12 slim
FROM python:3.12-slim

# Atur direktori kerja
WORKDIR /app

# Atur environment variables untuk efisiensi
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Salin file requirements terlebih dahulu untuk caching layer
COPY requirements.txt .

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin file bot ke dalam container
COPY . .

# Health check untuk memverifikasi bot berhasil login (session file ada)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/bot_session.session') else 1)"

# Jalankan bot
CMD ["python", "bot.py"]
