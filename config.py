# /proyek_bot/config.py

import os

# ================== KONFIGURASI BOT ==================
API_ID = os.getenv('ID')
API_HASH = os.getenv('HASH')
BOT_TOKEN = os.getenv('BOT')
DATABASE_URL = "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

# Channel dan Grup yang wajib diikuti pengguna
REQUIRED_CHANNELS = ['@durov69_1']
REQUIRED_GROUPS = ['@durov69_2']

# Jumlah maksimal riwayat percakapan yang disimpan
MAX_HISTORY = 20

# ================== MODEL AI ==================
# Model Teks
TEXT_MODELS = {
    # Model Premium
    "gpt-5": "openai",          # Default - GPT-5 Gratis
    "gpt-5-mini": "openai-fast",
    "gpt-5-chat": "openai-large",
    "o4-mini": "openai-reasoning",
    
    # Model Visi (Gambar)
    "gemini": "gemini",
    "gemini-search": "gemini-search",
    
    # Model Spesialis
    "deepseek": "deepseek",
    "deepseek-r1": "deepseek-reasoning",
    "qwen-coder": "qwen-coder",
    "mistral": "mistral",
    
    # Audio
    "audio": "openai-audio",
    
    # Komunitas
    "evil": "evil",
    "unity": "unity"
}

# Model Gambar
IMAGE_MODELS = {
    "flux": "Default - Kualitas seimbang",
    "flux-realism": "Gambar fotorealistis",
    "flux-anime": "Gaya anime",
    "flux-3d": "Gaya render 3D",
    "any-dark": "Estetika gelap",
    "turbo": "Generasi cepat",
    "kontext": "Transformasi gambar-ke-gambar"
}

# Pilihan Suara untuk Audio
AUDIO_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage"]
