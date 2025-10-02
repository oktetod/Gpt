"""
Smart AI Telegram Bot - Configuration
Production-ready configuration with AI chaining models
"""

import os
from typing import List, Dict

# ================== ENV VARIABLES ==================
def get_env(key: str, default: str = None, required: bool = True) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

API_ID = int(get_env('TELEGRAM_API_ID'))
API_HASH = get_env('TELEGRAM_API_HASH')
BOT_TOKEN = get_env('TELEGRAM_BOT_TOKEN')
DATABASE_URL = get_env('DATABASE_URL', 
    "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres")

# ================== SUBSCRIPTION REQUIREMENTS ==================
REQUIRED_CHANNELS: List[str] = ['@durov69_1']
REQUIRED_GROUPS: List[str] = ['@durov69_2']

# ================== AI CHAINING MODELS ==================
AI_CHAIN = {
    # Step 3: Reranking & Context Understanding
    "reranking": "qwen-3-235b-thinking",
    
    # Step 5: Main reasoning & generation
    "reasoning": "nemotron-ultra-rag",
    
    # Step 6: Code-specific generation
    "coding": "qwen-3-coder-480b",
    "coding_fallback": "llama-4-maverick",
    
    # Step 9: Summarization
    "summarization": "gpt-oss-120b",
    
    # General purpose
    "general": "gpt-oss-120b"
}

# ================== IMAGE GENERATION ==================
IMAGE_MODELS: Dict[str, str] = {
    "flux": "High quality, balanced",
    "flux-realism": "Photorealistic images",
    "flux-anime": "Anime/manga style",
    "flux-3d": "3D rendered look",
    "any-dark": "Dark/gothic aesthetic",
    "turbo": "Fast generation"
}

DEFAULT_IMAGE_MODEL = "flux"
IMAGE_BASE_URL = "https://image.pollinations.ai"

# Image generation parameters
IMAGE_DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "nologo": True,
    "enhance": True
}

# ================== SYSTEM LIMITS ==================
MAX_HISTORY = 20
MAX_MESSAGE_LENGTH = 4096
API_TIMEOUT = 180
MAX_TOKENS = 8000
CHUNK_SIZE = 2000
TOKEN_BUDGET = 7500

# Rate limiting
MAX_MESSAGES_PER_MINUTE = 20
MAX_IMAGES_PER_HOUR = 10
CACHE_TTL = 3600

# ================== API ENDPOINTS ==================
AI_ENDPOINTS = {
    "text": "https://text.pollinations.ai",
    "image": "https://image.pollinations.ai"
}

# ================== SECURITY ==================
DEBUG = get_env("DEBUG", "False", required=False).lower() == "true"
ADMIN_IDS: List[int] = []

# ================== BOT BEHAVIOR ==================
BOT_NAME = "Smart AI Bot"
BOT_VERSION = "3.0"
DEFAULT_LANGUAGE = "id"

# Response templates
MESSAGES = {
    "welcome": """👋 Selamat datang di {bot_name}!

🤖 Saya adalah asisten AI yang cerdas dan dapat membantu Anda:

✨ **Fitur Utama:**
• 💬 Chat cerdas dengan AI reasoning
• 💻 Generate & debug kode
• 🎨 Buat gambar dari deskripsi
• 📊 Analisis data & dokumen
• 🔍 Riset mendalam dengan web search

📝 **Cara Menggunakan:**
Cukup kirim pesan, saya akan otomatis memahami kebutuhan Anda!

Contoh:
• "Buatkan fungsi Python untuk sorting"
• "Gambar pemandangan gunung saat sunset"
• "Jelaskan cara kerja blockchain"

Mulai chat sekarang! 🚀""",
    
    "subscription_required": """⚠️ **Akses Terbatas**

Untuk menggunakan bot ini, Anda harus bergabung dengan:

📢 Channel: {channels}
👥 Group: {groups}

Setelah bergabung, kirim /start lagi.""",
    
    "error": "❌ Terjadi kesalahan: {error}",
    "processing": "⏳ Sedang memproses...",
    "generating_image": "🎨 Membuat gambar...",
    "generating_code": "💻 Menulis kode...",
    "thinking": "🤔 Berpikir...",
}

# ================== INTENT DETECTION ==================
INTENT_KEYWORDS = {
    "image": ["gambar", "buat", "lukis", "generate", "draw", "image", "picture", "foto"],
    "code": ["kode", "coding", "program", "fungsi", "script", "code", "function", "debug", "fix"],
    "analysis": ["analisa", "analisis", "jelaskan", "explain", "research", "riset"],
    "task": ["buatkan", "tolong", "help", "bantu", "create", "make"]
}

# ================== LOGGING ==================
LOG_LEVEL = "INFO" if not DEBUG else "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
