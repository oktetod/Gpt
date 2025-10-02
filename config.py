"""
Smart AI Telegram Bot - Configuration v2
Multi-provider AI support
"""

import os
from typing import List, Dict

# ================== ENV VARIABLES ==================
def get_env(key: str, default: str = None, required: bool = True) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

# Telegram credentials
API_ID = int(get_env('TELEGRAM_API_ID'))
API_HASH = get_env('TELEGRAM_API_HASH')
BOT_TOKEN = get_env('TELEGRAM_BOT_TOKEN')

# Database
DATABASE_URL = get_env('DATABASE_URL', 
    "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres")

# AI Provider API Keys
CEREBRAS_API_KEY = get_env('CEREBRAS_API_KEY', required=False)
NVIDIA_API_KEY = get_env('NVIDIA_API_KEY', required=False)

# ================== SUBSCRIPTION REQUIREMENTS ==================
REQUIRED_CHANNELS: List[str] = ['@durov69_1']
REQUIRED_GROUPS: List[str] = ['@durov69_2']

# ================== AI MODELS CONFIGURATION ==================
AI_CHAIN = {
    # Deep thinking & reasoning (Cerebras)
    "reasoning": "qwen-3-235b-a22b-thinking-2507",
    
    # Code generation (Cerebras)
    "coding": "qwen-3-coder-480b",
    "coding_fallback": "llama-4-maverick-17b-128e-instruct",
    
    # General chat & summarization (Cerebras)
    "general": "gpt-oss-120b",
    "summarization": "gpt-oss-120b",
    
    # Reranking (Cerebras)
    "reranking": "qwen-3-235b-a22b-instruct-2507",
    
    # Ultra reasoning (NVIDIA)
    "ultra": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    
    # OCR (NVIDIA)
    "ocr": "nvidia-nemoretriever-ocr"
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
MAX_OCR_PER_HOUR = 5
CACHE_TTL = 3600

# ================== SECURITY ==================
DEBUG = get_env("DEBUG", "False", required=False).lower() == "true"
ADMIN_IDS: List[int] = [int(x) for x in get_env("ADMIN_IDS", "").split(",") if x]

# ================== BOT BEHAVIOR ==================
BOT_NAME = "Smart AI Bot"
BOT_VERSION = "3.0-MultiProvider"
DEFAULT_LANGUAGE = "id"

# Response templates
MESSAGES = {
    "welcome": """👋 Selamat datang di {bot_name} v3.0!

🤖 **AI Engine Terbaru:**
• Cerebras (Qwen-3, Llama-4, GPT-OSS)
• NVIDIA NIM (Nemotron Ultra, OCR)
• Multi-provider dengan auto-fallback

✨ **Kemampuan:**
• 💬 Chat dengan deep reasoning (65K tokens)
• 💻 Code generation (40K tokens)
• 🎨 Image generation (Flux models)
• 📄 OCR - Extract text dari gambar
• 📊 Analisis mendalam

📝 **Cara Pakai:**
Kirim pesan biasa atau foto dengan pertanyaan!

Contoh:
• "Buatkan algoritma sorting dengan penjelasan"
• "Gambar cyberpunk city, realistic"
• [Kirim foto] "Baca teks dalam gambar ini"
• "Analisa perbedaan Python vs Go secara mendalam"

🚀 Powered by Cerebras & NVIDIA!""",
    
    "subscription_required": """⚠️ **Akses Terbatas**

Untuk menggunakan bot ini, bergabung dulu:

📢 Channel: {channels}
👥 Group: {groups}

Setelah join, kirim /start lagi.""",
    
    "error": "❌ Error: {error}",
    "processing": "⏳ Processing dengan AI...",
    "ocr_processing": "📄 Extracting text dari gambar...",
    "thinking": "🧠 Deep thinking mode...",
    "coding": "💻 Generating code...",
}

# ================== INTENT KEYWORDS ==================
INTENT_KEYWORDS = {
    "image": ["gambar", "buat", "lukis", "generate", "draw", "image", "picture", "foto"],
    "code": ["kode", "coding", "program", "fungsi", "script", "code", "function", "debug", "fix", "algorithm"],
    "analysis": ["analisa", "analisis", "jelaskan", "explain", "research", "riset", "deep", "mendalam"],
    "task": ["buatkan", "tolong", "help", "bantu", "create", "make"],
    "ocr": ["baca", "extract", "ocr", "text", "tulisan"]
}

# ================== LOGGING ==================
LOG_LEVEL = "INFO" if not DEBUG else "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
