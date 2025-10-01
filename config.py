import os
from typing import List, Dict, Optional

# ================== KONFIGURASI ENV ==================
API_ID = int(os.getenv('TELEGRAM_API_ID'))
API_HASH = os.getenv('TELEGRAM_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# ================== CHANNEL & GRUP WAJIB ==================
REQUIRED_CHANNELS: List[str] = ['@durov69_1']
REQUIRED_GROUPS: List[str] = ['@durov69_2']

# ================== MODEL AI ==================
TEXT_MODELS: Dict[str, str] = {
    "gpt-oss": "Cerebras GPT-OSS 120B",
    "gemini": "gemini",
    "gemini-search": "gemini-search",
    "deepseek": "deepseek",
    "deepseek-r1": "deepseek-reasoning",
    "qwen-coder": "qwen-coder",
    "mistral": "mistral"
}

# Daftar model teks yang didukung
SUPPORTED_TEXT_MODELS: List[str] = list(TEXT_MODELS.keys())

# Daftar model gambar yang didukung
image_models: List[str] = [
    "dall-e-3",
    "dall-e-2"
]

# Daftar suara untuk TTS (audio)
AUDIO_VOICES: List[str] = [
    'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
]

# ================== BATASAN SISTEM ==================
MAX_HISTORY: int = 20
MAX_MESSAGE_LENGTH: int = 4096
API_TIMEOUT: int = 30
MAX_MESSAGES_PER_HOUR: int = 100
CACHE_TTL: int = 3600

# ================== BOT & PESAN ==================
COMMAND_PREFIXES: List[str] = ["/", "!", "?"]
VALID_ROLES: List[str] = ["user", "assistant", "system"]
BOT_API_URL: str = "https://api.telegram.org/bot"

# ================== KEAMANAN & ADMIN ==================
DEBUG: bool = False
ADMIN_IDS: List[int] = []
OPENAI_API_KEY: Optional[str] = None

default_model: str = "gpt-oss"

# ================== PROVIDER AI ==================
AI_PROVIDERS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "pollinations": "https://text.pollinations.ai",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "cerebras": "https://api.cerebras.ai/v1"
}
