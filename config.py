"""
Smart AI Telegram Bot - Configuration v3.1 (FIXED)
GPT-5 by @durov9369 - With Web Search & Multiple API Keys
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# ================== LOAD ENV FILE FIRST ==================
# CRITICAL: Load .env before accessing any environment variables
load_dotenv(override=True)

# ================== ENV VARIABLES ==================
def get_env(key: str, default: str = None, required: bool = True) -> str:
    """Get environment variable with proper error handling"""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

# Telegram credentials
API_ID = int(get_env('TELEGRAM_API_ID'))
API_HASH = get_env('TELEGRAM_API_HASH')
BOT_TOKEN = get_env('TELEGRAM_BOT_TOKEN')

# Database
DATABASE_URL = get_env('DATABASE_URL')

# ================== MULTIPLE API KEYS (LOAD BALANCING) ==================
# Parse multiple Cerebras API keys from comma-separated list
CEREBRAS_API_KEYS_RAW = get_env('CEREBRAS_API_KEY', required=False)
CEREBRAS_API_KEYS = []
if CEREBRAS_API_KEYS_RAW:
    CEREBRAS_API_KEYS = [key.strip() for key in CEREBRAS_API_KEYS_RAW.split(',') if key.strip()]

# Select primary key (will rotate in ai_engine.py)
CEREBRAS_API_KEY = CEREBRAS_API_KEYS[0] if CEREBRAS_API_KEYS else None

# NVIDIA API Key
NVIDIA_API_KEY = get_env('NVIDIA_API_KEY', required=False)

# ================== WEB SEARCH CONFIGURATION ==================
WEB_SEARCH_ENABLED = True
WEB_SEARCH_SOURCES = ['duckduckgo', 'wikipedia', 'bing', 'archive', 'scholar', 'news']
WEB_SEARCH_MAX_RESULTS_PER_SOURCE = 5
WEB_SEARCH_TIMEOUT = 30
WEB_SEARCH_AUTO_TRIGGER_KEYWORDS = [
    'cari', 'search', 'terbaru', 'berita', 'news', 'update',
    'informasi', 'info', 'what is', 'apa itu', 'siapa',
    'when', 'kapan', 'where', 'dimana', 'latest', 'recent'
]

# ================== SUBSCRIPTION REQUIREMENTS ==================
REQUIRED_CHANNELS: List[str] = ['@durov69_1']
REQUIRED_GROUPS: List[str] = ['@durov69_2']

# ================== AI MODELS CONFIGURATION (GPT-5 Stack) ==================
AI_CHAIN = {
    # Ultra reasoning with RAG & Vision (NVIDIA Nemotron)
    "ultra": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    
    # Deep thinking & reasoning (Cerebras Qwen-3)
    "reasoning": "qwen-3-235b-thinking",
    
    # Code generation (Cerebras Qwen-3 Coder)
    "coding": "qwen-3-coder-480b",
    "coding_fallback": "llama-4-maverick",
    
    # General chat & summarization
    "general": "gpt-oss-120b",
    "summarization": "gpt-oss-120b",
    
    # Reranking & Web Search Summary
    "reranking": "qwen-3-235b-thinking",
    "web_summary": "qwen-3-235b-thinking",
    
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
MAX_MESSAGE_LENGTH = 4096  # Telegram limit
API_TIMEOUT = 180
MAX_TOKENS = 8000
CHUNK_SIZE = 3800  # Safe chunk size (leave room for formatting)
TOKEN_BUDGET = 7500

# Rate limiting
MAX_MESSAGES_PER_MINUTE = 20
MAX_IMAGES_PER_HOUR = 10
MAX_OCR_PER_HOUR = 5
MAX_WEB_SEARCHES_PER_HOUR = 30
CACHE_TTL = 3600

# ================== SECURITY ==================
DEBUG = get_env("DEBUG", "False", required=False).lower() == "true"
ADMIN_IDS: List[int] = [int(x) for x in get_env("ADMIN_IDS", "").split(",") if x]

# ================== BOT BEHAVIOR ==================
BOT_NAME = "GPT-5"
BOT_VERSION = "3.1-WebSearch"
BOT_CREATOR = "@durov9369"
DEFAULT_LANGUAGE = "id"

# Response templates
MESSAGES = {
    "welcome": """👋 **Selamat datang di GPT-5 + Web Search!**

🤖 **AI Generasi Terbaru dengan Real-Time Search**
Dikembangkan dan di-fine-tune oleh **@durov9369**

**🔬 Advanced Architecture:**
• Multi-provider AI chaining (Cerebras + NVIDIA)
• **Real-Time Web Search** (DuckDuckGo, Bing, Wikipedia, Archive.org, Scholar)
• RAG (Retrieval Augmented Generation)  
• Vision analysis dengan Nemotron Ultra 253B
• Code generation dengan Qwen-3 Coder 480B
• Deep reasoning hingga 65K tokens

**✨ Kemampuan Utama:**
• 💬 **Chat**: Natural conversation dengan context awareness
• 🌐 **Web Search**: Real-time info dari internet
• 💻 **Code**: Production-ready code generation
• 🎨 **Image**: High-quality image generation (Flux)
• 🔍 **Vision**: Multi-modal image analysis
• 📄 **OCR**: Text extraction dari gambar
• 🧠 **Analysis**: Deep reasoning & research

**📝 Contoh Penggunaan:**

*Web Search:*
"Cari berita terbaru tentang AI"
"Apa itu quantum computing?"
"Siapa pemenang Nobel 2024?"

*Chat + Search:*
"Jelaskan blockchain dengan referensi terbaru"

*Code Generation:*
"Buatkan REST API dengan authentication"

*Image Generation:*
"Gambar cyberpunk city at night, realistic"

*Vision Analysis:*
[Kirim foto] "Analisa gambar ini"

*OCR:*
[Kirim foto] "Baca teks dalam gambar"

**⚡ Features:**
✓ Multiple search engines (6 sources)
✓ Deep web access (Archive.org)
✓ Academic papers (Google Scholar)
✓ Real-time news
✓ Load-balanced API keys
✓ Smart query routing

Ketik /help untuk panduan lengkap

---
💫 *GPT-5 v3.1 by @durov9369*
🌐 *With Real-Time Web Search*""",
    
    "subscription_required": """⚠️ **Akses Terbatas**

Untuk menggunakan **GPT-5 + Web Search**, silakan bergabung:

📢 **Channel:**
{channels}

👥 **Group:**
{groups}

Setelah join, kirim /start lagi untuk aktivasi.

---
💫 *GPT-5 by @durov9369*""",
    
    "error": "❌ **Error:** {error}\n\n_Jika masalah berlanjut, hubungi @durov9369_",
    
    "processing": "⏳ **GPT-5 Processing...**\n_Analyzing your request with advanced AI_",
    
    "web_searching": "🌐 **Searching the Web...**\n_Querying multiple sources: DuckDuckGo, Bing, Wikipedia, Archive.org, Scholar_",
    
    "web_summarizing": "📊 **Analyzing {count} results...**\n_Powered by Qwen-3 235B Thinking_",
    
    "ocr_processing": "📄 **Extracting Text...**\n_Using NVIDIA OCR technology_",
    
    "thinking": "🧠 **Deep Thinking Mode**\n_Qwen-3 235B analyzing (65K tokens context)_",
    
    "coding": "💻 **Code Generation**\n_Qwen-3 Coder 480B working on it_",
}

# ================== INTENT KEYWORDS (ENHANCED) ==================
INTENT_KEYWORDS = {
    "web_search": [
        "cari", "search", "terbaru", "berita", "news", "update",
        "informasi", "info", "what is", "apa itu", "siapa", "who is",
        "when", "kapan", "where", "dimana", "latest", "recent",
        "find", "temukan", "lookup", "google", "bing"
    ],
    
    "image": [
        "gambar", "buat gambar", "lukis", "generate image",
        "draw", "image", "picture", "foto", "bikin gambar",
        "create image", "paint"
    ],
    
    "code": [
        "kode", "coding", "program", "fungsi", "script",
        "code", "function", "debug", "fix", "algorithm",
        "buatkan code", "write code", "implement", "develop",
        "api", "backend", "frontend", "database"
    ],
    
    "analysis": [
        "analisa", "analisis", "jelaskan", "explain",
        "research", "riset", "deep", "mendalam",
        "detail", "comprehensive", "study", "investigate"
    ],
    
    "task": [
        "buatkan", "tolong", "help", "bantu",
        "create", "make", "generate", "build"
    ],
    
    "ocr": [
        "baca", "extract", "ocr", "text", "tulisan",
        "read", "scan", "extract text", "get text"
    ],
    
    "vision": [
        "lihat", "cek", "periksa", "analisa gambar",
        "analyze image", "what's in", "describe",
        "identify", "explain image", "apa ini", "what is"
    ]
}

# ================== GPT-5 METADATA ==================
GPT5_INFO = {
    "name": "GPT-5",
    "version": "3.1",
    "developer": "@durov9369",
    "architecture": "Hybrid Multi-Provider + Web Search",
    "core_models": [
        "NVIDIA Nemotron Ultra 253B (RAG & Vision)",
        "Cerebras Qwen-3 235B (Deep Reasoning)",
        "Cerebras Qwen-3 Coder 480B (Code Gen)",
        "Cerebras Llama-4 Maverick (Fast Response)"
    ],
    "search_engines": [
        "DuckDuckGo (Privacy-focused)",
        "Bing (Web Search)",
        "Wikipedia (Encyclopedia)",
        "Archive.org (Deep Web)",
        "Google Scholar (Academic)",
        "Bing News (Real-time News)"
    ],
    "capabilities": [
        "Natural Language Understanding",
        "Real-Time Web Search",
        "Code Generation & Debugging",
        "Image Generation (Flux)",
        "Vision Analysis & OCR",
        "Deep Reasoning (65K tokens)",
        "Multi-modal Processing",
        "RAG (Retrieval Augmented Generation)",
        "Academic Research",
        "News Monitoring"
    ],
    "performance": {
        "response_time": "<5 seconds (with search)",
        "context_window": "40K-65K tokens",
        "accuracy": "95%+",
        "uptime": "99.9%",
        "search_sources": "6 engines",
        "api_redundancy": "Multiple keys"
    }
}

# ================== LOGGING ==================
LOG_LEVEL = "INFO" if not DEBUG else "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ================== FEATURE FLAGS ==================
FEATURES = {
    "vision_analysis": True,     # Nemotron vision
    "ocr": True,                  # NVIDIA OCR
    "rag": True,                  # RAG with history
    "code_gen": True,             # Qwen-3 Coder
    "image_gen": True,            # Flux models
    "deep_thinking": True,        # Qwen-3 Thinking
    "web_search": True,           # Multi-engine search
    "deep_web": True,             # Archive.org
    "academic_search": True,      # Scholar
    "news_search": True,          # News
    "rate_limiting": True,        # User rate limits
    "analytics": True,            # Usage tracking
    "api_rotation": True          # Multiple API keys
}

# ================== RESPONSE CUSTOMIZATION ==================
RESPONSE_STYLES = {
    "casual": "friendly and conversational",
    "professional": "formal and detailed",
    "technical": "precise and technical"
}

DEFAULT_STYLE = "casual"
