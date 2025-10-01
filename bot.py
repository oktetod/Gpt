import os
import asyncio
import random
import traceback
from typing import List, Dict, Optional
from urllib.parse import quote

from telethon import TelegramClient, events, Button
from telethon.errors import MessageTooLongError
import httpx
from cerebras.cloud.sdk import Cerebras

# ================== 1. KONFIGURASI ==================
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

REQUIRED_CHANNELS = ['@durov69_1']
REQUIRED_GROUPS = ['@durov69_2']
MAX_HISTORY = 20

# Model Teks diperbarui: GPT-5 diganti dengan GPT-OSS
TEXT_MODELS = {
    # Model Utama Baru
    "gpt-oss": "Cerebras GPT-OSS 120B",
    
    # Model Visi (Gambar)
    "gemini": "gemini",
    "gemini-search": "gemini-search",
    
    # Model Spesialis dari Pollinations
    "deepseek": "deepseek",
    "deepseek-r1": "deepseek-reasoning",
    "qwen-coder": "qwen-coder",
    "mistral": "mistral"
}

# Suara untuk TTS (Audio)
AUDIO_VOICES = [
    'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
]

# Batas panjang pesan Telegram
MAX_MESSAGE_LENGTH = 4096

# ================== 2. KELAS DATABASE ==================
class Database:
    def __init__(self, url: str):
        self.url = url
        self.pool = None
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                statement_cache_size=0  # Fix untuk error PGBouncer/Supabase
            )
            await self.init_tables()
            print("Koneksi database berhasil dibuat.")
    
    async def init_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_history_user 
                ON chat_history(user_id, created_at DESC)
            ''')

    async def get_or_create_user(self, user_id: int, username: str = None, first_name: str = None):
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            if not user:
                await conn.execute(
                    'INSERT INTO users (user_id, username, first_name) VALUES ($1, $2, $3)',
                    user_id, username, first_name
                )
            else:
                await conn.execute(
                    'UPDATE users SET last_active = NOW() WHERE user_id = $1',
                    user_id
                )
            return await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)

    async def is_verified(self, user_id: int) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT is_verified FROM users WHERE user_id = $1", user_id
            )
            return result if result is not None else False

    async def set_verified(self, user_id: int, status: bool = True):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET is_verified = $1 WHERE user_id = $2",
                status, user_id
            )

    async def save_message(self, user_id: int, role: str, content: str, image_url: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_history (user_id, role, content, image_url)
                VALUES ($1, $2, $3, $4)
                """,
                user_id, role, content, image_url
            )

    async def get_history(self, user_id: int) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, image_url, created_at FROM chat_history
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id, MAX_HISTORY
            )
            return [dict(row) for row in rows]


# ================== 3. KELAS AI ENGINE ==================
class PollinationsAI:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self.base_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"

    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Deteksi cerdas untuk memahami keinginan pengguna."""
        message_lower = message.lower()
        
        # Permintaan Audio
        if any(word in message_lower for word in ['suara', 'bicara', 'bilang', 'katakan', 'ngomong', 'speak', 'voice', 'audio', 'tts', 'say']):
            voice = 'alloy' # Default
            for v in AUDIO_VOICES:
                if v in message_lower:
                    voice = v
                    break
            return {'type': 'audio', 'voice': voice, 'text': message}
        
        # Permintaan Generasi Gambar
        image_keywords = ['gambar', 'buatkan', 'buat', 'lukis', 'generate', 'create', 'make', 'draw', 'image']
        if any(word in message_lower for word in image_keywords) and not has_photo:
            model = 'flux' # Default
            if any(word in message_lower for word in ['realistis', 'nyata', 'foto', 'realistic', 'photo', 'real']):
                model = 'flux-realism'
            elif any(word in message_lower for word in ['anime', 'kartun', 'manga', 'cartoon']):
                model = 'flux-anime'
            elif any(word in message_lower for word in ['3d', 'render', 'cgi']):
                model = 'flux-3d'
            elif any(word in message_lower for word in ['gelap', 'seram', 'dark', 'gothic', 'horror']):
                model = 'any-dark'
            elif any(word in message_lower for word in ['cepat', 'fast', 'quick']):
                model = 'turbo'
            
            return {'type': 'image', 'model': model, 'prompt': message}
        
        # Permintaan Transformasi Gambar (jika pengguna mengirim foto)
        if has_photo:
            transform_keywords = ['ubah', 'edit', 'transformasi', 'ganti', 'konversi', 'jadikan', 'jadi', 'transform', 'change', 'convert', 'into', 'style']
            if not message or not any(word in message_lower for word in transform_keywords):
                # Jika tidak ada teks atau kata kunci, defaultnya adalah analisis gambar
                return {'type': 'chat', 'model': 'gemini'}
            
            return {'type': 'image_transform', 'model': 'kontext', 'prompt': message}
        
        # Permintaan Coding/Pemrograman
        if any(word in message_lower for word in ['kode', 'coding', 'program', 'fungsi', 'skrip', 'debug', 'bug', 'code', 'function', 'script']):
            return {'type': 'chat', 'model': 'qwen-coder'}
        
        # Permintaan Penalaran/Analisis
        if any(word in message_lower for word in ['analisa', 'analisis', 'reason', 'penalaran', 'logika', 'kesimpulan', 'inferensi']):
            return {'type': 'chat', 'model': 'deepseek-r1'}
        
        # Permintaan Pencarian
        if any(word in message_lower for word in ['cari', 'search', 'temukan', 'google', 'lookup']):
            return {'type': 'chat', 'model': 'gemini-search'}
        
        # Default: gunakan model umum
        return {'type': 'chat', 'model': 'gpt-oss'}

    async def generate_text(self, prompt: str, model: str = "gpt-oss") -> str:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "prompt": prompt,
                    "model": model,
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            )
            if response.status_code == 200:
                return response.json().get("text", "Tidak ada respons dari AI.")
            else:
                return f"❌ Error AI: {response.status_code}"
        except Exception as e:
            return f"❌ Gagal menghubungi AI: {str(e)}"

    async def generate_image(self, prompt: str, model: str = "flux") -> str:
        try:
            url = f"{self.image_url}/p/{quote(prompt)}?model={model}&width=1024&height=1024&seed={random.randint(1, 10000)}"
            return url  # Langsung kembalikan URL gambar
        except Exception as e:
            return f"❌ Gagal membuat gambar: {str(e)}"


# Inisialisasi klien dan database
db = Database(DATABASE_URL)
ai_engine = PollinationsAI()
client = TelegramClient('bot', API_ID, API_HASH)

# ================== 4. FUNGSI BANTUAN ==================
async def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Membagi teks menjadi bagian-bagian yang lebih kecil tanpa memotong kata."""
    if len(text) <= max_length:
        return [text]

    parts = []
    while len(text) > max_length:
        # Cari pemisah kata terakhir dalam batas
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length  # Potong paksa jika tidak ada spasi
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts

# ================== 5. HANDLER UTAMA: handle_chat ==================
@client.on(events.NewMessage(incoming=True, pattern=r'.*'))
async def handle_chat(event):
    if event.message.out or not event.is_private:
        return  # Abaikan pesan keluar atau bukan DM

    user_id = event.sender_id
    username = getattr(event.sender, 'username', None)
    first_name = getattr(event.sender, 'first_name', 'Pengguna')

    # Dapatkan atau buat pengguna di database
    await db.get_or_create_user(user_id, username, first_name)

    # Cek verifikasi
    if not await db.is_verified(user_id):
        # Logika verifikasi channel/group
        try:
            for ch in REQUIRED_CHANNELS:
                await client.get_entity(ch)
            for gr in REQUIRED_GROUPS:
                await client.get_entity(gr)
            await db.set_verified(user_id, True)
        except Exception:
            # Kirim tombol verifikasi jika belum join
            markup = event.client.build_reply_markup([
                [Button.url("Gabung Channel", f"https://t.me/{REQUIRED_CHANNELS[0][1:]}")],
                [Button.url("Gabung Group", f"https://t.me/{REQUIRED_GROUPS[0][1:]}")],
                [Button.inline("✅ Cek Verifikasi", b"check_join")]
            ])
            await event.reply(
                "⚠️ Anda harus bergabung dengan channel dan group berikut untuk menggunakan bot ini:",
                buttons=markup
            )
            return

    # Deteksi niat pengguna
    has_photo = event.photo is not None
    intent = await ai_engine.detect_intent(event.message.message, has_photo)

    # Tampilkan "typing"
    async with client.action(event.chat_id, 'typing'):
        try:
            # Simulasi AI memproses (untuk UX)
            await asyncio.sleep(1)

            if intent['type'] == 'audio':
                # Fitur audio/TTS
                voice = intent['voice']
                text = intent['text']
                tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice={voice}&text={quote(text)}"
                await event.reply(f"🎧 Audio dari AI (suara: {voice}):", file=tts_url)

            elif intent['type'] == 'image':
                # Generasi gambar
                prompt = intent['prompt']
                model = intent['model']
                image_url = await ai_engine.generate_image(prompt, model)
                await event.reply(f"🎨 Gambar dari AI (model: {model}):")
                await event.respond(image_url)
                # Simpan riwayat
                await db.save_message(user_id, "user", event.message.message)
                await db.save_message(user_id, "assistant", f"Gambar dihasilkan dengan model {model}", image_url)

            elif intent['type'] == 'image_transform':
                # Transformasi gambar
                # Fitur ini memerlukan integrasi tambahan
                await event.reply("🖼️ Transformasi gambar sedang dalam pengembangan.")

            else:
                # Mode chat biasa
                # Ambil riwayat percakapan
                history = await db.get_history(user_id)
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

                # Prompt gabungan
                full_prompt = f"{context}\nuser: {event.message.message}\nassistant: "

                # Kirim ke AI
                model = intent['model']
                response = await ai_engine.generate_text(full_prompt, model=model)

                # Simpan ke database
                await db.save_message(user_id, "user", event.message.message)
                await db.save_message(user_id, "assistant", response)

                # Bagi pesan jika terlalu panjang
                parts = await split_message(response)

                # Kirim bagian pertama sebagai reply
                first_msg = await event.reply(f"🧠 AI: {parts[0]}")

                # Kirim sisa bagian sebagai pesan tambahan
                for part in parts[1:]:
                    await event.respond(part)

        except MessageTooLongError:
            # Backup plan: Pisah dan kirim
            parts = await split_message(f"🧠 AI: {response}")
            for i, part in enumerate(parts):
                if i == 0:
                    await event.reply(part)
                else:
                    await event.respond(part)
        except Exception as e:
            error_msg = f"❌ Terjadi kesalahan: {str(e)[:4000]}"
            await event.reply(error_msg)
            print(f"Error di handle_chat: {e}")
            traceback.print_exc()


# Tombol verifikasi
@client.on(events.CallbackQuery(data=b"check_join"))
async def handle_verification(event):
    user_id = event.sender_id
    try:
        for ch in REQUIRED_CHANNELS:
            await client.get_permissions(ch, user_id)
        for gr in REQUIRED_GROUPS:
            await client.get_permissions(gr, user_id)
        await db.set_verified(user_id, True)
        await event.edit("✅ Verifikasi berhasil! Anda sekarang dapat menggunakan bot.", buttons=None)
    except Exception:
        markup = event.client.build_reply_markup([
            [Button.url("Gabung Channel", f"https://t.me/{REQUIRED_CHANNELS[0][1:]}")],
            [Button.url("Gabung Group", f"https://t.me/{REQUIRED_GROUPS[0][1:]}")],
            [Button.inline("✅ Cek Verifikasi", b"check_join")]
        ])
        await event.edit("❌ Anda belum bergabung. Silakan gabung dulu.", buttons=markup)


# ================== 6. JALANKAN BOT ==================
async def main():
    await db.connect()
    print("✅ Bot siap digunakan.")
    await client.start(bot_token=BOT_TOKEN)
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())
