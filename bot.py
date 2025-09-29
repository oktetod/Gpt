# /proyek_bot/bot.py

import os
import asyncio
import random
from typing import List, Dict, Optional
from urllib.parse import quote
import asyncpg
from telethon import TelegramClient, events, Button
from telethon.errors.rpcerrorlist import UserNotParticipantError
from telethon.tl.types import DocumentAttributeAudio
import httpx

# ================== 1. KONFIGURASI ==================
API_ID = os.getenv('ID')
API_HASH = os.getenv('HASH')
BOT_TOKEN = os.getenv('BOT')
DATABASE_URL = "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

REQUIRED_CHANNELS = ['@durov69_1']
REQUIRED_GROUPS = ['@durov69_2']
MAX_HISTORY = 20

TEXT_MODELS = {
    "gpt-5": "openai", "gpt-5-mini": "openai-fast", "gpt-5-chat": "openai-large", "o4-mini": "openai-reasoning",
    "gemini": "gemini", "gemini-search": "gemini-search", "deepseek": "deepseek", "deepseek-r1": "deepseek-reasoning",
    "qwen-coder": "qwen-coder", "mistral": "mistral", "audio": "openai-audio", "evil": "evil", "unity": "unity"
}
IMAGE_MODELS = {
    "flux": "Kualitas seimbang", "flux-realism": "Fotorealistis", "flux-anime": "Gaya anime",
    "flux-3d": "Gaya render 3D", "any-dark": "Estetika gelap", "turbo": "Generasi cepat", "kontext": "Transformasi gambar"
}
AUDIO_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage"]

# ================== 2. KELAS DATABASE ==================
class Database:
    def __init__(self, url: str):
        self.url = url
        self.pool = None
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.url, min_size=5, max_size=20, command_timeout=60)
            await self.init_tables()
            print("Koneksi database berhasil dibuat.")
    
    async def init_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY, username TEXT, first_name TEXT, is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY, user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE, role TEXT NOT NULL,
                    content TEXT NOT NULL, image_url TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, created_at DESC)')
    
    async def get_or_create_user(self, user_id: int, username: str = None, first_name: str = None):
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            if not user:
                await conn.execute('INSERT INTO users (user_id, username, first_name) VALUES ($1, $2, $3)', user_id, username, first_name)
            else:
                await conn.execute('UPDATE users SET last_active = NOW() WHERE user_id = $1', user_id)
            return await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
    
    async def set_verified(self, user_id: int, verified: bool = True):
        async with self.pool.acquire() as conn:
            await conn.execute('UPDATE users SET is_verified = $1 WHERE user_id = $2', verified, user_id)
    
    async def is_verified(self, user_id: int) -> bool:
        async with self.pool.acquire() as conn:
            return await conn.fetchval('SELECT is_verified FROM users WHERE user_id = $1', user_id) or False
    
    async def add_message(self, user_id: int, role: str, content: str, image_url: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute('INSERT INTO chat_history (user_id, role, content, image_url) VALUES ($1, $2, $3, $4)', user_id, role, content, image_url)
            await conn.execute('DELETE FROM chat_history WHERE id IN (SELECT id FROM chat_history WHERE user_id = $1 ORDER BY created_at DESC OFFSET $2)', user_id, MAX_HISTORY * 2)
    
    async def get_history(self, user_id: int) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT role, content, image_url FROM chat_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2', user_id, MAX_HISTORY * 2)
            return [dict(row) for row in reversed(rows)]
    
    async def clear_history(self, user_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM chat_history WHERE user_id = $1', user_id)

# ================== 3. KELAS GATEKEEPER ==================
class Gatekeeper:
    def __init__(self, client: TelegramClient):
        self.client = client
    
    async def check_membership(self, user_id: int) -> tuple[bool, list]:
        not_joined = []
        for entity in REQUIRED_CHANNELS + REQUIRED_GROUPS:
            try:
                await self.client.get_permissions(entity, user_id)
            except UserNotParticipantError:
                not_joined.append(entity)
            except Exception:
                not_joined.append(entity)
        return len(not_joined) == 0, not_joined
    
    def get_verification_message(self, not_joined: list) -> tuple[str, list]:
        message = "🔐 **Verifikasi Dibutuhkan**\n\nUntuk menggunakan bot, Anda wajib bergabung dengan:\n\n"
        buttons = []
        for entity in not_joined:
            label = f"📢 Channel {entity}" if entity in REQUIRED_CHANNELS else f"👥 Grup {entity}"
            message += f"• {label}\n"
            buttons.append([Button.url(label, f"https://t.me/{entity.replace('@', '')}")])
        message += "\n✅ Setelah bergabung, kirim /start lagi."
        return message, buttons

# ================== 4. KELAS AI ENGINE ==================
class AiEngine:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self.base_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        message_lower = message.lower()
        if any(w in message_lower for w in ['suara', 'bicara', 'bilang', 'katakan', 'ngomong', 'speak', 'voice', 'audio']):
            voice = next((v for v in AUDIO_VOICES if v in message_lower), 'alloy')
            return {'type': 'audio', 'voice': voice, 'text': message}
        if any(w in message_lower for w in ['gambar', 'buatkan', 'buat', 'lukis', 'generate', 'create', 'image']) and not has_photo:
            model = 'flux'
            if any(w in message_lower for w in ['realistis', 'nyata', 'foto', 'realistic']): model = 'flux-realism'
            elif any(w in message_lower for w in ['anime', 'kartun', 'manga']): model = 'flux-anime'
            elif any(w in message_lower for w in ['3d', 'render']): model = 'flux-3d'
            return {'type': 'image', 'model': model, 'prompt': message}
        if has_photo:
            if not message or not any(w in message_lower for w in ['ubah', 'edit', 'transformasi', 'ganti', 'jadikan']):
                return {'type': 'chat', 'model': 'gemini'}
            return {'type': 'image_transform', 'model': 'kontext', 'prompt': message}
        if any(w in message_lower for w in ['kode', 'coding', 'program', 'fungsi', 'skrip', 'debug']): return {'type': 'chat', 'model': 'qwen-coder'}
        if any(w in message_lower for w in ['analisa', 'analisis', 'pikirkan', 'jelaskan', 'mengapa']): return {'type': 'chat', 'model': 'deepseek-r1'}
        if any(w in message_lower for w in ['cari', 'berita', 'terbaru', 'informasi', 'search']): return {'type': 'chat', 'model': 'gemini-search'}
        return {'type': 'chat', 'model': 'gpt-5'}

    async def chat(self, messages: List[Dict], model: str = "gpt-5") -> str:
        try:
            model_name = TEXT_MODELS.get(model, TEXT_MODELS['gpt-5'])
            response = await self.client.post(f"{self.base_url}/openai", json={"model": model_name, "messages": messages, "max_tokens": 4096})
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "Maaf, terjadi kesalahan.")
        except Exception as e: return f"❌ Error saat menghubungi AI: {str(e)}"

    def generate_image_url(self, prompt: str, model: str = "flux") -> str:
        seed = random.randint(1, 1000000)
        return f"{self.image_url}/prompt/{quote(prompt)}?model={model}&width=1024&height=1024&seed={seed}&nologo=true"

    def transform_image_url(self, prompt: str, image_url: str) -> str:
        return f"{self.image_url}/prompt/{quote(prompt)}?model=kontext&image={quote(image_url)}&nologo=true"

    def generate_audio_url(self, text: str, voice: str = "alloy") -> str:
        return f"{self.base_url}/{quote(text)}?model=openai-audio&voice={voice}"

    async def enhance_prompt(self, prompt: str) -> str:
        if len(prompt.split()) > 25: return prompt
        messages = [{"role": "user", "content": f"Enhance this image prompt to be artistic and descriptive (max 50 words, in English): {prompt}"}]
        enhanced = await self.chat(messages, "gpt-5-mini")
        return enhanced.strip().replace('"', '')
    
    async def close(self): await self.client.aclose()

# ================== 5. KELAS UTAMA BOT ==================
class SmartAIBot:
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.db = Database(DATABASE_URL)
        self.ai = AiEngine()
        self.gatekeeper = Gatekeeper(self.client)

    async def start(self):
        await self.client.start(bot_token=BOT_TOKEN)
        await self.db.connect()
        me = await self.client.get_me()
        print(f"✅ Bot aktif: @{me.username}")
        print(f"🤖 Powered by @durov9369")
        self.register_handlers()
        await self.client.run_until_disconnected()

    async def upload_to_telegraph(self, image_bytes: bytes) -> Optional[str]:
        try:
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            async with httpx.AsyncClient() as client:
                response = await client.post('https://telegra.ph/upload', files=files)
            if response.status_code == 200 and (data := response.json()):
                return f"https://telegra.ph{data[0]['src']}"
        except Exception as e:
            print(f"Error unggah ke Telegraph: {e}")
        return None

    # --- HANDLER DENGAN LOGIKA EDIT PESAN ---
    async def handle_audio(self, event, intent):
        status_msg = await event.respond("🎙️ Mempersiapkan studio rekaman...")
        try:
            text, voice = intent['text'], intent['voice']
            await status_msg.edit(f"🎙️ Sedang merekam suara dengan voice **{voice}**...")
            audio_url = self.ai.generate_audio_url(text, voice)
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=120.0)
                response.raise_for_status()
            await self.client.send_file(event.chat_id, file=response.content, voice_note=True)
            await status_msg.delete()
        except Exception as e:
            await status_msg.edit(f"❌ Gagal membuat audio: {str(e)}")

    async def handle_image_generation(self, event, intent):
        status_msg = await event.respond("🎨 Menyiapkan kanvas...")
        try:
            prompt, model = intent['prompt'], intent['model']
            await status_msg.edit("💡 Memperkaya imajinasi (prompt enhancement)...")
            enhanced_prompt = await self.ai.enhance_prompt(prompt)
            await status_msg.edit(f"🖌️ Sedang melukis gambar dengan model **{model}**...")
            image_url = self.ai.generate_image_url(enhanced_prompt, model)
            await event.respond(file=image_url, message=f"🎨 **Karya Seni Selesai**\n\n**Imajinasi:** `{enhanced_prompt}`")
            await status_msg.delete()
            await self.db.add_message(event.sender_id, 'user', f"Buat gambar: {prompt}")
            await self.db.add_message(event.sender_id, 'assistant', f"Gambar dibuat", image_url)
        except Exception as e:
            await status_msg.edit(f"❌ Gagal melukis: {str(e)}")

    async def handle_image_transform(self, event, intent, image_url):
        status_msg = await event.respond("✨ Mempersiapkan sihir transformasi...")
        try:
            prompt = intent['prompt'] or "tingkatkan kualitas gambar ini"
            await status_msg.edit(f"✨ Merapal mantra transformasi pada gambar...")
            transform_url = self.ai.transform_image_url(prompt, image_url)
            await event.respond(file=transform_url, message=f"✨ **Transformasi Berhasil**\n\n**Mantra:** `{prompt}`")
            await status_msg.delete()
            await self.db.add_message(event.sender_id, 'user', f"Transformasi: {prompt}", image_url)
            await self.db.add_message(event.sender_id, 'assistant', "Hasil transformasi", transform_url)
        except Exception as e:
            await status_msg.edit(f"❌ Gagal transformasi: {str(e)}")

    async def handle_chat(self, event, intent, message_text, image_url=None):
        user_id = event.sender_id
        model = intent['model']
        emoji_map = {'gpt-5':'🤖', 'qwen-coder':'💻', 'deepseek-r1':'🧠', 'gemini-search':'🔍', 'gemini':'👁️'}
        status_msg = await event.respond(f"{emoji_map.get(model, '🤖')} Sedang berpikir...")
        try:
            content = message_text or "Jelaskan gambar ini secara detail."
            await self.db.add_message(user_id, 'user', content, image_url)
            history = await self.db.get_history(user_id)
            messages = [{"role": m['role'], "content": f"[Gambar: {m['image_url']}]\n{m['content']}" if m.get('image_url') else m['content']} for m in history]
            
            response = await self.ai.chat(messages, model)
            await self.db.add_message(user_id, 'assistant', response)
            model_name = model.replace('-', ' ').upper()
            await status_msg.edit(f"{emoji_map.get(model, '🤖')} **{model_name}**\n\n{response}", parse_mode='markdown')
        except Exception as e:
            await status_msg.edit(f"❌ Terjadi kesalahan saat chat: {str(e)}")

    # --- REGISTRASI EVENT ---
    def register_handlers(self):
        @self.client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            user = await event.get_sender()
            await self.db.get_or_create_user(user.id, user.username, user.first_name)
            if not await self.check_verification(event): return
            await event.respond(
                f"🤖 **Bot AI Cerdas Aktif**\n\nHalo **{user.first_name}**! Saya siap membantu Anda.\n\n"
                "**🎯 Kemampuan Utama:**\n"
                "💬 Chat Cerdas (GPT-5)\n🎨 Membuat Gambar\n✏️ Mengedit Gambar\n"
                "👁️ Menganalisis Foto\n🎙️ Teks ke Suara\n💻 Bantuan Koding\n\n"
                "Kirim pesan atau gambar untuk memulai. Gunakan `/help` untuk panduan.",
                parse_mode='markdown'
            )

        @self.client.on(events.NewMessage(pattern='/clear'))
        async def clear_handler(event):
            if not await self.check_verification(event): return
            await self.db.clear_history(event.sender_id)
            await event.respond("🗑️ Riwayat percakapan Anda telah dihapus.")
        
        @self.client.on(events.NewMessage(pattern='/help'))
        async def help_handler(event):
            if not await self.check_verification(event): return
            await event.respond(
                "📖 **Panduan Lengkap Bot**\n\n"
                "**🎨 Gambar:**\n`buatkan gambar pemandangan senja`\n`lukis gadis anime rambut biru`\n\n"
                "**👁️ Foto:**\nKirim foto untuk dianalisis, atau kirim dengan perintah: `ubah jadi kartun`\n\n"
                "**🎙️ Suara:**\n`katakan halo dengan suara nova`\n(Suara: `alloy, echo, fable, onyx, nova, shimmer`)\n\n"
                "**💻 Koding:**\n`buatkan kode python untuk ...`\n`debug kode ini ...`\n\n"
                "**🔍 Info:**\n`cari berita terbaru tentang AI`\n"
                "Bot akan otomatis mendeteksi keinginan Anda.",
                parse_mode='markdown'
            )

        @self.client.on(events.NewMessage(incoming=True, func=lambda e: not e.text.startswith('/')))
        async def message_handler(event):
            if not await self.check_verification(event): return
            
            message_text = event.message.text or ""
            has_photo = bool(event.message.photo)
            
            image_url = None
            if has_photo:
                status_msg = await event.respond("📸 Foto diterima, sedang diunggah...")
                try:
                    photo_bytes = await event.message.download_media(bytes)
                    image_url = await self.upload_to_telegraph(photo_bytes)
                    if not image_url:
                        await status_msg.edit("❌ Gagal mengunggah foto. Coba lagi.")
                        return
                    await status_msg.delete()
                except Exception as e:
                    await status_msg.edit(f"❌ Error memproses foto: {e}")
                    return

            intent = await self.ai.detect_intent(message_text, has_photo)
            
            if intent['type'] == 'audio': await self.handle_audio(event, intent)
            elif intent['type'] == 'image': await self.handle_image_generation(event, intent)
            elif intent['type'] == 'image_transform': await self.handle_image_transform(event, intent, image_url)
            elif intent['type'] == 'chat': await self.handle_chat(event, intent, message_text, image_url)
            
    async def check_verification(self, event) -> bool:
        if await self.db.is_verified(event.sender_id): return True
        is_member, not_joined = await self.gatekeeper.check_membership(event.sender_id)
        if not is_member:
            message, buttons = self.gatekeeper.get_verification_message(not_joined)
            await event.respond(message, buttons=buttons, parse_mode='markdown')
            return False
        await self.db.set_verified(event.sender_id, True)
        return True

# ================== 6. FUNGSI UTAMA UNTUK MENJALANKAN BOT ==================
async def main():
    bot = SmartAIBot()
    try:
        await bot.start()
    except Exception as e:
        print(f"❌ Bot berhenti karena error fatal: {e}")
    finally:
        await bot.ai.close()
        await bot.client.disconnect()
        print("🛑 Bot telah berhenti.")

if __name__ == '__main__':
    asyncio.run(main())
