import os
import asyncio
import re
import random
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote
import asyncpg
from telethon import TelegramClient, events, Button
import httpx

# ================== KONFIGURASI ==================
API_ID = os.getenv('10446785')
API_HASH = os.getenv('4261b62d60200eb99a38dcd8b71c8634')
BOT_TOKEN = os.getenv('8493708418:AAEjvF837afIBDs4OZSvuPFG3F6O6P1_D7U')
DATABASE_URL = "postgresql://postgres.kzmeyjdceukikzazbjjy:gilpad008@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

# Channels dan Groups wajib
REQUIRED_CHANNELS = ['@durov69_1']
REQUIRED_GROUPS = ['@durov69_2']

MAX_HISTORY = 20

# ================== POLLINATIONS.AI MODELS ==================
TEXT_MODELS = {
    # Premium Models
    "gpt-5": "openai",  # Default - GPT-5 Free
    "gpt-5-mini": "openai-fast",
    "gpt-5-chat": "openai-large",
    "o4-mini": "openai-reasoning",
    
    # Vision Models
    "gemini": "gemini",
    "gemini-search": "gemini-search",
    
    # Specialized Models
    "deepseek": "deepseek",
    "deepseek-r1": "deepseek-reasoning",
    "qwen-coder": "qwen-coder",
    "mistral": "mistral",
    
    # Audio
    "audio": "openai-audio",
    
    # Community
    "evil": "evil",
    "unity": "unity"
}

IMAGE_MODELS = {
    "flux": "Default - Balanced quality",
    "flux-realism": "Photorealistic images",
    "flux-anime": "Anime style",
    "flux-3d": "3D rendered style",
    "any-dark": "Dark aesthetic",
    "turbo": "Fast generation",
    "kontext": "Image-to-image transformation"
}

AUDIO_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage"]

# ================== DATABASE ==================
class Database:
    def __init__(self, url: str):
        self.url = url
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        await self.init_tables()
    
    async def init_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_active TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
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
    
    async def set_verified(self, user_id: int, verified: bool = True):
        async with self.pool.acquire() as conn:
            await conn.execute(
                'UPDATE users SET is_verified = $1 WHERE user_id = $2',
                verified, user_id
            )
    
    async def is_verified(self, user_id: int) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                'SELECT is_verified FROM users WHERE user_id = $1', user_id
            )
            return result or False
    
    async def add_message(self, user_id: int, role: str, content: str, image_url: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO chat_history (user_id, role, content, image_url) VALUES ($1, $2, $3, $4)',
                user_id, role, content, image_url
            )
            
            # Limit history
            await conn.execute('''
                DELETE FROM chat_history
                WHERE id IN (
                    SELECT id FROM chat_history
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    OFFSET $2
                )
            ''', user_id, MAX_HISTORY * 2)
    
    async def get_history(self, user_id: int, limit: int = MAX_HISTORY) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                'SELECT role, content, image_url FROM chat_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2',
                user_id, limit
            )
            return [dict(row) for row in reversed(rows)]
    
    async def clear_history(self, user_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM chat_history WHERE user_id = $1', user_id)

# ================== GATEKEEPER ==================
class Gatekeeper:
    def __init__(self, client: TelegramClient):
        self.client = client
    
    async def check_membership(self, user_id: int) -> tuple[bool, list]:
        not_joined = []
        all_required = REQUIRED_CHANNELS + REQUIRED_GROUPS
        
        for entity_username in all_required:
            try:
                entity = await self.client.get_entity(entity_username)
                try:
                    participant = await self.client.get_permissions(entity, user_id)
                    if participant is None:
                        not_joined.append(entity_username)
                except:
                    not_joined.append(entity_username)
            except Exception as e:
                print(f"Error checking {entity_username}: {e}")
                not_joined.append(entity_username)
        
        return len(not_joined) == 0, not_joined
    
    def get_verification_message(self, not_joined: list) -> tuple[str, list]:
        message = "🔐 **Verifikasi Diperlukan**\n\n"
        message += "Untuk menggunakan bot, join channel & group berikut:\n\n"
        
        buttons = []
        for entity in not_joined:
            entity_clean = entity.replace('@', '')
            if entity in REQUIRED_CHANNELS:
                label = f"📢 {entity}"
            else:
                label = f"👥 {entity}"
            
            message += f"• {label}\n"
            buttons.append([Button.url(label, f"https://t.me/{entity_clean}")])
        
        message += "\n✅ Setelah join semua, kirim /start lagi"
        return message, buttons

# ================== AI SMART ENGINE ==================
class PollinationsAI:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
        self.base_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Smart detection: apa yang user inginkan?"""
        message_lower = message.lower()
        
        # Audio requests
        if any(word in message_lower for word in ['speak', 'voice', 'audio', 'tts', 'say', 'suara', 'bicara']):
            voice = 'alloy'
            for v in AUDIO_VOICES:
                if v in message_lower:
                    voice = v
                    break
            return {
                'type': 'audio',
                'voice': voice,
                'text': message
            }
        
        # Image generation requests
        image_keywords = ['generate', 'create', 'make', 'draw', 'gambar', 'buat', 'buatkan', 'image', 'picture', 'photo']
        if any(word in message_lower for word in image_keywords) and not has_photo:
            # Detect style
            model = 'flux'
            if any(word in message_lower for word in ['realistic', 'photo', 'real', 'realistis']):
                model = 'flux-realism'
            elif any(word in message_lower for word in ['anime', 'manga', 'cartoon']):
                model = 'flux-anime'
            elif any(word in message_lower for word in ['3d', 'render', 'cgi']):
                model = 'flux-3d'
            elif any(word in message_lower for word in ['dark', 'gothic', 'horror']):
                model = 'any-dark'
            elif any(word in message_lower for word in ['fast', 'quick', 'cepat']):
                model = 'turbo'
            
            return {
                'type': 'image',
                'model': model,
                'prompt': message
            }
        
        # Image transformation (user send photo)
        if has_photo:
            transform_keywords = ['ubah', 'edit', 'transform', 'change', 'modify', 'convert', 'jadi', 'into', 'to', 'style', 'buat']
            
            # If no text or no transform keywords, just analyze
            if not message or not any(word in message_lower for word in transform_keywords):
                # Use vision model to analyze image
                return {
                    'type': 'chat',
                    'model': 'gemini'
                }
            
            return {
                'type': 'image_transform',
                'model': 'kontext',
                'prompt': message
            }
        
        # Code/programming requests
        if any(word in message_lower for word in ['code', 'program', 'function', 'script', 'debug', 'bug', 'kode', 'coding']):
            return {
                'type': 'chat',
                'model': 'qwen-coder'
            }
        
        # Reasoning/analysis requests
        if any(word in message_lower for word in ['analyze', 'think', 'reason', 'explain', 'why', 'analisa', 'jelaskan', 'mengapa']):
            return {
                'type': 'chat',
                'model': 'deepseek-r1'
            }
        
        # Search/current info requests
        if any(word in message_lower for word in ['search', 'find', 'latest', 'news', 'cari', 'terbaru', 'berita']):
            return {
                'type': 'chat',
                'model': 'gemini-search'
            }
        
        # Default: GPT-5
        return {
            'type': 'chat',
            'model': 'gpt-5'
        }
    
    async def chat(self, messages: List[Dict], model: str = "gpt-5") -> str:
        """Chat dengan AI"""
        try:
            model_name = TEXT_MODELS.get(model, TEXT_MODELS['gpt-5'])
            
            response = await self.client.post(
                f"{self.base_url}/openai",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 4096
                },
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "Maaf, terjadi kesalahan.")
        except Exception as e:
            print(f"Chat error: {e}")
            return f"❌ Error: {str(e)}"
    
    def generate_image_url(self, prompt: str, model: str = "flux", width: int = 1024, 
                          height: int = 1024, seed: int = None) -> str:
        """Generate image URL"""
        if seed is None:
            seed = random.randint(1, 1000000)
        
        encoded_prompt = quote(prompt)
        url = f"{self.image_url}/prompt/{encoded_prompt}?model={model}&width={width}&height={height}&seed={seed}&nologo=true"
        return url
    
    def transform_image_url(self, prompt: str, image_url: str, model: str = "kontext") -> str:
        """Transform existing image"""
        encoded_prompt = quote(prompt)
        encoded_image = quote(image_url)
        url = f"{self.image_url}/prompt/{encoded_prompt}?model={model}&image={encoded_image}&nologo=true"
        return url
    
    def generate_audio_url(self, text: str, voice: str = "alloy") -> str:
        """Generate audio URL"""
        encoded_text = quote(text)
        url = f"{self.base_url}/{encoded_text}?model=openai-audio&voice={voice}"
        return url
    
    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt untuk image generation"""
        if len(prompt.split()) > 30:
            return prompt
        
        enhance_request = f"Enhance this image prompt to be detailed and descriptive (max 50 words): {prompt}"
        messages = [{"role": "user", "content": enhance_request}]
        enhanced = await self.chat(messages, "gpt-5")
        return enhanced.strip()
    
    async def close(self):
        await self.client.aclose()

# ================== TELEGRAM BOT ==================
class SmartAIBot:
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.db = Database(DATABASE_URL)
        self.gatekeeper = None
        self.ai = PollinationsAI()
    
    async def start(self):
        await self.client.start(bot_token=BOT_TOKEN)
        await self.db.connect()
        self.gatekeeper = Gatekeeper(self.client)
        
        me = await self.client.get_me()
        print(f"✅ Bot aktif: @{me.username}")
        print(f"🤖 Powered by Pollinations.AI")
        print(f"🔐 Security: Gatekeeper enabled")
        
        self.register_handlers()
        await self.client.run_until_disconnected()
    
    async def upload_to_telegraph(self, image_bytes: bytes) -> Optional[str]:
        """Upload image to Telegraph for public URL"""
        try:
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            response = await self.ai.client.post(
                'https://telegra.ph/upload',
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return f"https://telegra.ph{data[0]['src']}"
            
            return None
        except Exception as e:
            print(f"Telegraph upload error: {e}")
            return None
    
    def register_handlers(self):
        @self.client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            user = await event.get_sender()
            await self.db.get_or_create_user(user.id, user.username, user.first_name)
            
            is_verified = await self.db.is_verified(user.id)
            
            if not is_verified:
                is_member, not_joined = await self.gatekeeper.check_membership(user.id)
                
                if not is_member:
                    message, buttons = self.gatekeeper.get_verification_message(not_joined)
                    await event.respond(message, buttons=buttons, parse_mode='markdown')
                    return
                else:
                    await self.db.set_verified(user.id, True)
            
            await event.respond(
                f"🤖 **Smart AI Bot Active**\n\n"
                f"Halo **{user.first_name}**! Saya bot AI yang cerdas dan serba bisa.\n\n"
                f"**🎯 Kemampuan Saya:**\n"
                f"💬 Chat cerdas dengan GPT-5\n"
                f"🎨 Generate gambar dari deskripsi\n"
                f"✏️ Edit & transform gambar\n"
                f"👁️ Analisis foto dengan Vision AI\n"
                f"🎙️ Text-to-speech audio\n"
                f"💻 Coding & debugging\n"
                f"🔍 Search & analisis\n"
                f"🧠 Deep reasoning\n\n"
                f"**✨ Cara Pakai:**\n"
                f"• Kirim pesan untuk chat biasa\n"
                f"• Minta \"buat gambar...\" untuk generate\n"
                f"• Kirim foto untuk analisis otomatis\n"
                f"• Kirim foto + \"ubah jadi...\" untuk edit\n"
                f"• Minta \"say...\" untuk audio\n"
                f"• Tanya coding untuk bantuan kode\n\n"
                f"**📝 Command:**\n"
                f"/clear - Hapus history\n"
                f"/help - Bantuan lengkap\n\n"
                f"💡 Saya akan otomatis pilih model AI terbaik untuk request Anda!",
                parse_mode='markdown'
            )
        
        @self.client.on(events.NewMessage(pattern='/clear'))
        async def clear_handler(event):
            if not await self.check_verification(event):
                return
            
            await self.db.clear_history(event.sender_id)
            await event.respond("🗑️ History berhasil dihapus! Mulai percakapan baru.")
        
        @self.client.on(events.NewMessage(pattern='/help'))
        async def help_handler(event):
            if not await self.check_verification(event):
                return
            
            await event.respond(
                "📖 **Panduan Lengkap Smart AI Bot**\n\n"
                "**🎨 Generate Gambar:**\n"
                "• \"buat gambar pemandangan gunung\"\n"
                "• \"create anime girl with blue hair\"\n"
                "• \"gambar 3D robot futuristik\"\n\n"
                "**👁️ Analisis Foto:**\n"
                "• Kirim foto tanpa caption\n"
                "• Kirim foto + \"apa ini?\"\n"
                "• Kirim foto + \"jelaskan gambar ini\"\n\n"
                "**✏️ Edit Gambar:**\n"
                "• Kirim foto + \"ubah jadi anime style\"\n"
                "• Kirim foto + \"buat lebih realistic\"\n"
                "• Kirim foto + \"transform to cyberpunk\"\n\n"
                "**🎙️ Audio (Text-to-Speech):**\n"
                "• \"say hello world with voice nova\"\n"
                "• \"speak this text with echo voice\"\n"
                "• Available voices: alloy, echo, fable, onyx, nova, shimmer\n\n"
                "**💻 Coding Help:**\n"
                "• \"code python function for...\"\n"
                "• \"debug this code...\"\n"
                "• \"explain this algorithm...\"\n\n"
                "**🔍 Search & Info:**\n"
                "• \"search latest news about...\"\n"
                "• \"find information on...\"\n"
                "• \"what's the latest...\"\n\n"
                "**🧠 Deep Analysis:**\n"
                "• \"analyze this situation...\"\n"
                "• \"explain why...\"\n"
                "• \"think about...\"\n\n"
                "**💡 Tips:**\n"
                "• Bot otomatis deteksi intent Anda\n"
                "• Semakin detail request, semakin baik hasil\n"
                "• History disimpan 20 pesan terakhir\n"
                "• Semua powered by Pollinations.AI\n\n"
                "Silakan kirim pesan untuk mencoba! 🚀",
                parse_mode='markdown'
            )
        
        @self.client.on(events.NewMessage(incoming=True, func=lambda e: not e.message.text.startswith('/')))
        async def message_handler(event):
            if not await self.check_verification(event):
                return
            
            user_id = event.sender_id
            message_text = event.message.text or ""
            has_photo = bool(event.message.photo)
            
            # Download and upload photo if exists
            image_url = None
            if has_photo:
                try:
                    status_msg = await event.respond("📸 Foto diterima! Sedang upload...")
                    
                    # Download photo
                    photo = await event.message.download_media(bytes)
                    
                    # Upload to Telegraph
                    image_url = await self.upload_to_telegraph(photo)
                    
                    if image_url:
                        await status_msg.edit(f"✅ Foto berhasil di-upload! Memproses...")
                    else:
                        await status_msg.edit("❌ Gagal upload foto. Coba lagi.")
                        return
                        
                except Exception as e:
                    await event.respond(f"❌ Error processing foto: {str(e)}")
                    return
            
            # Smart intent detection
            intent = await self.ai.detect_intent(message_text, has_photo)
            
            # Process based on intent
            if intent['type'] == 'audio':
                await self.handle_audio(event, intent)
            
            elif intent['type'] == 'image':
                await self.handle_image_generation(event, intent)
            
            elif intent['type'] == 'image_transform':
                await self.handle_image_transform(event, intent, image_url)
            
            elif intent['type'] == 'chat':
                await self.handle_chat(event, intent, message_text, image_url)
        
        async def handle_audio(self, event, intent):
            """Handle audio generation"""
            text = intent['text']
            voice = intent['voice']
            
            await event.respond(f"🎙️ Generating audio dengan voice **{voice}**...", parse_mode='markdown')
            
            # Generate audio URL
            audio_url = self.ai.generate_audio_url(text, voice)
            
            try:
                # Download and send
                response = await self.ai.client.get(audio_url)
                audio_data = response.content
                
                await event.respond(
                    file=audio_data,
                    attributes=[],
                    voice_note=True
                )
                await event.respond(f"✅ Audio berhasil! Voice: {voice}")
            except Exception as e:
                await event.respond(f"❌ Error generating audio: {str(e)}")
        
        async def handle_image_generation(self, event, intent):
            """Handle image generation"""
            model = intent['model']
            prompt = intent['prompt']
            
            await event.respond(
                f"🎨 Generating image dengan **{model}** model...\n"
                f"💡 Prompt: {prompt[:100]}...",
                parse_mode='markdown'
            )
            
            # Enhance prompt
            enhanced_prompt = await self.ai.enhance_prompt(prompt)
            
            # Generate image
            seed = random.randint(1, 1000000)
            image_url = self.ai.generate_image_url(enhanced_prompt, model, seed=seed)
            
            try:
                await event.respond(
                    file=image_url,
                    message=f"🎨 **Generated Image**\n\n"
                           f"📝 Prompt: {enhanced_prompt[:200]}\n"
                           f"🎭 Model: {model}\n"
                           f"🎲 Seed: {seed}\n\n"
                           f"💡 Ubah prompt untuk variasi baru!",
                    parse_mode='markdown'
                )
                
                # Save to history
                user_id = event.sender_id
                await self.db.add_message(user_id, 'user', f"Generate: {prompt}")
                await self.db.add_message(user_id, 'assistant', f"Generated image: {model}", image_url)
            except Exception as e:
                await event.respond(f"❌ Error: {str(e)}")
        
        async def handle_image_transform(self, event, intent, image_url):
            """Handle image transformation"""
            if not image_url:
                await event.respond("❌ Gagal mendapatkan URL foto. Silakan coba lagi.")
                return
            
            prompt = intent['prompt'] or "enhance this image"
            model = intent['model']
            
            await event.respond(
                f"✏️ **Transforming Image...**\n\n"
                f"📝 Instruction: {prompt}\n"
                f"🎭 Model: {model}\n\n"
                f"⏳ Mohon tunggu...",
                parse_mode='markdown'
            )
            
            transform_url = self.ai.transform_image_url(prompt, image_url, model)
            
            try:
                await event.respond(
                    file=transform_url,
                    message=f"✨ **Transformed Image**\n\n"
                           f"📝 {prompt}\n"
                           f"🎭 Model: {model}\n"
                           f"🔗 Original: {image_url[:50]}...",
                    parse_mode='markdown'
                )
                
                # Save to history
                user_id = event.sender_id
                await self.db.add_message(user_id, 'user', f"Transform: {prompt}", image_url)
                await self.db.add_message(user_id, 'assistant', f"Transformed with {model}", transform_url)
                
            except Exception as e:
                await event.respond(f"❌ Error transforming image: {str(e)}\n\n💡 Coba dengan instruksi berbeda.")
        
        async def handle_chat(self, event, intent, message_text, image_url=None):
            """Handle chat conversation"""
            user_id = event.sender_id
            model = intent['model']
            
            # Prepare message content
            if image_url and model in ['gemini', 'gemini-search', 'openai', 'openai-large', 'openai-reasoning']:
                # Vision-capable models
                if not message_text:
                    message_text = "Describe this image in detail."
                message_content = f"[Analyzing image: {image_url}]\n\n{message_text}"
                await self.db.add_message(user_id, 'user', message_content, image_url)
            else:
                await self.db.add_message(user_id, 'user', message_text, image_url)
            
            # Get history
            history = await self.db.get_history(user_id, MAX_HISTORY)
            
            # Prepare messages
            messages = []
            for msg in history:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            # Show typing
            async with self.client.action(event.chat_id, 'typing'):
                # Get AI response
                model_emoji = {
                    'gpt-5': '🤖',
                    'qwen-coder': '💻',
                    'deepseek-r1': '🧠',
                    'gemini-search': '🔍',
                    'gemini': '👁️'
                }.get(model, '🤖')
                
                response = await self.ai.chat(messages, model)
                
                # Save AI response
                await self.db.add_message(user_id, 'assistant', response)
                
                # Send response
                model_name = model.replace('-', ' ').upper()
                await event.respond(
                    f"{model_emoji} **{model_name}**\n\n{response}",
                    parse_mode='markdown'
                )
    
    async def check_verification(self, event) -> bool:
        is_verified = await self.db.is_verified(event.sender_id)
        
        if not is_verified:
            is_member, not_joined = await self.gatekeeper.check_membership(event.sender_id)
            
            if not is_member:
                message, buttons = self.gatekeeper.get_verification_message(not_joined)
                await event.respond(message, buttons=buttons, parse_mode='markdown')
                return False
            else:
                await self.db.set_verified(event.sender_id, True)
                return True
        
        return True

# ================== MAIN ==================
async def main():
    bot = SmartAIBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await bot.ai.close()
        await bot.client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
