# /proyek_bot/bot.py

import asyncio
from typing import Optional

from telethon import TelegramClient, events
from telethon.tl.types import DocumentAttributeAudio

from config import API_ID, API_HASH, BOT_TOKEN, DATABASE_URL
from database import Database
from gatekeeper import Gatekeeper
from ai_engine import PollinationsAI

class SmartAIBot:
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.db = Database(DATABASE_URL)
        self.ai = PollinationsAI()
        self.gatekeeper = Gatekeeper(self.client) # Inisialisasi langsung

    async def start(self):
        """Memulai bot, koneksi database, dan mendaftarkan handler."""
        await self.client.start(bot_token=BOT_TOKEN)
        await self.db.connect()
        
        me = await self.client.get_me()
        print(f"✅ Bot aktif: @{me.username}")
        print(f"🤖 Powered by @durov9369")
        print(f"🔐 Keamanan: Gatekeeper diaktifkan")
        
        self.register_handlers()
        await self.client.run_until_disconnected()

    async def upload_to_telegraph(self, image_bytes: bytes) -> Optional[str]:
        """Mengunggah gambar ke Telegra.ph untuk mendapatkan URL publik."""
        try:
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            async with httpx.AsyncClient() as client:
                response = await client.post('https://telegra.ph/upload', files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0 and 'src' in data[0]:
                    return f"https://telegra.ph{data[0]['src']}"
            return None
        except Exception as e:
            print(f"Error saat unggah ke Telegraph: {e}")
            return None

    # ================== HANDLER UTAMA (FIXED) ==================
    # Handler ini sekarang adalah method dari kelas SmartAIBot
    
    async def handle_audio(self, event, intent):
        """Menangani permintaan pembuatan audio."""
        text = intent['text']
        voice = intent['voice']
        
        status_msg = await event.respond(f"🎙️ Sedang membuat audio dengan suara **{voice}**...", parse_mode='markdown')
        
        audio_url = self.ai.generate_audio_url(text, voice)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=120.0)
                response.raise_for_status()
                audio_data = response.content
            
            await self.client.send_file(
                event.chat_id,
                file=audio_data,
                voice_note=True,
                attributes=[DocumentAttributeAudio(duration=30, title="Generated Audio", performer="SmartAI")]
            )
            await status_msg.edit(f"✅ Audio berhasil dibuat! (Suara: {voice})")
        except Exception as e:
            await status_msg.edit(f"❌ Gagal membuat audio: {str(e)}")

    async def handle_image_generation(self, event, intent):
        """Menangani permintaan pembuatan gambar."""
        model = intent['model']
        prompt = intent['prompt']
        
        status_msg = await event.respond(
            f"🎨 Sedang membuat gambar dengan model **{model}**...\n"
            f"⏳ *Prompt awal sedang diperkaya untuk hasil yang lebih baik...*",
            parse_mode='markdown'
        )
        
        enhanced_prompt = await self.ai.enhance_prompt(prompt)
        await status_msg.edit(
            f"🎨 Sedang membuat gambar dengan model **{model}**...\n"
            f"💡 Prompt: `{enhanced_prompt[:100]}...`",
            parse_mode='markdown'
        )
        
        image_url = self.ai.generate_image_url(enhanced_prompt, model)
        
        try:
            await event.respond(
                file=image_url,
                message=f"🎨 **Gambar Berhasil Dibuat**\n\n"
                       f"📝 **Prompt:** `{enhanced_prompt}`\n"
                       f"🎭 **Model:** `{model}`",
                parse_mode='markdown'
            )
            await status_msg.delete()
            
            await self.db.add_message(event.sender_id, 'user', f"Buatkan gambar: {prompt}")
            await self.db.add_message(event.sender_id, 'assistant', f"Gambar dibuat: {model}", image_url)
        except Exception as e:
            await status_msg.edit(f"❌ Gagal membuat gambar: {str(e)}")

    async def handle_image_transform(self, event, intent, image_url):
        """Menangani permintaan transformasi gambar."""
        if not image_url:
            await event.respond("❌ Gagal mendapatkan URL foto. Silakan coba lagi.")
            return
        
        prompt = intent['prompt'] or "tingkatkan kualitas gambar ini"
        model = intent['model']
        
        status_msg = await event.respond(
            f"✏️ **Memproses Transformasi Gambar...**\n"
            f"📝 **Instruksi:** `{prompt}`\n"
            f"⏳ Mohon tunggu...",
            parse_mode='markdown'
        )
        
        transform_url = self.ai.transform_image_url(prompt, image_url, model)
        
        try:
            await event.respond(
                file=transform_url,
                message=f"✨ **Gambar Berhasil Ditransformasi**\n\n"
                       f"📝 **Instruksi:** `{prompt}`",
                parse_mode='markdown'
            )
            await status_msg.delete()
            
            await self.db.add_message(event.sender_id, 'user', f"Transformasi: {prompt}", image_url)
            await self.db.add_message(event.sender_id, 'assistant', f"Hasil transformasi dengan {model}", transform_url)
        except Exception as e:
            await status_msg.edit(f"❌ Gagal transformasi gambar: {str(e)}\n\n💡 Coba dengan instruksi yang berbeda.")

    async def handle_chat(self, event, intent, message_text, image_url=None):
        """Menangani percakapan chat."""
        user_id = event.sender_id
        model = intent['model']
        
        if image_url:
            content = message_text or "Jelaskan gambar ini secara detail."
            await self.db.add_message(user_id, 'user', content, image_url)
        else:
            await self.db.add_message(user_id, 'user', message_text)
        
        history = await self.db.get_history(user_id)
        
        messages = []
        for msg in history:
            content = msg['content']
            if msg.get('image_url') and model in ['gemini', 'gemini-search', 'openai', 'openai-large']:
                # Format untuk model vision
                content = f"[Menganalisis gambar: {msg['image_url']}]\n\n{content}"
            messages.append({"role": msg['role'], "content": content})
        
        async with self.client.action(event.chat_id, 'typing'):
            emoji_map = {'gpt-5': '🤖', 'qwen-coder': '💻', 'deepseek-r1': '🧠', 'gemini-search': '🔍', 'gemini': '👁️'}
            model_emoji = emoji_map.get(model, '🤖')
            model_name = model.replace('-', ' ').upper()
            
            response = await self.ai.chat(messages, model)
            
            await self.db.add_message(user_id, 'assistant', response)
            
            await event.respond(f"{model_emoji} **{model_name}**\n\n{response}", parse_mode='markdown')

    # ================== REGISTRASI EVENT HANDLER ==================

    def register_handlers(self):
        @self.client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            user = await event.get_sender()
            await self.db.get_or_create_user(user.id, user.username, user.first_name)
            
            is_member, not_joined = await self.gatekeeper.check_membership(user.id)
            if not is_member:
                message, buttons = self.gatekeeper.get_verification_message(not_joined)
                await event.respond(message, buttons=buttons, parse_mode='markdown')
                return
            
            await self.db.set_verified(user.id, True)
            
            await event.respond(
                f"🤖 **Bot AI Cerdas Aktif**\n\n"
                f"Halo **{user.first_name}**! Saya adalah bot AI serbaguna yang siap membantu Anda.\n\n"
                f"**🎯 Kemampuan Utama:**\n"
                f"💬 Chat cerdas dengan model GPT-5\n"
                f"🎨 Membuat gambar dari teks\n"
                f"✏️ Mengubah dan mengedit gambar\n"
                f"👁️ Menganalisis isi foto\n"
                f"🎙️ Mengubah teks menjadi suara\n"
                f"💻 Membantu coding dan debugging\n"
                f"🔍 Mencari informasi terkini\n"
                f"🧠 Melakukan penalaran mendalam\n\n"
                f"**✨ Cara Penggunaan:**\n"
                f"• Kirim pesan apa saja untuk memulai chat.\n"
                f"• Awali dengan \"buatkan gambar...\" untuk membuat gambar.\n"
                f"• Kirim foto untuk dianalisis, atau kirim foto dengan instruksi (contoh: \"ubah jadi anime\").\n"
                f"• Awali dengan \"katakan...\" untuk mengubah teks menjadi suara.\n\n"
                f"**📝 Perintah Lain:**\n"
                f"`/clear` - Hapus riwayat percakapan\n"
                f"`/help` - Tampilkan bantuan lengkap\n\n"
                f"💡 Saya akan otomatis memilih model AI terbaik untuk setiap permintaan Anda!",
                parse_mode='markdown'
            )
        
        @self.client.on(events.NewMessage(pattern='/clear'))
        async def clear_handler(event):
            if not await self.check_verification(event): return
            await self.db.clear_history(event.sender_id)
            await event.respond("🗑️ Riwayat percakapan Anda telah berhasil dihapus.")
        
        @self.client.on(events.NewMessage(pattern='/help'))
        async def help_handler(event):
            if not await self.check_verification(event): return
            await event.respond(
                "📖 **Panduan Lengkap Bot AI Cerdas**\n\n"
                "**🎨 Membuat Gambar:**\n"
                "• `buatkan gambar pemandangan senja di pantai`\n"
                "• `generate gadis anime dengan rambut biru`\n"
                "• `gambar 3D robot futuristik realistis`\n\n"
                "**👁️ Analisis & Edit Foto:**\n"
                "• Kirim foto (tanpa teks) untuk dianalisis.\n"
                "• Kirim foto + `apa isi gambar ini?`\n"
                "• Kirim foto + `ubah jadi gaya anime`\n"
                "• Kirim foto + `jadikan lebih realistis`\n\n"
                "**🎙️ Teks ke Suara (TTS):**\n"
                "• `katakan halo apa kabar dengan suara nova`\n"
                "• `suarakan teks ini pakai suara echo`\n"
                "• Pilihan suara: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`\n\n"
                "**💻 Bantuan Koding:**\n"
                "• `buatkan kode python untuk sorting list`\n"
                "• `debug kode javascript ini ...`\n\n"
                "**🔍 Pencarian & Informasi:**\n"
                "• `cari berita terbaru tentang teknologi AI`\n"
                "• `informasi mengenai ...`\n\n"
                "💡 Bot secara otomatis mendeteksi keinginan Anda. Semakin detail permintaan Anda, semakin baik hasilnya. Selamat mencoba! 🚀",
                parse_mode='markdown'
            )

        @self.client.on(events.NewMessage(incoming=True, func=lambda e: e.message.text and not e.message.text.startswith('/')))
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
                    if image_url:
                        await status_msg.edit("✅ Foto berhasil diunggah! Memproses permintaan Anda...")
                    else:
                        await status_msg.edit("❌ Gagal mengunggah foto. Coba lagi dengan foto lain.")
                        return
                except Exception as e:
                    await status_msg.edit(f"❌ Terjadi error saat memproses foto: {e}")
                    return
            
            intent = await self.ai.detect_intent(message_text, has_photo)
            
            if intent['type'] == 'audio':
                await self.handle_audio(event, intent)
            elif intent['type'] == 'image':
                await self.handle_image_generation(event, intent)
            elif intent['type'] == 'image_transform':
                await self.handle_image_transform(event, intent, image_url)
            elif intent['type'] == 'chat':
                await self.handle_chat(event, intent, message_text, image_url)

    async def check_verification(self, event) -> bool:
        """Memeriksa verifikasi pengguna sebelum memproses perintah."""
        is_verified = await self.db.is_verified(event.sender_id)
        if is_verified:
            return True
        
        is_member, not_joined = await self.gatekeeper.check_membership(event.sender_id)
        if not is_member:
            message, buttons = self.gatekeeper.get_verification_message(not_joined)
            await event.respond(message, buttons=buttons, parse_mode='markdown')
            return False
        else:
            await self.db.set_verified(event.sender_id, True)
            return True

async def main():
    bot = SmartAIBot()
    try:
        await bot.start()
    except Exception as e:
        print(f"❌ Bot berhenti karena error: {e}")
    finally:
        await bot.ai.close()
        await bot.client.disconnect()
        print("🛑 Bot telah berhenti dan koneksi ditutup.")

if __name__ == '__main__':
    asyncio.run(main())
