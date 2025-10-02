# 📄 bot.py v2 (versi lengkap dan benar)

import os
import asyncio
import traceback
from typing import Optional
from telethon import TelegramClient, events, Button
from telethon.errors import MessageTooLongError
from telethon.tl.types import Message

# Impor eksternal
from database import Database
from ai_engine import PollinationsAI
from config import BOT_TOKEN, REQUIRED_CHANNELS, REQUIRED_GROUPS, MAX_MESSAGE_LENGTH, DEBUG, ADMIN_IDS

# Inisialisasi komponen
db = Database(os.getenv("DATABASE_URL"))
ai = PollinationsAI()

# Inisialisasi bot
bot = TelegramClient('bot', int(os.getenv("TELEGRAM_API_ID")), os.getenv("TELEGRAM_API_HASH")).start(bot_token=BOT_TOKEN)

# Cek subscription
async def check_subscription(user_id: int) -> bool:
    try:
        for channel in REQUIRED_CHANNELS:
            await bot.get_permissions(channel, user_id)
        for group in REQUIRED_GROUPS:
            await bot.get_permissions(group, user_id)
        return True
    except Exception:
        return False

# Helper: Potong pesan panjang
def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> List[str]:
    return [text[i:i+limit] for i in range(0, len(text), limit)]

# Handler: /start
@bot.on(events.NewMessage(pattern='/start'))
async def start(event: events.NewMessage.Event):
    user_id = event.sender_id
    username = event.sender.username
    first_name = event.sender.first_name

    await db.get_or_create_user(user_id, username, first_name)

    welcome = (
        "👋 Halo! Aku adalah Smart AI Bot.\n"
        "Kirim pesan apa saja, dan aku akan bantu!\n\n"
        "Contoh:\n"
        "• /image lukis seekor kucing di luar angkasa\n"
        "• /chat Jelaskan teori relativitas\n"
        "• /tts Suarakan teks ini"
    )
    await event.reply(welcome)

# Handler: Pesan teks biasa
@bot.on(events.NewMessage(func=lambda e: not e.text.startswith('/')))
async def handle_message(event: Message):
    user_id = event.sender_id
    text = event.text
    has_photo = event.photo is not None

    if not await check_subscription(user_id):
        await event.reply("⚠️ Anda harus bergabung dengan channel dan grup berikut:\n" + "\n".join(REQUIRED_CHANNELS + REQUIRED_GROUPS))
        return

    intent = await ai.detect_intent(text, has_photo)

    try:
        if intent['type'] == 'audio':
            await event.reply(f"🎧 Audio akan dibuat dengan suara: {intent['voice']}")
        elif intent['type'] == 'image':
            img_url = await ai.generate_image(intent['prompt'], intent['model'])
            await event.reply("🖼️ Hasil gambar:", file=img_url)
        elif intent['type'] == 'image_transform':
            await event.reply("🖼️ Transformasi gambar tidak didukung saat ini.")
        elif intent['type'] == 'chat':
            response = await ai.generate_text(text, intent['model'])
            for part in split_message(response):
                await event.reply(part)
    except Exception as e:
        await event.reply(f"❌ Terjadi kesalahan: {str(e)}")
        if DEBUG:
            print(traceback.format_exc())

# Handler: /image
@bot.on(events.NewMessage(pattern='/image'))
async def image_command(event: events.NewMessage.Event):
    prompt = event.text[len('/image'):].strip()
    if not prompt:
        await event.reply("📌 Gunakan: /image [prompt]")
        return

    img_url = await ai.generate_image(prompt)
    await event.reply("🖼️ Hasil gambar:", file=img_url)

# Handler: /clear
@bot.on(events.NewMessage(pattern='/clear'))
async def clear_history(event: events.NewMessage.Event):
    user_id = event.sender_id
    await db.clear_history(user_id)
    await event.reply("🗑️ Riwayat obrolan telah dihapus.")

# Handler: /help
@bot.on(events.NewMessage(pattern='/help'))
async def help_command(event: events.NewMessage.Event):
    help_text = (
        "📌 Bantuan Smart AI Bot:\n\n"
        "• Kirim pesan biasa → AI akan merespons otomatis\n"
        "• /image [prompt] → Buat gambar\n"
        "• /clear → Hapus riwayat\n"
        "• /help → Tampilkan bantuan ini"
    )
    await event.reply(help_text)

# Main loop
async def main():
    await db.connect()
    print("🚀 Bot siap digunakan!")
    await bot.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())
