"""
Smart AI Telegram Bot - Main Bot
Production-ready Telegram bot with AI chaining
"""

import os
import asyncio
import logging
from typing import List
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError, UserIsBlockedError
from telethon.tl.types import Message

from database import Database
from ai_engine import AIChainEngine
from config import (
    API_ID, API_HASH, BOT_TOKEN,
    REQUIRED_CHANNELS, REQUIRED_GROUPS,
    MAX_MESSAGE_LENGTH, DEBUG, MESSAGES,
    MAX_MESSAGES_PER_MINUTE, MAX_IMAGES_PER_HOUR,
    DATABASE_URL, LOG_LEVEL, LOG_FORMAT
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize components
db = Database(DATABASE_URL)
ai = AIChainEngine()

# Initialize bot
bot = TelegramClient(
    'bot_session',
    API_ID,
    API_HASH
).start(bot_token=BOT_TOKEN)

logger.info("🤖 Bot initialized")


# ================== UTILITIES ==================

async def check_subscription(user_id: int) -> bool:
    """Check if user is subscribed to required channels/groups"""
    try:
        # Check channels
        for channel in REQUIRED_CHANNELS:
            try:
                participant = await bot.get_permissions(channel, user_id)
                if not participant or participant.is_banned:
                    return False
            except Exception:
                return False
        
        # Check groups
        for group in REQUIRED_GROUPS:
            try:
                participant = await bot.get_permissions(group, user_id)
                if not participant or participant.is_banned:
                    return False
            except Exception:
                return False
        
        return True
    except Exception as e:
        logger.error(f"Subscription check error: {e}")
        return False


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split long messages into chunks"""
    if len(text) <= limit:
        return [text]
    
    chunks = []
    current = ""
    
    for line in text.split('\n'):
        if len(current) + len(line) + 1 <= limit:
            current += line + '\n'
        else:
            if current:
                chunks.append(current.strip())
            current = line + '\n'
    
    if current:
        chunks.append(current.strip())
    
    return chunks


async def send_long_message(event, text: str) -> None:
    """Send message, splitting if necessary"""
    chunks = split_message(text)
    for chunk in chunks:
        await event.respond(chunk)


async def verify_user(user_id: int) -> bool:
    """Verify and update user subscription status"""
    is_subscribed = await check_subscription(user_id)
    await db.update_verification(user_id, is_subscribed)
    return is_subscribed


# ================== EVENT HANDLERS ==================

@bot.on(events.NewMessage(pattern='/start'))
async def handle_start(event: events.NewMessage.Event):
    """Handle /start command"""
    user = event.sender
    
    # Register or update user
    await db.get_or_create_user(
        user.id,
        user.username,
        user.first_name,
        user.last_name,
        user.lang_code or 'id'
    )
    
    # Check subscription
    if not await verify_user(user.id):
        channels_text = '\n'.join(f"• {c}" for c in REQUIRED_CHANNELS)
        groups_text = '\n'.join(f"• {g}" for g in REQUIRED_GROUPS)
        
        message = MESSAGES["subscription_required"].format(
            channels=channels_text,
            groups=groups_text
        )
        await event.respond(message)
        return
    
    # Send welcome message
    welcome = MESSAGES["welcome"].format(bot_name="Smart AI Bot")
    await event.respond(welcome)
    
    logger.info(f"✅ User {user.id} started bot")


@bot.on(events.NewMessage(pattern='/help'))
async def handle_help(event: events.NewMessage.Event):
    """Handle /help command"""
    help_text = """📚 **Panduan Penggunaan Smart AI Bot**

🤖 **Cara Menggunakan:**
Cukup kirim pesan biasa, saya akan otomatis memahami kebutuhan Anda!

✨ **Kemampuan Saya:**

💬 **Chat Cerdas**
• Jelaskan topik kompleks
• Jawab pertanyaan
• Diskusi mendalam
Contoh: "Jelaskan quantum computing"

💻 **Programming & Code**
• Generate kode
• Debug & fix errors
• Explain code
Contoh: "Buatkan fungsi Python untuk sorting quicksort"

🎨 **Generate Gambar**
• Realistic photos
• Anime/cartoon
• 3D renders
• Artistic styles
Contoh: "Gambar pemandangan gunung saat sunset, realistis"

📊 **Analisis & Research**
• Analisis data
• Research topics
• Detailed explanations
Contoh: "Analisa perbedaan React vs Vue"

⚙️ **Perintah:**
• `/start` - Mulai bot
• `/help` - Panduan ini
• `/clear` - Hapus riwayat chat
• `/stats` - Lihat statistik penggunaan

💡 **Tips:**
- Semakin detail permintaan, semakin baik hasilnya
- Untuk gambar, sebutkan style yang diinginkan
- Untuk kode, sebutkan bahasa pemrograman

🚀 Mulai chat sekarang!"""
    
    await event.respond(help_text)


@bot.on(events.NewMessage(pattern='/clear'))
async def handle_clear(event: events.NewMessage.Event):
    """Handle /clear command"""
    user_id = event.sender_id
    
    deleted = await db.clear_history(user_id)
    await event.respond(f"🗑️ Berhasil menghapus {deleted} pesan dari riwayat chat Anda.")
    
    logger.info(f"🗑️ User {user_id} cleared history")


@bot.on(events.NewMessage(pattern='/stats'))
async def handle_stats(event: events.NewMessage.Event):
    """Handle /stats command"""
    user_id = event.sender_id
    
    stats = await db.get_user_stats(user_id)
    
    stats_text = f"""📊 **Statistik Penggunaan Anda**

💬 Total Pesan: {stats['total_messages']}
🎨 Total Gambar: {stats['total_images']}
💻 Total Kode: {stats['total_code_requests']}
🔢 Total Token: {stats['total_tokens']:,}

⏰ Terakhir Reset: {stats.get('last_reset', 'N/A')}

Terus gunakan bot untuk meningkatkan statistik Anda! 🚀"""
    
    await event.respond(stats_text)


@bot.on(events.NewMessage(func=lambda e: not e.text.startswith('/')))
async def handle_message(event: Message):
    """Handle regular messages with AI chain"""
    user = event.sender
    user_id = user.id
    text = event.text
    has_photo = event.photo is not None
    
    # Verify subscription
    if not await verify_user(user_id):
        channels_text = '\n'.join(f"• {c}" for c in REQUIRED_CHANNELS)
        groups_text = '\n'.join(f"• {g}" for g in REQUIRED_GROUPS)
        
        message = MESSAGES["subscription_required"].format(
            channels=channels_text,
            groups=groups_text
        )
        await event.respond(message)
        return
    
    # Check rate limit
    if not await db.check_rate_limit(user_id, 'all', MAX_MESSAGES_PER_MINUTE, 1):
        await event.respond("⚠️ Terlalu banyak permintaan. Tunggu sebentar ya!")
        return
    
    # Empty message check
    if not text or len(text.strip()) == 0:
        await event.respond("❓ Silakan kirim pesan yang valid.")
        return
    
    try:
        # Send processing indicator
        status_msg = await event.respond(MESSAGES["processing"])
        
        # Get chat history
        history = await db.get_chat_history(user_id)
        
        # Save user message
        await db.save_message(user_id, 'user', text)
        
        # Process with AI chain
        result = await ai.process_query(text, history, has_photo)
        
        # Delete status message
        await status_msg.delete()
        
        if not result['success']:
            await event.respond(MESSAGES["error"].format(error=result['content']))
            return
        
        # Handle different result types
        if result['type'] == 'image':
            # Check image rate limit
            if not await db.check_rate_limit(user_id, 'image', MAX_IMAGES_PER_HOUR, 60):
                await event.respond("⚠️ Batas generate gambar tercapai. Coba lagi nanti.")
                return
            
            await event.respond("🎨 Gambar Anda:")
            await event.respond(file=result['content'])
            
            # Save to history
            await db.save_message(
                user_id, 'assistant', 'Generated image',
                intent_type='image',
                image_url=result['content']
            )
        
        else:
            # Text response
            await send_long_message(event, result['content'])
            
            # Save to history
            intent = result.get('intent', {})
            await db.save_message(
                user_id, 'assistant', result['content'],
                intent_type=intent.get('type'),
                model_used=intent.get('model')
            )
        
        logger.info(f"✅ Processed message from user {user_id}: {result['type']}")
    
    except FloodWaitError as e:
        await event.respond(f"⚠️ Rate limit exceeded. Wait {e.seconds} seconds.")
        logger.warning(f"FloodWait: {e.seconds}s for user {user_id}")
    
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=DEBUG)
        await event.respond(MESSAGES["error"].format(error="Terjadi kesalahan. Coba lagi."))


@bot.on(events.NewMessage(pattern='/admin'))
async def handle_admin(event: events.NewMessage.Event):
    """Admin commands (restricted)"""
    user_id = event.sender_id
    
    # Simple admin check (you can enhance this)
    if user_id not in [123456789]:  # Replace with actual admin IDs
        await event.respond("⛔ Unauthorized")
        return
    
    args = event.text.split()[1:] if len(event.text.split()) > 1 else []
    
    if not args:
        admin_help = """👨‍💼 **Admin Commands**

`/admin stats` - Get bot statistics
`/admin users` - List active users
`/admin health` - Check system health"""
        
        await event.respond(admin_help)
        return
    
    command = args[0]
    
    if command == 'health':
        db_health = await db.health_check()
        status = "✅ Healthy" if db_health else "❌ Unhealthy"
        await event.respond(f"🏥 Database: {status}")
    
    elif command == 'users':
        users = await db.get_active_users(7)
        text = f"👥 Active Users (7 days): {len(users)}\n\n"
        for user in users[:10]:  # Top 10
            text += f"• {user['first_name']} (@{user['username']}) - {user['message_count']} msgs\n"
        await event.respond(text)
    
    else:
        await event.respond("❓ Unknown admin command")


# ================== MAIN ==================

async def main():
    """Main entry point"""
    try:
        # Connect database
        await db.connect()
        logger.info("✅ Database connected")
        
        # Bot info
        me = await bot.get_me()
        logger.info(f"🤖 Bot started: @{me.username}")
        
        # Keep running
        await bot.run_until_disconnected()
    
    except KeyboardInterrupt:
        logger.info("⚠️ Bot stopped by user")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        await ai.close()
        await db.close()
        logger.info("👋 Bot shutdown complete")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
