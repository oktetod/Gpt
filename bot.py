"""
Smart AI Telegram Bot v3.1 - PRODUCTION READY
Fixed startup and error handling
"""

import asyncio
import logging
import sys
import signal
from datetime import datetime
from typing import Optional

from telethon import TelegramClient, events, Button
from telethon.tl.types import DocumentAttributeFilename
import anthropic

from config import (
    API_ID, API_HASH, BOT_TOKEN, CLAUDE_API_KEY,
    DATABASE_URL, MAX_HISTORY, MAX_MESSAGES_PER_MINUTE,
    ADMIN_USER_ID, ALLOWED_USERS
)
from database import Database
from intent_detector import IntentDetector
from ai_handler import AIHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global instances
client: Optional[TelegramClient] = None
db: Optional[Database] = None
intent_detector: Optional[IntentDetector] = None
ai_handler: Optional[AIHandler] = None
shutdown_event = asyncio.Event()


async def initialize_services():
    """Initialize all services with proper error handling"""
    global db, intent_detector, ai_handler
    
    try:
        # 1. Initialize Database
        logger.info("🔄 Initializing database...")
        db = Database(DATABASE_URL)
        await db.connect()
        logger.info("✅ Database connected")
        
        # 2. Initialize Claude Client
        logger.info("🔄 Initializing Claude AI...")
        claude_client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)
        logger.info("✅ Claude AI initialized")
        
        # 3. Initialize Intent Detector
        logger.info("🔄 Initializing Intent Detector...")
        intent_detector = IntentDetector()
        logger.info("✅ Intent Detector ready")
        
        # 4. Initialize AI Handler
        logger.info("🔄 Initializing AI Handler...")
        ai_handler = AIHandler(claude_client, db)
        logger.info("✅ AI Handler ready")
        
        logger.info("🎉 All services initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}", exc_info=True)
        return False


def check_access(user_id: int) -> bool:
    """Check if user has access to bot"""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS or user_id == ADMIN_USER_ID


async def send_message_safe(event, text: str, **kwargs):
    """Send message with error handling"""
    try:
        await event.respond(text, **kwargs)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        try:
            # Fallback to simple text
            await event.respond("⚠️ Terjadi kesalahan saat mengirim respons.")
        except:
            pass


@events.register(events.NewMessage(pattern='/start'))
async def start_handler(event):
    """Handle /start command"""
    try:
        user_id = event.sender_id
        
        if not check_access(user_id):
            await send_message_safe(
                event,
                "⛔ Maaf, Anda tidak memiliki akses ke bot ini.\n"
                "Hubungi admin untuk mendapatkan akses."
            )
            return
        
        # Get or create user
        sender = await event.get_sender()
        await db.get_or_create_user(
            user_id=user_id,
            username=sender.username,
            first_name=sender.first_name,
            last_name=sender.last_name
        )
        
        welcome_text = f"""
🤖 **Selamat Datang di Smart AI Bot!**

Halo {sender.first_name}! 👋

Saya adalah asisten AI yang dapat membantu Anda dengan:
✨ Percakapan umum dan tanya jawab
🖼️ Analisis gambar (kirim foto dengan caption)
💻 Bantuan coding dan debugging
🌐 Pencarian informasi web

**Perintah yang tersedia:**
/start - Memulai bot
/clear - Hapus riwayat chat
/stats - Lihat statistik penggunaan
/help - Panduan lengkap

Kirim pesan untuk memulai! 💬
"""
        
        await send_message_safe(event, welcome_text, parse_mode='md')
        logger.info(f"✅ User {user_id} started the bot")
        
    except Exception as e:
        logger.error(f"Error in start handler: {e}", exc_info=True)
        await send_message_safe(event, "❌ Terjadi kesalahan. Silakan coba lagi.")


@events.register(events.NewMessage(pattern='/clear'))
async def clear_handler(event):
    """Handle /clear command"""
    try:
        user_id = event.sender_id
        
        if not check_access(user_id):
            return
        
        deleted = await db.clear_history(user_id)
        await send_message_safe(
            event,
            f"✅ Riwayat chat telah dihapus!\n"
            f"📊 {deleted} pesan dihapus dari database."
        )
        logger.info(f"✅ User {user_id} cleared chat history")
        
    except Exception as e:
        logger.error(f"Error in clear handler: {e}", exc_info=True)
        await send_message_safe(event, "❌ Gagal menghapus riwayat.")


@events.register(events.NewMessage(pattern='/stats'))
async def stats_handler(event):
    """Handle /stats command"""
    try:
        user_id = event.sender_id
        
        if not check_access(user_id):
            return
        
        stats = await db.get_user_stats(user_id)
        
        stats_text = f"""
📊 **Statistik Penggunaan Anda**

💬 Total Pesan: {stats['total_messages']}
🖼️ Analisis Gambar: {stats['total_images']}
💻 Request Coding: {stats['total_code_requests']}
🌐 Web Search: {stats['total_web_searches']}
🔢 Total Token: {stats['total_tokens']:,}

⏰ Terakhir Reset: {stats.get('last_reset', 'N/A')}
"""
        
        await send_message_safe(event, stats_text, parse_mode='md')
        
    except Exception as e:
        logger.error(f"Error in stats handler: {e}", exc_info=True)
        await send_message_safe(event, "❌ Gagal mengambil statistik.")


@events.register(events.NewMessage(pattern='/help'))
async def help_handler(event):
    """Handle /help command"""
    try:
        if not check_access(event.sender_id):
            return
        
        help_text = """
📚 **Panduan Penggunaan Smart AI Bot**

**Perintah Dasar:**
/start - Memulai bot
/clear - Hapus riwayat percakapan
/stats - Lihat statistik penggunaan
/help - Tampilkan panduan ini

**Cara Menggunakan:**

1️⃣ **Chat Biasa**
Kirim pesan teks untuk bertanya atau berdiskusi

2️⃣ **Analisis Gambar**
Kirim foto dengan caption untuk analisis gambar

3️⃣ **Coding Help**
Tanya tentang programming, debugging, atau minta contoh kode

4️⃣ **Web Search**
Bot otomatis mencari info terkini jika diperlukan

**Tips:**
• Semakin jelas pertanyaan, semakin akurat jawabannya
• Bot mengingat konteks percakapan sebelumnya
• Gunakan /clear untuk memulai topik baru

Selamat menggunakan! 🚀
"""
        
        await send_message_safe(event, help_text, parse_mode='md')
        
    except Exception as e:
        logger.error(f"Error in help handler: {e}", exc_info=True)


@events.register(events.NewMessage)
async def handle_message(event):
    """Main message handler"""
    # Skip if it's a command
    if event.message.text and event.message.text.startswith('/'):
        return
    
    user_id = event.sender_id
    
    try:
        # Check access
        if not check_access(user_id):
            await send_message_safe(
                event,
                "⛔ Anda tidak memiliki akses ke bot ini."
            )
            return
        
        # Rate limiting
        if not await db.check_rate_limit(user_id, 'all', MAX_MESSAGES_PER_MINUTE, 1):
            await send_message_safe(
                event,
                f"⏳ Terlalu banyak pesan! Maksimal {MAX_MESSAGES_PER_MINUTE} pesan per menit.\n"
                "Tunggu sebentar ya..."
            )
            return
        
        # Get or create user
        sender = await event.get_sender()
        await db.get_or_create_user(
            user_id=user_id,
            username=sender.username,
            first_name=sender.first_name,
            last_name=sender.last_name
        )
        
        # Show typing indicator
        async with client.action(event.chat_id, 'typing'):
            # Process message
            message_text = event.message.text or ""
            image_data = None
            
            # Check for image
            if event.message.photo:
                try:
                    image_bytes = await event.message.download_media(bytes)
                    import base64
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    await send_message_safe(event, "❌ Gagal memproses gambar.")
                    return
            
            # Detect intent
            intent = intent_detector.detect_intent(message_text, has_image=bool(image_data))
            logger.info(f"User {user_id} - Intent: {intent}")
            
            # Get chat history
            history = await db.get_chat_history(user_id, MAX_HISTORY)
            
            # Generate AI response
            response = await ai_handler.generate_response(
                user_id=user_id,
                message=message_text,
                intent=intent,
                history=history,
                image_data=image_data
            )
            
            # Send response
            if len(response) > 4000:
                # Split long messages
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await send_message_safe(event, chunk, parse_mode='md')
                    await asyncio.sleep(0.5)
            else:
                await send_message_safe(event, response, parse_mode='md')
            
            logger.info(f"✅ Response sent to user {user_id}")
    
    except Exception as e:
        logger.error(f"Error handling message from {user_id}: {e}", exc_info=True)
        await send_message_safe(
            event,
            "❌ Maaf, terjadi kesalahan saat memproses pesan Anda.\n"
            "Silakan coba lagi dalam beberapa saat."
        )


async def shutdown_handler(sig=None):
    """Handle graceful shutdown"""
    if sig:
        logger.info(f"🛑 Received signal {sig.name}, shutting down...")
    else:
        logger.info("🛑 Shutting down...")
    
    shutdown_event.set()
    
    # Cleanup
    if db:
        await db.close()
    
    if client:
        await client.disconnect()
    
    logger.info("👋 Shutdown complete")


async def main():
    """Main bot function"""
    global client
    
    try:
        logger.info("🚀 Starting Smart AI Bot...")
        
        # Initialize services
        if not await initialize_services():
            logger.error("❌ Failed to initialize services")
            sys.exit(1)
        
        # Create Telegram client
        logger.info("🔄 Connecting to Telegram...")
        client = TelegramClient('bot_session', API_ID, API_HASH)
        
        # Register event handlers
        client.add_event_handler(start_handler)
        client.add_event_handler(clear_handler)
        client.add_event_handler(stats_handler)
        client.add_event_handler(help_handler)
        client.add_event_handler(handle_message)
        
        # Start the bot
        await client.start(bot_token=BOT_TOKEN)
        
        bot_info = await client.get_me()
        logger.info(f"✅ Bot started successfully!")
        logger.info(f"👤 Bot username: @{bot_info.username}")
        logger.info(f"🆔 Bot ID: {bot_info.id}")
        logger.info(f"🌐 Listening for messages...")
        
        # Keep running until shutdown signal
        await shutdown_event.wait()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await shutdown_handler()


if __name__ == '__main__':
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown_handler(s))
        )
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("⚠️ Interrupted by user")
    finally:
        loop.close()
        logger.info("🔒 Event loop closed")
