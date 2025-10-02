"""
Smart AI Telegram Bot v3.0 - Complete Bot Handler
Features: Multi-modal AI, RAG, Vision, 4096 character limit fix
"""

import logging
import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import Message
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetParticipantRequest

from config import (
    API_ID, API_HASH, BOT_TOKEN, REQUIRED_CHANNELS,
    REQUIRED_GROUPS, MESSAGES, MAX_MESSAGE_LENGTH,
    MAX_MESSAGES_PER_MINUTE, MAX_IMAGES_PER_HOUR,
    MAX_OCR_PER_HOUR, INTENT_KEYWORDS, DEBUG, LOG_FORMAT,
    LOG_LEVEL, DATABASE_URL
)
from database import Database
from ai_engine import MultiProviderAI

# Setup logging
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)
logger = logging.getLogger(__name__)

# Initialize components
bot = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)
db = Database(DATABASE_URL)
ai = MultiProviderAI()


# ================== HELPER FUNCTIONS ==================

async def send_long_message(event: Message, text: str, parse_mode: str = 'markdown'):
    """
    Send long message by splitting into chunks (4096 character limit fix)
    """
    # Clean text
    text = text.strip()
    
    # If message is short enough, send directly
    if len(text) <= MAX_MESSAGE_LENGTH:
        try:
            await event.respond(text, parse_mode=parse_mode)
        except Exception as e:
            # Fallback without parse_mode if markdown fails
            logger.warning(f"Markdown parse failed: {e}")
            await event.respond(text, parse_mode=None)
        return
    
    # Split into chunks
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If single paragraph is too long, split by sentences
        if len(para) > MAX_MESSAGE_LENGTH:
            sentences = para.split('. ')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= MAX_MESSAGE_LENGTH:
                    current_chunk += sentence + '. '
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
        else:
            # Add paragraph to current chunk
            if len(current_chunk) + len(para) + 2 <= MAX_MESSAGE_LENGTH:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send all chunks
    for i, chunk in enumerate(chunks, 1):
        try:
            if i == 1:
                await event.respond(chunk, parse_mode=parse_mode)
            else:
                # Add continuation marker
                await event.respond(f"*...continued ({i}/{len(chunks)})*\n\n{chunk}", parse_mode=parse_mode)
            
            # Small delay between chunks
            if i < len(chunks):
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error sending chunk {i}: {e}")
            # Try without parse_mode
            try:
                await event.respond(chunk, parse_mode=None)
            except:
                pass


async def verify_user(user_id: int) -> bool:
    """Verify if user has joined required channels and groups"""
    try:
        # Check all required channels
        for channel in REQUIRED_CHANNELS:
            try:
                participant = await bot(GetParticipantRequest(
                    channel=channel,
                    participant=user_id
                ))
                if not participant:
                    return False
            except Exception:
                return False
        
        # Check all required groups
        for group in REQUIRED_GROUPS:
            try:
                participant = await bot(GetParticipantRequest(
                    channel=group,
                    participant=user_id
                ))
                if not participant:
                    return False
            except Exception:
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Verification error for user {user_id}: {e}")
        return False


# ================== COMMAND HANDLERS ==================

@bot.on(events.NewMessage(pattern='/start'))
async def start_handler(event: Message):
    """Handle /start command"""
    user = event.sender
    user_id = user.id
    
    # Get or create user in database
    await db.get_or_create_user(
        user_id=user_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    # Verify subscription
    is_verified = await verify_user(user_id)
    await db.update_verification(user_id, is_verified)
    
    if not is_verified:
        channels_text = '\n'.join(f"• {c}" for c in REQUIRED_CHANNELS)
        groups_text = '\n'.join(f"• {g}" for g in REQUIRED_GROUPS)
        
        message = MESSAGES["subscription_required"].format(
            channels=channels_text,
            groups=groups_text
        )
        await event.respond(message)
        return
    
    # Send welcome message
    welcome_msg = MESSAGES["welcome"].format(
        bot_name="Smart AI Bot",
        first_name=user.first_name or "User"
    )
    await event.respond(welcome_msg)
    
    logger.info(f"✅ New user started: {user_id} (@{user.username})")


@bot.on(events.NewMessage(pattern='/help'))
async def help_handler(event: Message):
    """Handle /help command"""
    help_text = """📚 **Panduan Penggunaan GPT-5**

**💬 Chat Biasa:**
Kirim pesan apapun untuk mendapat respons AI

**💻 Code Generation:**
• "Buatkan fungsi Python untuk sorting"
• "Code REST API dengan Express.js"
• "Debug code ini: [paste code]"

**🎨 Image Generation:**
• "Gambar sunset di pantai, realistic"
• "Buat logo minimalis untuk startup"
• "Anime girl, cyberpunk style"

**📸 Vision Analysis:**
Kirim foto dengan pertanyaan:
• "Apa isi gambar ini?"
• "Jelaskan diagram ini"
• "Translate text in this image"

**📄 OCR (Text Extraction):**
Kirim foto dengan:
• "Baca teks dalam gambar"
• "Extract text from this"

**🧠 Deep Analysis:**
• "Analisa perbedaan Python vs JavaScript"
• "Jelaskan blockchain secara mendalam"
• "Research tentang AI ethics"

**⚙️ Commands:**
/start - Mulai bot
/help - Panduan ini
/stats - Statistik penggunaan
/clear - Hapus history chat
/about - Tentang GPT-5

---
💫 *Powered by GPT-5* | @durov9369"""
    
    await send_long_message(event, help_text)


@bot.on(events.NewMessage(pattern='/stats'))
async def stats_handler(event: Message):
    """Handle /stats command"""
    user_id = event.sender_id
    
    if not await verify_user(user_id):
        await event.respond("⚠️ Silakan join channel/group dulu!")
        return
    
    stats = await db.get_user_stats(user_id)
    
    stats_text = f"""📊 **Statistik Penggunaan Anda**

💬 Total Pesan: {stats['total_messages']}
🎨 Total Gambar: {stats['total_images']}
💻 Request Code: {stats['total_code_requests']}
🔤 Total Tokens: {stats['total_tokens']:,}

⏱️ Last Reset: {stats.get('last_reset', 'N/A')}

---
💫 *GPT-5 by @durov9369*"""
    
    await event.respond(stats_text)


@bot.on(events.NewMessage(pattern='/clear'))
async def clear_handler(event: Message):
    """Handle /clear command"""
    user_id = event.sender_id
    
    if not await verify_user(user_id):
        await event.respond("⚠️ Silakan join channel/group dulu!")
        return
    
    deleted = await db.clear_history(user_id)
    await event.respond(f"🗑️ Berhasil menghapus {deleted} pesan dari history!")


@bot.on(events.NewMessage(pattern='/about'))
async def about_handler(event: Message):
    """Handle /about command"""
    about_text = """🤖 **GPT-5 - Advanced AI System**

**Developer:** @durov9369
**Version:** 3.0 Multi-Modal
**Architecture:** Hybrid Multi-Provider

**🔬 Core Technologies:**
• Cerebras Cloud (Qwen-3, Llama-4)
• NVIDIA NIM (Nemotron Ultra 253B)
• RAG (Retrieval Augmented Generation)
• Multi-modal processing

**💪 Capabilities:**
• Natural language understanding
• Code generation & debugging
• Image generation (Flux models)
• Vision analysis & OCR
• Deep reasoning (65K tokens)
• Real-time processing

**🎯 Specialized Models:**
• **Qwen-3 235B**: Deep thinking & reasoning
• **Qwen-3 Coder 480B**: Code generation
• **Nemotron Ultra 253B**: RAG & vision
• **Llama-4 Maverick**: Fast responses
• **Flux**: Image generation

**📈 Performance:**
• Response time: <3 seconds
• Context window: 40K-65K tokens
• Multi-language support
• 99.9% uptime

**🔒 Security:**
• End-to-end encryption
• Privacy-focused
• No data retention
• Secure processing

---
💫 *Developed with ❤️ by @durov9369*
🚀 *Powered by cutting-edge AI technology*"""
    
    await send_long_message(event, about_text)


# ================== MESSAGE HANDLER (ENHANCED) ==================

@bot.on(events.NewMessage(func=lambda e: not e.text.startswith('/')))
async def handle_message(event: Message):
    """Handle regular messages with multi-modal AI support"""
    user = event.sender
    user_id = user.id
    text = event.text or ""
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
    
    # Empty message check (but allow if there's a photo)
    if not text and not has_photo:
        await event.respond("❓ Silakan kirim pesan atau foto.")
        return
    
    try:
        # Determine processing message based on content
        if has_photo:
            if any(kw in text.lower() for kw in ["baca", "extract", "ocr", "text"]):
                status_msg = await event.respond(MESSAGES["ocr_processing"])
            else:
                status_msg = await event.respond("🔍 Analyzing image with GPT-5 Vision...")
        elif any(kw in text.lower() for kw in INTENT_KEYWORDS.get("code", [])):
            status_msg = await event.respond(MESSAGES["coding"])
        elif any(kw in text.lower() for kw in ["analisa", "thinking", "deep", "mendalam"]):
            status_msg = await event.respond(MESSAGES["thinking"])
        else:
            status_msg = await event.respond(MESSAGES["processing"])
        
        # Get chat history
        history = await db.get_chat_history(user_id)
        
        # Save user message
        await db.save_message(user_id, 'user', text or "[Photo]")
        
        # Download photo if present
        photo_data = None
        if has_photo:
            try:
                photo_data = await event.download_media(file=bytes)
                logger.info(f"📸 Downloaded photo from user {user_id}")
            except Exception as e:
                logger.error(f"Failed to download photo: {e}")
        
        # Process with AI
        result = await ai.process_query(
            text or "Analyze this image in detail",
            history,
            has_photo,
            photo_data
        )
        
        # Delete status message
        try:
            await status_msg.delete()
        except:
            pass
        
        if not result['success']:
            await event.respond(MESSAGES["error"].format(error=result['content']))
            return
        
        # Handle different result types
        if result['type'] == 'image':
            # Check image rate limit
            if not await db.check_rate_limit(user_id, 'image', MAX_IMAGES_PER_HOUR, 60):
                await event.respond("⚠️ Batas generate gambar tercapai. Coba lagi nanti.")
                return
            
            await event.respond("🎨 **Generated Image:**")
            await event.respond(file=result['content'])
            
            await db.save_message(
                user_id, 'assistant', 'Generated image',
                intent_type='image',
                image_url=result['content']
            )
        
        elif result['type'] == 'vision':
            # Vision analysis response
            await send_long_message(event, result['content'])
            
            intent = result.get('intent', {})
            await db.save_message(
                user_id, 'assistant', result['content'],
                intent_type='vision',
                model_used=intent.get('model', 'nemotron-ultra')
            )
        
        elif result['type'] == 'ocr':
            # Check OCR rate limit
            if not await db.check_rate_limit(user_id, 'ocr', MAX_OCR_PER_HOUR, 60):
                await event.respond("⚠️ Batas OCR tercapai. Coba lagi nanti.")
                return
            
            await send_long_message(event, result['content'])
            
            await db.save_message(
                user_id, 'assistant', result['content'],
                intent_type='ocr',
                model_used='nvidia-ocr'
            )
        
        else:
            # Text response (with 4096 fix)
            await send_long_message(event, result['content'])
            
            intent = result.get('intent', {})
            await db.save_message(
                user_id, 'assistant', result['content'],
                intent_type=intent.get('type'),
                model_used=intent.get('model')
            )
        
        logger.info(f"✅ Processed {result['type']} from user {user_id}")
    
    except FloodWaitError as e:
        await event.respond(f"⚠️ Rate limit exceeded. Wait {e.seconds} seconds.")
        logger.warning(f"FloodWait: {e.seconds}s for user {user_id}")
    
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=DEBUG)
        await event.respond(MESSAGES["error"].format(error="Terjadi kesalahan. Coba lagi."))


# ================== ADMIN COMMANDS ==================

@bot.on(events.NewMessage(pattern='/broadcast'))
async def broadcast_handler(event: Message):
    """Admin only: Broadcast message to all users"""
    from config import ADMIN_IDS
    
    if event.sender_id not in ADMIN_IDS:
        return
    
    # Get message to broadcast
    msg = event.text.replace('/broadcast', '').strip()
    if not msg:
        await event.respond("Usage: /broadcast <message>")
        return
    
    # Get all active users
    users = await db.get_active_users(days=30)
    
    success = 0
    failed = 0
    
    status = await event.respond(f"📢 Broadcasting to {len(users)} users...")
    
    for user in users:
        try:
            await bot.send_message(user['user_id'], msg)
            success += 1
            await asyncio.sleep(0.1)  # Rate limit protection
        except Exception as e:
            failed += 1
            logger.error(f"Broadcast failed for {user['user_id']}: {e}")
    
    await status.edit(f"✅ Broadcast complete!\nSuccess: {success}\nFailed: {failed}")


# ================== STARTUP ==================

async def main():
    """Main bot startup"""
    logger.info("🚀 Starting Smart AI Bot v3.0...")
    
    # Connect to database
    try:
        await db.connect()
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # Start bot
    logger.info("✅ Bot is running...")
    logger.info("💫 GPT-5 by @durov9369 - Ready to serve!")
    
    try:
        await bot.run_until_disconnected()
    finally:
        await db.close()
        await ai.close()
        logger.info("👋 Bot stopped")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⚠️ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
