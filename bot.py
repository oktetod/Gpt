"""
Updated message handler for bot.py
Replace the handle_message function with this
"""

@bot.on(events.NewMessage(func=lambda e: not e.text.startswith('/')))
async def handle_message(event: Message):
    """Handle regular messages with multi-provider AI"""
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
        if has_photo and any(kw in text.lower() for kw in INTENT_KEYWORDS.get("ocr", [])):
            status_msg = await event.respond(MESSAGES["ocr_processing"])
        elif any(kw in text.lower() for kw in INTENT_KEYWORDS.get("code", [])):
            status_msg = await event.respond(MESSAGES["coding"])
        elif any(kw in text.lower() for kw in ["analisa", "thinking", "deep"]):
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
            except Exception as e:
                logger.error(f"Failed to download photo: {e}")
        
        # Process with AI
        result = await ai.process_query(
            text or "What's in this image?",
            history,
            has_photo,
            photo_data
        )
        
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
            
            await db.save_message(
                user_id, 'assistant', 'Generated image',
                intent_type='image',
                image_url=result['content']
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
            # Text response
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