# 📄 gatekeeper.py v2 (Middleware otorisasi)

from typing import Callable, Any
from telethon import events

class Gatekeeper:
    def __init__(self, required_channels: list, required_groups: list):
        self.required_channels = required_channels
        self.required_groups = required_groups

    def require_subscription(self, func: Callable) -> Callable:
        """Decorator untuk memastikan user tergabung di channel & grup."""
        async def wrapper(event: events.NewMessage.Event):
            try:
                # Cek subscription (di bot.py kita ambil dari client)
                # Di sini kita hanya placeholder
                # Implementasi sebenarnya di bot.py
                return await func(event)
            except Exception:
                await event.reply("⚠️ Anda harus berlangganan channel dan grup terlebih dahulu.")
        return wrapper
