# /proyek_bot/gatekeeper.py

from telethon import TelegramClient, Button
from telethon.errors.rpcerrorlist import UserNotParticipantError
from config import REQUIRED_CHANNELS, REQUIRED_GROUPS

class Gatekeeper:
    def __init__(self, client: TelegramClient):
        self.client = client
    
    async def check_membership(self, user_id: int) -> tuple[bool, list]:
        """Memeriksa apakah pengguna telah bergabung dengan semua channel & grup."""
        not_joined = []
        all_required = REQUIRED_CHANNELS + REQUIRED_GROUPS
        
        for entity_username in all_required:
            try:
                # Cek keanggotaan menggunakan cara yang lebih andal
                await self.client.get_permissions(entity_username, user_id)
            except UserNotParticipantError:
                not_joined.append(entity_username)
            except Exception:
                # Menangani kasus di mana bot tidak memiliki akses atau channel/grup tidak ada
                not_joined.append(entity_username)
        
        return len(not_joined) == 0, not_joined
    
    def get_verification_message(self, not_joined: list) -> tuple[str, list]:
        """Membuat pesan verifikasi beserta tombol join."""
        message = "🔐 **Verifikasi Dibutuhkan**\n\n"
        message += "Untuk menggunakan bot, Anda wajib bergabung dengan channel & grup berikut:\n\n"
        
        buttons = []
        for entity in not_joined:
            entity_clean = entity.replace('@', '')
            if entity in REQUIRED_CHANNELS:
                label = f"📢 Channel {entity}"
            else:
                label = f"👥 Grup {entity}"
            
            message += f"• {label}\n"
            buttons.append([Button.url(label, f"https://t.me/{entity_clean}")])
        
        message += "\n✅ Setelah bergabung, silakan kirim /start lagi untuk verifikasi ulang."
        return message, buttons
