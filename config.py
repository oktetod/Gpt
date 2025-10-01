import asyncpg
from typing import List, Dict
from config import MAX_HISTORY


class Database:
    def __init__(self, url: str):
        self.url = url
        self.pool = None

    async def connect(self):
        """Membuat koneksi pool ke database."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    self.url,
                    min_size=1,
                    max_size=10,
                    command_timeout=60
                )
                print("✅ Koneksi ke database berhasil.")
            except Exception as e:
                print(f"❌ Gagal menghubungkan ke database: {e}")
                raise

    async def close(self):
        """Menutup koneksi ke database."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def get_or_create_user(self, user_id: int, username: str, first_name: str, last_name: str = None):
        """Mendapatkan atau membuat pengguna baru di database."""
        async with self.pool.acquire() as conn:
            # Cek apakah pengguna sudah ada
            query = """
                SELECT id, username, first_name, last_name, is_verified, created_at, updated_at
                FROM users WHERE user_id = $1
            """
            row = await conn.fetchrow(query, user_id)
            
            if row:
                return dict(row)
            
            # Jika tidak ada, buat pengguna baru
            insert_query = """
                INSERT INTO users (user_id, username, first_name, last_name, is_verified)
                VALUES ($1, $2, $3, $4, false)
                RETURNING id, user_id, username, first_name, last_name, is_verified, created_at, updated_at
            """
            new_user = await conn.fetchrow(
                insert_query,
                user_id,
                username,
                first_name,
                last_name
            )
            return dict(new_user)

    async def update_user_verification(self, user_id: int, is_verified: bool):
        """Memperbarui status verifikasi pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET is_verified = $1, updated_at = NOW() WHERE user_id = $2",
                is_verified, user_id
            )

    async def is_verified(self, user_id: int) -> bool:
        """Memeriksa apakah pengguna telah diverifikasi (bergabung dengan semua channel)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT is_verified FROM users WHERE user_id = $1", user_id
            )
            if not row:
                return False
            return row['is_verified']

    async def save_message(self, user_id: int, role: str, content: str, model: str = None):
        """Menyimpan pesan ke riwayat percakapan pengguna."""
        async with self.pool.acquire() as conn:
            # Simpan pesan
            await conn.execute(
                """
                INSERT INTO messages (user_id, role, content, model, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                user_id, role, content, model
            )

            # Cek jumlah riwayat pesan
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE user_id = $1", user_id
            )
            
            # Hapus yang tertua jika melebihi batas
            if count > MAX_HISTORY:
                to_delete = count - MAX_HISTORY
                await conn.execute(
                    """
                    DELETE FROM messages
                    WHERE id IN (
                        SELECT id FROM messages
                        WHERE user_id = $1
                        ORDER BY created_at ASC
                        LIMIT $2
                    )
                    """,
                    user_id, to_delete
                )

    async def get_history(self, user_id: int) -> List[Dict]:
        """Mendapatkan riwayat percakapan pengguna."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, model, created_at FROM messages WHERE user_id = $1 ORDER BY created_at",
                user_id
            )
            return [dict(row) for row in rows]

    async def clear_history(self, user_id: int):
        """Menghapus riwayat percakapan pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM messages WHERE user_id = $1", user_id
            )
