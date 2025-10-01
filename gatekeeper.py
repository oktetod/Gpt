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
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    statement_cache_size=0
                )
                await self.init_tables()
            except Exception as e:
                raise ConnectionError(f\"Gagal terhubung ke database: {e}\")

    async def close(self):
        """Menutup koneksi pool."""
        if self.pool:
            await self.pool.close()

    async def init_tables(self):
        """Inisialisasi tabel database jika belum ada."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_history_user 
                ON chat_history(user_id)
            ''')

    async def get_or_create_user(self, user_id: int, username: str = None, first_name: str = None):
        """Mendapatkan atau membuat pengguna baru di database."""
        async with self.pool.acquire() as conn:
            # Cek apakah user sudah ada
            query = """
                SELECT user_id, username, first_name, is_verified, last_active 
                FROM users WHERE user_id = $1
            """
            row = await conn.fetchrow(query, user_id)

            if row:
                # Update data jika ada perubahan
                await conn.execute(
                    """
                    UPDATE users SET username = $2, first_name = $3, last_active = NOW() 
                    WHERE user_id = $1
                    """,
                    user_id, username, first_name
                )
                return dict(row)
            else:
                # Buat user baru
                await conn.execute(
                    """
                    INSERT INTO users (user_id, username, first_name, is_verified, created_at, last_active)
                    VALUES ($1, $2, $3, false, NOW(), NOW())
                    """,
                    user_id, username, first_name
                )
                return {
                    'user_id': user_id,
                    'username': username,
                    'first_name': first_name,
                    'is_verified': False,
                    'last_active': None
                }

    async def is_verified(self, user_id: int) -> bool:
        """Memeriksa apakah pengguna telah diverifikasi (bergabung ke channel/grup)."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT is_verified FROM users WHERE user_id = $1", user_id
            )
            return result if result is not None else False

    async def set_verified(self, user_id: int, status: bool = True):
        """Mengatur status verifikasi pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET is_verified = $1 WHERE user_id = $2", status, user_id
            )

    async def save_message(self, user_id: int, role: str, content: str, image_url: str = None):
        """Menyimpan pesan ke riwayat chat."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_history (user_id, role, content, image_url, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                user_id, role, content, image_url
            )

    async def get_chat_history(self, user_id: int) -> List[Dict]:
        """Mengambil riwayat chat pengguna (maksimal MAX_HISTORY)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, image_url, created_at 
                FROM chat_history 
                WHERE user_id = $1 
                ORDER BY created_at ASC 
                LIMIT $2
                """,
                user_id, MAX_HISTORY
            )
            return [dict(row) for row in rows]

    async def clear_chat_history(self, user_id: int):
        """Menghapus riwayat chat pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM chat_history WHERE user_id = $1", user_id)
