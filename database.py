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
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            await self.init_tables()
            print("Koneksi database berhasil dibuat.")
    
    async def init_tables(self):
        """Membuat tabel jika belum ada."""
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
                ON chat_history(user_id, created_at DESC)
            ''')
    
    async def get_or_create_user(self, user_id: int, username: str = None, first_name: str = None):
        """Mendapatkan atau membuat pengguna baru dan memperbarui waktu aktif."""
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            
            if not user:
                await conn.execute(
                    'INSERT INTO users (user_id, username, first_name) VALUES ($1, $2, $3)',
                    user_id, username, first_name
                )
            else:
                await conn.execute(
                    'UPDATE users SET last_active = NOW() WHERE user_id = $1',
                    user_id
                )
            
            return await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
    
    async def set_verified(self, user_id: int, verified: bool = True):
        """Mengatur status verifikasi pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                'UPDATE users SET is_verified = $1 WHERE user_id = $2',
                verified, user_id
            )
    
    async def is_verified(self, user_id: int) -> bool:
        """Memeriksa apakah pengguna sudah terverifikasi."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                'SELECT is_verified FROM users WHERE user_id = $1', user_id
            )
            return result or False
    
    async def add_message(self, user_id: int, role: str, content: str, image_url: str = None):
        """Menambahkan pesan ke riwayat percakapan dan membatasi jumlahnya."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO chat_history (user_id, role, content, image_url) VALUES ($1, $2, $3, $4)',
                user_id, role, content, image_url
            )
            
            # Membatasi jumlah riwayat agar database tidak membengkak
            await conn.execute('''
                DELETE FROM chat_history
                WHERE id IN (
                    SELECT id FROM chat_history
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    OFFSET $2
                )
            ''', user_id, MAX_HISTORY * 2) # *2 untuk user dan asisten
    
    async def get_history(self, user_id: int) -> List[Dict]:
        """Mengambil riwayat percakapan pengguna."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                'SELECT role, content, image_url FROM chat_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2',
                user_id, MAX_HISTORY * 2
            )
            # Mengembalikan dalam urutan yang benar (paling lama ke paling baru)
            return [dict(row) for row in reversed(rows)]
    
    async def clear_history(self, user_id: int):
        """Menghapus seluruh riwayat percakapan pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM chat_history WHERE user_id = $1', user_id)
