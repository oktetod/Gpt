# 📄 database.py v2 (versi lengkap dan benar)

import asyncpg
from typing import List, Dict, Optional
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
                    statement_cache_size=0  # Kompatibel dengan Supabase/PGBouncer
                )
                await self.init_tables()
                print("✅ Koneksi database berhasil dibuat.")
            except Exception as e:
                raise ConnectionError(f"Gagal terhubung ke database: {e}")

    async def close(self):
        """Menutup koneksi pool."""
        if self.pool:
            await self.pool.close()
            print("🔒 Koneksi database ditutup.")

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
            print("📊 Tabel database siap digunakan.")
    
    async def get_or_create_user(self, user_id: int, username: str = None, first_name: str = None):
        """Mendapatkan atau membuat pengguna baru dan memperbarui waktu aktif."""
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            
            if not user:
                await conn.execute(
                    '''
                    INSERT INTO users (user_id, username, first_name, is_verified)
                    VALUES ($1, $2, $3, FALSE)
                    ''',
                    user_id, username, first_field
                )
                print(f"🆕 User {user_id} dibuat baru.")
                return {
                    'user_id': user_id,
                    'username': username,
                    'first_name': first_name,
                    'is_verified': False,
                    'created_at': None,
                    'last_active': None
                }
            else:
                await conn.execute(
                    '''
                    UPDATE users 
                    SET username = $1, first_name = $2, last_active = NOW() 
                    WHERE user_id = $3
                    ''',
                    username, first_name, user_id
                )
                print(f"🔄 User {user_id} diperbarui.")
                return dict(user)

    async def save_message(self, user_id: int, role: str, content: str, image_url: str = None):
        """Simpan pesan ke riwayat obrolan."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO chat_history (user_id, role, content, image_url)
                VALUES ($1, $2, $3, $4)
                ''',
                user_id, role, content, image_url
            )

    async def get_chat_history(self, user_id: int, limit: int = MAX_HISTORY) -> List[Dict]:
        """Ambil riwayat obrolan terakhir untuk pengguna."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                '''
                SELECT role, content, image_url, created_at 
                FROM chat_history 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2
                ''',
                user_id, limit
            )
            return [dict(row) for row in rows]

    async def clear_history(self, user_id: int):
        """Hapus riwayat obrolan pengguna."""
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM chat_history WHERE user_id = $1', user_id)
