"""
Smart AI Telegram Bot - Database Layer v3.1
Production-ready database with web search tracking
"""

import asyncpg
from typing import List, Dict, Optional
from datetime import datetime
from config import MAX_HISTORY
import logging

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database handler with connection pooling"""
    
    def __init__(self, url: str):
        self.url = url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'application_name': 'smart_ai_bot'
                }
            )
            await self._init_tables()
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    async def close(self) -> None:
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Database connections closed")
    
    async def _init_tables(self) -> None:
        """Create required tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Users table with all columns
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    language_code TEXT DEFAULT 'id',
                    is_premium BOOLEAN DEFAULT FALSE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Add columns if not exist (migration safe)
            migrations = [
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS image_count INTEGER DEFAULT 0",
            ]
            
            for migration in migrations:
                try:
                    await conn.execute(migration)
                except Exception as e:
                    logger.warning(f"Migration warning: {e}")
            
            # Chat history table with all columns
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Add foreign key if not exists
            try:
                await conn.execute('''
                    DO $ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'chat_history_user_id_fkey'
                        ) THEN
                            ALTER TABLE chat_history 
                            ADD CONSTRAINT chat_history_user_id_fkey 
                            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
                        END IF;
                    END $;
                ''')
            except:
                pass
            
            # Add additional chat_history columns
            chat_migrations = [
                "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS intent_type TEXT",
                "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS model_used TEXT",
                "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS tokens_used INTEGER DEFAULT 0",
                "ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS image_url TEXT",
            ]
            
            for migration in chat_migrations:
                try:
                    await conn.execute(migration)
                except Exception as e:
                    logger.warning(f"Chat history migration warning: {e}")
            
            # User stats table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id BIGINT PRIMARY KEY,
                    total_messages INTEGER DEFAULT 0,
                    total_images INTEGER DEFAULT 0,
                    total_code_requests INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    last_reset TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Add foreign key for user_stats
            try:
                await conn.execute('''
                    DO $ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'user_stats_user_id_fkey'
                        ) THEN
                            ALTER TABLE user_stats 
                            ADD CONSTRAINT user_stats_user_id_fkey 
                            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
                        END IF;
                    END $;
                ''')
            except:
                pass
            
            # Add web search column
            try:
                await conn.execute('''
                    ALTER TABLE user_stats 
                    ADD COLUMN IF NOT EXISTS total_web_searches INTEGER DEFAULT 0
                ''')
            except Exception as e:
                logger.warning(f"Stats migration warning: {e}")
            
            # Create indices for performance
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_history_user_time 
                ON chat_history(user_id, created_at DESC)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_users_last_active 
                ON users(last_active DESC)
            ''')
            
            logger.info("📊 Database tables initialized")
    
    async def get_or_create_user(
        self,
        user_id: int,
        username: str = None,
        first_name: str = None,
        last_name: str = None,
        language_code: str = 'id'
    ) -> Dict:
        """Get existing user or create new one"""
        async with self.pool.acquire() as conn:
            # Try to get existing user
            user = await conn.fetchrow(
                'SELECT * FROM users WHERE user_id = $1',
                user_id
            )
            
            if user:
                # Update last active and user info
                await conn.execute('''
                    UPDATE users 
                    SET username = $1,
                        first_name = $2,
                        last_name = $3,
                        language_code = $4,
                        last_active = NOW(),
                        message_count = message_count + 1
                    WHERE user_id = $5
                ''', username, first_name, last_name, language_code, user_id)
                
                logger.info(f"🔄 Updated user {user_id}")
                return dict(user)
            else:
                # Create new user
                await conn.execute('''
                    INSERT INTO users (
                        user_id, username, first_name, last_name, 
                        language_code, is_verified
                    )
                    VALUES ($1, $2, $3, $4, $5, FALSE)
                ''', user_id, username, first_name, last_name, language_code)
                
                # Create stats entry
                await conn.execute('''
                    INSERT INTO user_stats (user_id)
                    VALUES ($1)
                ''', user_id)
                
                logger.info(f"🆕 Created new user {user_id}")
                
                # Return the newly created user
                user = await conn.fetchrow(
                    'SELECT * FROM users WHERE user_id = $1',
                    user_id
                )
                return dict(user)
    
    async def save_message(
        self,
        user_id: int,
        role: str,
        content: str,
        intent_type: str = None,
        model_used: str = None,
        tokens_used: int = 0,
        image_url: str = None
    ) -> None:
        """Save message to chat history"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO chat_history (
                    user_id, role, content, intent_type, 
                    model_used, tokens_used, image_url
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', user_id, role, content, intent_type, 
                model_used, tokens_used, image_url)
            
            # Update stats
            if role == 'user':
                await conn.execute('''
                    UPDATE user_stats
                    SET total_messages = total_messages + 1,
                        total_tokens = total_tokens + $1
                    WHERE user_id = $2
                ''', tokens_used, user_id)
                
                if intent_type == 'image':
                    await conn.execute('''
                        UPDATE user_stats
                        SET total_images = total_images + 1
                        WHERE user_id = $1
                    ''', user_id)
                elif intent_type == 'code':
                    await conn.execute('''
                        UPDATE user_stats
                        SET total_code_requests = total_code_requests + 1
                        WHERE user_id = $1
                    ''', user_id)
                elif intent_type == 'web_search':
                    await conn.execute('''
                        UPDATE user_stats
                        SET total_web_searches = total_web_searches + 1
                        WHERE user_id = $1
                    ''', user_id)
    
    async def get_chat_history(
        self,
        user_id: int,
        limit: int = MAX_HISTORY
    ) -> List[Dict]:
        """Get recent chat history for user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT role, content, intent_type, model_used, 
                       tokens_used, image_url, created_at
                FROM chat_history
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ''', user_id, limit)
            
            # Return in chronological order
            return [dict(row) for row in reversed(rows)]
    
    async def clear_history(self, user_id: int) -> int:
        """Clear chat history for user"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                'DELETE FROM chat_history WHERE user_id = $1',
                user_id
            )
            deleted = int(result.split()[-1])
            logger.info(f"🗑️ Cleared {deleted} messages for user {user_id}")
            return deleted
    
    async def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for user"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow('''
                SELECT * FROM user_stats WHERE user_id = $1
            ''', user_id)
            
            if stats:
                return dict(stats)
            return {
                'total_messages': 0,
                'total_images': 0,
                'total_code_requests': 0,
                'total_web_searches': 0,
                'total_tokens': 0
            }
    
    async def update_verification(
        self,
        user_id: int,
        is_verified: bool
    ) -> None:
        """Update user verification status"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE users
                SET is_verified = $1
                WHERE user_id = $2
            ''', is_verified, user_id)
            
            logger.info(f"✓ Updated verification for user {user_id}: {is_verified}")
    
    async def get_active_users(self, days: int = 7) -> List[Dict]:
        """Get list of active users in the last N days"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT user_id, username, first_name, message_count,
                       last_active
                FROM users
                WHERE last_active > NOW() - INTERVAL '%s days'
                ORDER BY last_active DESC
            ''' % days)
            
            return [dict(row) for row in rows]
    
    async def check_rate_limit(
        self,
        user_id: int,
        limit_type: str,
        max_count: int,
        time_window_minutes: int = 60
    ) -> bool:
        """Check if user has exceeded rate limit"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval('''
                SELECT COUNT(*)
                FROM chat_history
                WHERE user_id = $1
                  AND created_at > NOW() - INTERVAL '%s minutes'
                  AND ($2 = 'all' OR intent_type = $2)
            ''' % time_window_minutes, user_id, limit_type)
            
            return count < max_count
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return False