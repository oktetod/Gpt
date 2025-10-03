"""
Smart AI Telegram Bot - Database Layer v3.2 (COMPLETE pgBouncer Fix)
Fixed: Use fetch/fetchrow/fetchval instead of execute for all queries
"""

import asyncpg
from typing import List, Dict, Optional
from datetime import datetime
from config import MAX_HISTORY
import logging

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database handler with FULL pgBouncer compatibility"""
    
    def __init__(self, url: str):
        self.url = url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Initialize database connection pool with pgBouncer compatibility"""
        try:
            # CRITICAL: Disable statement cache for pgBouncer
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                statement_cache_size=0,  # Disable prepared statements
                server_settings={
                    'application_name': 'smart_ai_bot'
                }
            )
            await self._init_tables()
            logger.info("✅ Database connected (pgBouncer compatible)")
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
            # Execute each statement separately for pgBouncer compatibility
            
            # Create users table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    language_code TEXT DEFAULT 'id',
                    is_premium BOOLEAN DEFAULT FALSE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    message_count INTEGER DEFAULT 0,
                    image_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Create chat_history table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    intent_type TEXT,
                    model_used TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    image_url TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Create user_stats table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id BIGINT PRIMARY KEY,
                    total_messages INTEGER DEFAULT 0,
                    total_images INTEGER DEFAULT 0,
                    total_code_requests INTEGER DEFAULT 0,
                    total_web_searches INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    last_reset TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Create indices (one at a time)
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
                # Update using fetchval instead of execute
                await conn.fetchval('''
                    UPDATE users 
                    SET username = $1,
                        first_name = $2,
                        last_name = $3,
                        language_code = $4,
                        last_active = NOW(),
                        message_count = message_count + 1
                    WHERE user_id = $5
                    RETURNING user_id
                ''', username, first_name, last_name, language_code, user_id)
                
                logger.info(f"🔄 Updated user {user_id}")
                return dict(user)
            else:
                # Create new user using fetchval
                await conn.fetchval('''
                    INSERT INTO users (
                        user_id, username, first_name, last_name, 
                        language_code, is_verified
                    )
                    VALUES ($1, $2, $3, $4, $5, FALSE)
                    RETURNING user_id
                ''', user_id, username, first_name, last_name, language_code)
                
                # Create stats entry
                await conn.fetchval('''
                    INSERT INTO user_stats (user_id)
                    VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                    RETURNING user_id
                ''', user_id)
                
                logger.info(f"🆕 Created new user {user_id}")
                
                # Return the newly created user
                new_user = await conn.fetchrow(
                    'SELECT * FROM users WHERE user_id = $1',
                    user_id
                )
                return dict(new_user)
    
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
            # Use fetchval instead of execute
            await conn.fetchval('''
                INSERT INTO chat_history (
                    user_id, role, content, intent_type, 
                    model_used, tokens_used, image_url
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            ''', user_id, role, content, intent_type, 
                model_used, tokens_used, image_url)
            
            # Update stats
            if role == 'user':
                await conn.fetchval('''
                    UPDATE user_stats
                    SET total_messages = total_messages + 1,
                        total_tokens = total_tokens + $1
                    WHERE user_id = $2
                    RETURNING user_id
                ''', tokens_used, user_id)
                
                # Update specific counters
                if intent_type == 'image':
                    await conn.fetchval('''
                        UPDATE user_stats
                        SET total_images = total_images + 1
                        WHERE user_id = $1
                        RETURNING user_id
                    ''', user_id)
                elif intent_type == 'code':
                    await conn.fetchval('''
                        UPDATE user_stats
                        SET total_code_requests = total_code_requests + 1
                        WHERE user_id = $1
                        RETURNING user_id
                    ''', user_id)
                elif intent_type == 'web_search':
                    await conn.fetchval('''
                        UPDATE user_stats
                        SET total_web_searches = total_web_searches + 1
                        WHERE user_id = $1
                        RETURNING user_id
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
            
            return [dict(row) for row in reversed(rows)]
    
    async def clear_history(self, user_id: int) -> int:
        """Clear chat history for user"""
        async with self.pool.acquire() as conn:
            # Use fetch to get count, then delete
            count = await conn.fetchval(
                'SELECT COUNT(*) FROM chat_history WHERE user_id = $1',
                user_id
            )
            
            await conn.fetchval(
                'DELETE FROM chat_history WHERE user_id = $1 RETURNING user_id',
                user_id
            )
            
            deleted = int(count) if count else 0
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
            
            # Create stats if not exists
            await conn.fetchval('''
                INSERT INTO user_stats (user_id)
                VALUES ($1)
                ON CONFLICT (user_id) DO NOTHING
                RETURNING user_id
            ''', user_id)
            
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
            await conn.fetchval('''
                UPDATE users
                SET is_verified = $1
                WHERE user_id = $2
                RETURNING user_id
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
            # Use fetchval instead of execute with interval formatting
            count = await conn.fetchval(f'''
                SELECT COUNT(*)
                FROM chat_history
                WHERE user_id = $1
                  AND created_at > NOW() - INTERVAL '{time_window_minutes} minutes'
                  AND ($2 = 'all' OR intent_type = $2)
            ''', user_id, limit_type)
            
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
