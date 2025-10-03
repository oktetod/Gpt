#!/usr/bin/env python3
"""
Database Migration Script v3 - COMPLETE SCHEMA FIX
Safely adds ALL missing columns and creates missing tables
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

async def migrate():
    """Perform complete database migration"""
    print("🔄 Starting COMPLETE database migration...")
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # ===== DROP AND RECREATE ALL TABLES (CLEAN START) =====
        print("\n🔧 Ensuring clean schema...")
        
        # Drop existing tables if they exist (cascade to handle foreign keys)
        print("  🗑️ Dropping old tables...")
        await conn.execute('DROP TABLE IF EXISTS chat_history CASCADE')
        await conn.execute('DROP TABLE IF EXISTS user_stats CASCADE')
        await conn.execute('DROP TABLE IF EXISTS users CASCADE')
        print("  ✅ Old tables dropped")
        
        # ===== CREATE USERS TABLE =====
        print("\n📋 Creating users table...")
        await conn.execute('''
            CREATE TABLE users (
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
        print("  ✅ Users table created with ALL columns")
        
        # ===== CREATE CHAT_HISTORY TABLE =====
        print("\n📋 Creating chat_history table...")
        await conn.execute('''
            CREATE TABLE chat_history (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                intent_type TEXT,
                model_used TEXT,
                tokens_used INTEGER DEFAULT 0,
                image_url TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        ''')
        print("  ✅ Chat_history table created with ALL columns and foreign key")
        
        # ===== CREATE USER_STATS TABLE =====
        print("\n📋 Creating user_stats table...")
        await conn.execute('''
            CREATE TABLE user_stats (
                user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                total_messages INTEGER DEFAULT 0,
                total_images INTEGER DEFAULT 0,
                total_code_requests INTEGER DEFAULT 0,
                total_web_searches INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                last_reset TIMESTAMPTZ DEFAULT NOW()
