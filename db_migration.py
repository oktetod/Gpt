#!/usr/bin/env python3
"""
Database Migration Script
Safely adds missing columns and updates schema
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

async def migrate():
    """Perform database migration"""
    print("🔄 Starting database migration...")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # ===== USERS TABLE MIGRATION =====
        print("\n📋 Migrating users table...")
        
        # Add message_count if not exists
        try:
            await conn.execute('''
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0
            ''')
            print("  ✅ Added message_count column")
        except Exception as e:
            print(f"  ⚠️ message_count: {e}")
        
        # Add image_count if not exists
        try:
            await conn.execute('''
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS image_count INTEGER DEFAULT 0
            ''')
            print("  ✅ Added image_count column")
        except Exception as e:
            print(f"  ⚠️ image_count: {e}")
        
        # ===== CHAT_HISTORY TABLE MIGRATION =====
        print("\n📋 Migrating chat_history table...")
        
        # Add intent_type if not exists
        try:
            await conn.execute('''
                ALTER TABLE chat_history 
                ADD COLUMN IF NOT EXISTS intent_type TEXT
            ''')
            print("  ✅ Added intent_type column")
        except Exception as e:
            print(f"  ⚠️ intent_type: {e}")
        
        # Add model_used if not exists
        try:
            await conn.execute('''
                ALTER TABLE chat_history 
                ADD COLUMN IF NOT EXISTS model_used TEXT
            ''')
            print("  ✅ Added model_used column")
        except Exception as e:
            print(f"  ⚠️ model_used: {e}")
        
        # Add tokens_used if not exists
        try:
            await conn.execute('''
                ALTER TABLE chat_history 
                ADD COLUMN IF NOT EXISTS tokens_used INTEGER DEFAULT 0
            ''')
            print("  ✅ Added tokens_used column")
        except Exception as e:
            print(f"  ⚠️ tokens_used: {e}")
        
        # Add image_url if not exists
        try:
            await conn.execute('''
                ALTER TABLE chat_history 
                ADD COLUMN IF NOT EXISTS image_url TEXT
            ''')
            print("  ✅ Added image_url column")
        except Exception as e:
            print(f"  ⚠️ image_url: {e}")
        
        # ===== USER_STATS TABLE MIGRATION =====
        print("\n📋 Migrating user_stats table...")
        
        # Add total_web_searches if not exists
        try:
            await conn.execute('''
                ALTER TABLE user_stats 
                ADD COLUMN IF NOT EXISTS total_web_searches INTEGER DEFAULT 0
            ''')
            print("  ✅ Added total_web_searches column")
        except Exception as e:
            print(f"  ⚠️ total_web_searches: {e}")
        
        # ===== VERIFY SCHEMA =====
        print("\n🔍 Verifying schema...")
        
        # Check users table
        users_columns = await conn.fetch('''
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'users'
            ORDER BY ordinal_position
        ''')
        print("\n  Users table columns:")
        for col in users_columns:
            print(f"    • {col['column_name']} ({col['data_type']})")
        
        # Check chat_history table
        history_columns = await conn.fetch('''
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'chat_history'
            ORDER BY ordinal_position
        ''')
        print("\n  Chat_history table columns:")
        for col in history_columns:
            print(f"    • {col['column_name']} ({col['data_type']})")
        
        # Check user_stats table
        stats_columns = await conn.fetch('''
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'user_stats'
            ORDER BY ordinal_position
        ''')
        print("\n  User_stats table columns:")
        for col in stats_columns:
            print(f"    • {col['column_name']} ({col['data_type']})")
        
        await conn.close()
        print("\n✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False

if __name__ == '__main__':
    success = asyncio.run(migrate())
    exit(0 if success else 1)
