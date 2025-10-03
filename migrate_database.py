# file: migrate_database.py

#!/usr/bin/env python3
"""
Database Migration Script v2
Safely adds all missing columns and updates schema
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
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # ===== USERS TABLE MIGRATION =====
        print("\n📋 Migrating users table...")
        columns_to_add = {
            "last_name": "TEXT",
            "message_count": "INTEGER DEFAULT 0",
            "image_count": "INTEGER DEFAULT 0"
        }
        for col, data_type in columns_to_add.items():
            try:
                await conn.execute(f'ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {data_type}')
                print(f"  ✅ Added '{col}' column to users table")
            except Exception as e:
                print(f"  ⚠️ Could not add column '{col}': {e}")

        # ===== CHAT_HISTORY TABLE MIGRATION =====
        print("\n📋 Migrating chat_history table...")
        history_columns = {
            "intent_type": "TEXT",
            "model_used": "TEXT",
            "tokens_used": "INTEGER DEFAULT 0",
            "image_url": "TEXT"
        }
        for col, data_type in history_columns.items():
            try:
                await conn.execute(f'ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS {col} {data_type}')
                print(f"  ✅ Added '{col}' column to chat_history table")
            except Exception as e:
                print(f"  ⚠️ Could not add column '{col}': {e}")
        
        # ===== USER_STATS TABLE MIGRATION =====
        print("\n📋 Migrating user_stats table...")
        try:
            await conn.execute('ALTER TABLE user_stats ADD COLUMN IF NOT EXISTS total_web_searches INTEGER DEFAULT 0')
            print("  ✅ Added 'total_web_searches' column to user_stats table")
        except Exception as e:
            print(f"  ⚠️ Could not add column 'total_web_searches': {e}")

        print("\n✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False
    finally:
        if conn:
            await conn.close()
            print("\n🔒 Database connection closed")

if __name__ == '__main__':
    success = asyncio.run(migrate())
    exit(0 if success else 1)
