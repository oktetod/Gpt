#!/bin/bash
# Startup script with FORCED database migration

set -e

echo "🚀 Starting GPT-5 Telegram Bot v3.1..."
echo ""

# CRITICAL: Always run migration first
echo "🔄 Running database migration (FORCED)..."
if [ -f "migrate_database.py" ]; then
    # Run migration with detailed output
    python migrate_database.py || {
        echo "❌ Migration encountered errors"
        echo "⚠️  Attempting to continue..."
    }
    echo "✅ Migration phase completed"
    echo ""
else
    echo "⚠️  migrate_database.py not found"
    echo ""
fi

# Small delay to ensure DB is ready
sleep 2

# Start the bot
echo "🤖 Starting bot..."
exec python -u bot.py
