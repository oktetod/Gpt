#!/bin/bash
# Startup script with automatic database migration

set -e

echo "🚀 Starting GPT-5 Telegram Bot v3.1..."
echo ""

# Check if migration script exists
if [ -f "migrate_database.py" ]; then
    echo "🔄 Running database migration..."
    python migrate_database.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Database migration completed"
        echo ""
    else
        echo "⚠️ Migration had issues, but continuing..."
        echo ""
    fi
fi

# Start the bot
echo "🤖 Starting bot..."
exec python -u bot.py
