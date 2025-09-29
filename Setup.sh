#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Banner
clear
echo -e "${BLUE}"
echo "╔═════════════════════════════════════════════╗"
echo "║                                             ║"
echo "║      🤖 SMART AI TELEGRAM BOT 🤖           ║"
echo "║                                             ║"
echo "║      Powered by Pollinations.AI             ║"
echo "║                                             ║"
echo "╚═════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Check Docker
echo -e "${YELLOW}[1/5] Checking requirements...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not installed!${NC}"
    echo -e "${YELLOW}Install: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not installed!${NC}"
    echo -e "${YELLOW}Install: https://docs.docker.com/compose/install/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose installed${NC}"

# Create directories
echo -e "\n${YELLOW}[2/5] Creating directories...${NC}"
mkdir -p sessions backups
echo -e "${GREEN}✓ Directories created${NC}"

# Setup .env
echo -e "\n${YELLOW}[3/5] Configuring environment...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}\n"
    
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${YELLOW}Please enter your credentials:${NC}\n"
    
    # Get API_ID
    echo -e "${BLUE}1. TELEGRAM_API_ID${NC}"
    echo -e "   Get from: ${YELLOW}https://my.telegram.org${NC}"
    read -p "   Enter API_ID: " api_id
    
    # Get API_HASH
    echo -e "\n${BLUE}2. TELEGRAM_API_HASH${NC}"
    echo -e "   Get from: ${YELLOW}https://my.telegram.org${NC}"
    read -p "   Enter API_HASH: " api_hash
    
    # Get BOT_TOKEN
    echo -e "\n${BLUE}3. TELEGRAM_BOT_TOKEN${NC}"
    echo -e "   Get from: ${YELLOW}@BotFather${NC} on Telegram"
    read -p "   Enter BOT_TOKEN: " bot_token
    
    # Update .env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your_api_id_here/$api_id/" .env
        sed -i '' "s/your_api_hash_here/$api_hash/" .env
        sed -i '' "s/your_bot_token_here/$bot_token/" .env
    else
        # Linux
        sed -i "s/your_api_id_here/$api_id/" .env
        sed -i "s/your_api_hash_here/$api_hash/" .env
        sed -i "s/your_bot_token_here/$bot_token/" .env
    fi
    
    echo -e "\n${GREEN}✓ Credentials configured${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Build image
echo -e "\n${YELLOW}[4/5] Building Docker image...${NC}"
docker-compose build
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Start bot
echo -e "\n${YELLOW}[5/5] Starting bot...${NC}"
docker-compose up -d
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bot started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start${NC}"
    exit 1
fi

# Show status
sleep 3
echo -e "\n${CYAN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}\n"

echo -e "${YELLOW}Bot Status:${NC}"
docker-compose ps

echo -e "\n${YELLOW}📝 Useful Commands:${NC}"
echo -e "  ${BLUE}make logs${NC}        - View logs"
echo -e "  ${BLUE}make status${NC}      - Check status"
echo -e "  ${BLUE}make restart${NC}     - Restart bot"
echo -e "  ${BLUE}make down${NC}        - Stop bot"
echo -e "  ${BLUE}make help${NC}        - All commands"

echo -e "\n${YELLOW}View logs:${NC}"
echo -e "  ${BLUE}docker-compose logs -f${NC}"

echo -e "\n${GREEN}✨ Bot is running! Open Telegram and send /start${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}\n"
