.PHONY: help build up down restart logs shell clean backup install

# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
BLUE   := \033[0;34m
NC     := \033[0m

help: ## Show this help
	@echo "$(GREEN)╔═══════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║   Smart AI Telegram Bot Commands     ║$(NC)"
	@echo "$(GREEN)╚═══════════════════════════════════════╝$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Setup and start bot (first time)
	@echo "$(BLUE)Setting up Smart AI Bot...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env file. Please edit it!$(NC)"; \
		echo "$(RED)Run: nano .env$(NC)"; \
		echo "$(RED)Then run: make up$(NC)"; \
	else \
		echo "$(GREEN).env exists$(NC)"; \
		make build; \
		make up; \
	fi

build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker-compose build

up: ## Start bot
	@echo "$(GREEN)Starting bot...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Bot started! Use 'make logs' to see output$(NC)"

down: ## Stop bot
	@echo "$(RED)Stopping bot...$(NC)"
	@docker-compose down

restart: ## Restart bot
	@echo "$(YELLOW)Restarting bot...$(NC)"
	@docker-compose restart
	@echo "$(GREEN)✓ Bot restarted!$(NC)"

logs: ## Show bot logs (follow)
	@docker-compose logs -f telegram-bot

logs-tail: ## Show last 100 lines
	@docker-compose logs --tail=100 telegram-bot

status: ## Show bot status
	@echo "$(GREEN)Bot Status:$(NC)"
	@docker-compose ps
	@echo ""
	@docker stats smart-ai-bot --no-stream 2>/dev/null || echo "$(RED)Bot not running$(NC)"

shell: ## Open shell in container
	@docker-compose exec telegram-bot /bin/bash

rebuild: ## Rebuild and restart
	@echo "$(YELLOW)Rebuilding...$(NC)"
	@docker-compose up -d --build
	@echo "$(GREEN)✓ Rebuilt!$(NC)"

clean: ## Remove everything
	@echo "$(RED)Cleaning up...$(NC)"
	@docker-compose down -v --rmi local
	@echo "$(GREEN)✓ Clean!$(NC)"

backup: ## Backup sessions
	@echo "$(YELLOW)Backing up...$(NC)"
	@mkdir -p backups
	@tar -czf backups/sessions_$$(date +%Y%m%d_%H%M%S).tar.gz sessions/ 2>/dev/null || echo "No sessions yet"
	@echo "$(GREEN)✓ Backup done!$(NC)"

dev: ## Run locally (without Docker)
	@echo "$(YELLOW)Running locally...$(NC)"
	@python bot.py

check: ## Check environment
	@echo "$(GREEN)Checking...$(NC)"
	@[ -f .env ] && echo "$(GREEN)✓ .env exists$(NC)" || echo "$(RED)✗ .env missing$(NC)"
	@[ -f bot.py ] && echo "$(GREEN)✓ bot.py exists$(NC)" || echo "$(RED)✗ bot.py missing$(NC)"
	@docker --version > /dev/null 2>&1 && echo "$(GREEN)✓ Docker installed$(NC)" || echo "$(RED)✗ Docker not installed$(NC)"
	@docker-compose --version > /dev/null 2>&1 && echo "$(GREEN)✓ Docker Compose installed$(NC)" || echo "$(RED)✗ Docker Compose not installed$(NC)"

health: ## Check bot health
	@docker inspect smart-ai-bot --format='{{.State.Status}}' 2>/dev/null | grep -q running && \
		echo "$(GREEN)✓ Bot running$(NC)" || echo "$(RED)✗ Bot not running$(NC)"

update: ## Pull and restart
	@echo "$(YELLOW)Updating...$(NC)"
	@git pull 2>/dev/null || echo "Not a git repo"
	@make rebuild

prune: ## Clean Docker system
	@echo "$(RED)Pruning Docker...$(NC)"
	@docker system prune -af
	@echo "$(GREEN)✓ Pruned!$(NC)"
