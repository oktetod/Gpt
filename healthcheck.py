"""
Healthcheck HTTP Server for Kinsta
Runs alongside the Telegram bot to provide health status
"""

import asyncio
import logging
from aiohttp import web
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthCheckServer:
    """Simple HTTP server for health checks"""
    
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.start_time = datetime.now()
        self.is_healthy = True
        self.bot_status = "Starting..."
        
        # Setup routes
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/status', self.handle_status)
    
    async def handle_root(self, request):
        """Handle root endpoint"""
        return web.Response(
            text="Smart AI Telegram Bot is running!\n",
            content_type='text/plain'
        )
    
    async def handle_health(self, request):
        """Handle health check endpoint"""
        if self.is_healthy:
            return web.Response(
                text="OK",
                status=200
            )
        else:
            return web.Response(
                text="Unhealthy",
                status=503
            )
    
    async def handle_status(self, request):
        """Handle status endpoint with details"""
        uptime = datetime.now() - self.start_time
        
        status = {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "bot_status": self.bot_status,
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime": str(uptime).split('.')[0],
            "timestamp": datetime.now().isoformat()
        }
        
        return web.json_response(status)
    
    def set_bot_status(self, status: str):
        """Update bot status"""
        self.bot_status = status
        logger.info(f"📊 Bot status: {status}")
    
    def set_healthy(self, healthy: bool):
        """Update health status"""
        self.is_healthy = healthy
    
    async def start(self):
        """Start the HTTP server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"🌐 Health check server running on port {self.port}")
        logger.info(f"   Endpoints:")
        logger.info(f"   - GET / (root)")
        logger.info(f"   - GET /health (health check)")
        logger.info(f"   - GET /status (detailed status)")


# Global instance
health_server = HealthCheckServer()
