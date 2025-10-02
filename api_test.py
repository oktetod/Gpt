#!/usr/bin/env python3
"""
API Testing Script - Test all AI providers before running bot
Run: python test_api.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

async def test_cerebras():
    """Test Cerebras API"""
    print_header("Testing Cerebras API")
    
    api_key = os.getenv('CEREBRAS_API_KEY')
    if not api_key:
        print_error("CEREBRAS_API_KEY not found in .env")
        return False
    
    try:
        from cerebras.cloud.sdk import Cerebras
        
        client = Cerebras(api_key=api_key)
        
        # Test with a simple prompt
        print("Testing gpt-oss-120b model...")
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'Hello from Cerebras!' in one sentence."}],
            model="gpt-oss-120b",
            stream=True,
            max_completion_tokens=50,
            temperature=0.7
        )
        
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        
        print_success(f"Cerebras API working!")
        print(f"Response: {response.strip()}")
        
        # Test quota
        print("\nAvailable Cerebras models:")
        models = [
            "gpt-oss-120b (General chat)",
            "qwen-3-coder-480b (Code generation)",
            "qwen-3-235b-a22b-thinking-2507 (Deep reasoning)",
            "llama-4-maverick-17b-128e-instruct (Fast responses)"
        ]
        for model in models:
            print(f"  • {model}")
        
        return True
        
    except Exception as e:
        print_error(f"Cerebras API failed: {str(e)}")
        return False

async def test_nvidia():
    """Test NVIDIA API"""
    print_header("Testing NVIDIA API")
    
    api_key = os.getenv('NVIDIA_API_KEY')
    if not api_key:
        print_warning("NVIDIA_API_KEY not found in .env (Optional)")
        print("OCR and Nemotron Ultra features will be disabled")
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        # Test with a simple prompt
        print("Testing Nemotron Ultra model...")
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from NVIDIA!' in one sentence."}
            ],
            temperature=0.6,
            max_tokens=50,
            stream=True
        )
        
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        
        print_success("NVIDIA API working!")
        print(f"Response: {response.strip()}")
        
        print("\nAvailable NVIDIA features:")
        print("  • Nemotron Ultra (Ultra reasoning)")
        print("  • NemoRetriever OCR (Text extraction)")
        
        return True
        
    except Exception as e:
        print_error(f"NVIDIA API failed: {str(e)}")
        print_warning("OCR features will be disabled")
        return False

async def test_telegram():
    """Test Telegram credentials"""
    print_header("Testing Telegram Credentials")
    
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not api_id or not api_hash:
        print_error("TELEGRAM_API_ID or TELEGRAM_API_HASH missing")
        return False
    
    if not bot_token:
        print_error("TELEGRAM_BOT_TOKEN missing")
        return False
    
    print_success("Telegram credentials found")
    print(f"  API_ID: {api_id}")
    print(f"  API_HASH: {api_hash[:8]}...")
    print(f"  BOT_TOKEN: {bot_token[:15]}...")
    
    return True

async def test_database():
    """Test database connection"""
    print_header("Testing Database Connection")
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print_error("DATABASE_URL missing")
        return False
    
    try:
        import asyncpg
        
        # Try to connect
        conn = await asyncpg.connect(db_url)
        
        # Test query
        result = await conn.fetchval('SELECT 1')
        await conn.close()
        
        if result == 1:
            print_success("Database connection working!")
            print(f"  URL: {db_url[:40]}...")
            return True
        
    except Exception as e:
        print_error(f"Database connection failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║           Smart AI Bot - API Testing Suite                ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    results = {}
    
    # Test Telegram
    results['telegram'] = await test_telegram()
    
    # Test Database
    results['database'] = await test_database()
    
    # Test Cerebras
    results['cerebras'] = await test_cerebras()
    
    # Test NVIDIA (optional)
    results['nvidia'] = await test_nvidia()
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    
    print("Required Components:")
    if results['telegram']:
        print_success("Telegram credentials")
    else:
        print_error("Telegram credentials")
        all_passed = False
    
    if results['database']:
        print_success("Database connection")
    else:
        print_error("Database connection")
        all_passed = False
    
    if results['cerebras']:
        print_success("Cerebras API (Primary AI)")
    else:
        print_error("Cerebras API (Primary AI)")
        all_passed = False
    
    print("\nOptional Components:")
    if results['nvidia']:
        print_success("NVIDIA API (OCR & Ultra reasoning)")
    elif results['nvidia'] is None:
        print_warning("NVIDIA API (Not configured)")
    else:
        print_error("NVIDIA API (Configured but failing)")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    if all_passed:
        print(f"{GREEN}✓ All required tests passed! Bot is ready to start.{RESET}")
        print(f"\n{BLUE}Run:{RESET} make up  {BLUE}or{RESET}  docker-compose up -d")
    else:
        print(f"{RED}✗ Some tests failed. Please fix the issues above.{RESET}")
        print(f"\n{YELLOW}Check your .env file and API keys.{RESET}")
    
    print(f"{BLUE}{'='*60}{RESET}\n")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
