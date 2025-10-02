#!/usr/bin/env python3
"""
Web Search Testing Script
Test all search engines before deploying
"""

import asyncio
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Colors
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

async def test_web_search():
    """Test web search functionality"""
    print_header("Testing Web Search Engines")
    
    try:
        from web_search import MultiSearchEngine
        
        search = MultiSearchEngine()
        test_query = "artificial intelligence 2024"
        
        print(f"Testing query: '{test_query}'\n")
        
        # Test each engine
        sources = ['duckduckgo', 'wikipedia', 'bing', 'archive', 'scholar', 'news']
        results = {}
        
        for source in sources:
            try:
                print(f"Testing {source}...")
                
                if source == 'duckduckgo':
                    result = await search.search_duckduckgo(test_query, max_results=3)
                elif source == 'wikipedia':
                    result = await search.search_wikipedia(test_query, max_results=3)
                elif source == 'bing':
                    result = await search.search_bing(test_query, max_results=3)
                elif source == 'archive':
                    result = await search.search_archive(test_query, max_results=3)
                elif source == 'scholar':
                    result = await search.search_scholar(test_query, max_results=3)
                elif source == 'news':
                    result = await search.search_news(test_query, max_results=3)
                
                results[source] = result
                
                if result and len(result) > 0:
                    print_success(f"{source}: Found {len(result)} results")
                    for i, item in enumerate(result[:2], 1):
                        print(f"  {i}. {item.get('title', 'No title')[:50]}...")
                else:
                    print_warning(f"{source}: No results")
                
            except Exception as e:
                print_error(f"{source}: {str(e)}")
                results[source] = None
            
            print()
        
        # Test unified search
        print_header("Testing Unified Search")
        
        unified_results = await search.unified_search(
            test_query,
            sources=['duckduckgo', 'wikipedia', 'bing'],
            max_results_per_source=3
        )
        
        total = sum(len(items) for items in unified_results.values())
        print_success(f"Unified search returned {total} total results")
        
        for source, items in unified_results.items():
            print(f"  • {source}: {len(items)} results")
        
        # Test formatting
        print_header("Testing Result Formatting")
        
        formatted = search.format_search_results(unified_results, max_total=10)
        print(f"Formatted output ({len(formatted)} chars):")
        print(formatted[:500] + "...")
        
        await search.close()
        
        return True
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        print_warning("Make sure web_search.py is in the same directory")
        return False
    
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False

async def test_ai_integration():
    """Test AI engine with web search"""
    print_header("Testing AI Engine Integration")
    
    try:
        from ai_engine import MultiProviderAI
        
        ai = MultiProviderAI()
        
        # Test web search detection
        print("Testing intent detection...")
        
        test_queries = [
            "cari berita terbaru AI",
            "what is quantum computing",
            "jelaskan machine learning",
            "buatkan code Python",
        ]
        
        for query in test_queries:
            intent = await ai.detect_intent(query)
            search_flag = "🌐 WITH SEARCH" if intent.get('use_web_search') else "💬 NO SEARCH"
            print(f"{search_flag} | {query}")
            print(f"  Type: {intent['type']}, Model: {intent['model']}")
        
        print()
        
        # Test actual search
        print("Testing web search execution...")
        search_result = await ai.perform_web_search("artificial intelligence")
        
        if search_result['success']:
            print_success(f"Search returned {search_result['total_results']} results")
            print(f"  Sources: {', '.join(search_result['sources_used'])}")
        else:
            print_error(f"Search failed: {search_result.get('error')}")
        
        await ai.close()
        
        return True
        
    except Exception as e:
        print_error(f"AI integration test failed: {e}")
        return False

async def test_api_keys():
    """Test API key rotation"""
    print_header("Testing API Key Rotation")
    
    try:
        from config import CEREBRAS_API_KEYS, NVIDIA_API_KEY
        
        print(f"Cerebras API Keys: {len(CEREBRAS_API_KEYS)} configured")
        for i, key in enumerate(CEREBRAS_API_KEYS, 1):
            print(f"  Key {i}: {key[:8]}...{key[-8:]}")
        
        if CEREBRAS_API_KEYS:
            print_success("Cerebras keys loaded")
        else:
            print_error("No Cerebras keys found!")
        
        print()
        
        if NVIDIA_API_KEY:
            print(f"NVIDIA API Key: {NVIDIA_API_KEY[:8]}...{NVIDIA_API_KEY[-8:]}")
            print_success("NVIDIA key loaded")
        else:
            print_warning("NVIDIA key not configured (optional)")
        
        return len(CEREBRAS_API_KEYS) > 0
        
    except Exception as e:
        print_error(f"API key test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║        GPT-5 Bot - Web Search Testing Suite v3.1          ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    results = {}
    
    # Test API keys
    results['api_keys'] = await test_api_keys()
    
    # Test web search
    results['web_search'] = await test_web_search()
    
    # Test AI integration
    results['ai_integration'] = await test_ai_integration()
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    
    if results['api_keys']:
        print_success("API Keys Configuration")
    else:
        print_error("API Keys Configuration")
        all_passed = False
    
    if results['web_search']:
        print_success("Web Search Engines")
    else:
        print_error("Web Search Engines")
        all_passed = False
    
    if results['ai_integration']:
        print_success("AI Engine Integration")
    else:
        print_error("AI Engine Integration")
        all_passed = False
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    if all_passed:
        print(f"{GREEN}✓ All tests passed! Bot is ready to deploy.{RESET}")
        print(f"\n{BLUE}Run:{RESET} make up  {BLUE}or{RESET}  docker-compose up -d")
    else:
        print(f"{RED}✗ Some tests failed. Please fix the issues above.{RESET}")
        print(f"\n{YELLOW}Check your configuration and API keys.{RESET}")
    
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    return all_passed

if __name__ == '__main__':
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        sys.exit(1)
