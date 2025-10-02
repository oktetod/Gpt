"""
Advanced Web Search Engine with Multiple Sources
Supports: DuckDuckGo, Bing, Wikipedia, Google Scholar, Archive.org, and more
"""

import httpx
import asyncio
import json
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import quote_plus
import logging

logger = logging.getLogger(__name__)


class MultiSearchEngine:
    """Advanced multi-source web search with deep web capabilities"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        # Search APIs
        self.duckduckgo_api = "https://api.duckduckgo.com/"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.archive_api = "https://archive.org/advancedsearch.php"
        
    # ================== DUCKDUCKGO SEARCH ==================
    
    async def search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo (privacy-focused)"""
        try:
            # DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = await self.http_client.get(self.duckduckgo_api, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Abstract (main answer)
            if data.get('Abstract'):
                results.append({
                    'source': 'DuckDuckGo',
                    'title': data.get('Heading', 'Answer'),
                    'snippet': data['Abstract'],
                    'url': data.get('AbstractURL', ''),
                    'type': 'instant_answer'
                })
            
            # Related topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'source': 'DuckDuckGo',
                        'title': topic.get('Text', '').split(' - ')[0],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', ''),
                        'type': 'related'
                    })
            
            logger.info(f"🦆 DuckDuckGo found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    # ================== BING SEARCH (via HTML scraping) ==================
    
    async def search_bing(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using Bing"""
        try:
            url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
            
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            # Parse HTML (basic extraction)
            text = response.text
            results = []
            
            # Simple regex-like extraction (production should use BeautifulSoup)
            import re
            
            # Find titles and snippets
            pattern = r'<h2><a[^>]*href="([^"]+)"[^>]*>([^<]+)</a></h2>.*?<p[^>]*>([^<]+)</p>'
            matches = re.findall(pattern, text, re.DOTALL)
            
            for url, title, snippet in matches[:max_results]:
                if not url.startswith('http'):
                    continue
                results.append({
                    'source': 'Bing',
                    'title': title.strip(),
                    'snippet': snippet.strip()[:300],
                    'url': url,
                    'type': 'web'
                })
            
            logger.info(f"🔍 Bing found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []
    
    # ================== WIKIPEDIA SEARCH ==================
    
    async def search_wikipedia(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Wikipedia for encyclopedic content"""
        try:
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': max_results,
                'format': 'json',
                'redirects': 'resolve'
            }
            
            response = await self.http_client.get(self.wikipedia_api, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if len(data) >= 4:
                titles = data[1]
                descriptions = data[2]
                urls = data[3]
                
                for i in range(len(titles)):
                    results.append({
                        'source': 'Wikipedia',
                        'title': titles[i],
                        'snippet': descriptions[i],
                        'url': urls[i],
                        'type': 'encyclopedia'
                    })
            
            logger.info(f"📚 Wikipedia found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    # ================== ARCHIVE.ORG SEARCH (Deep Web) ==================
    
    async def search_archive(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Internet Archive (historical/deep web content)"""
        try:
            params = {
                'q': query,
                'fl[]': ['identifier', 'title', 'description'],
                'rows': max_results,
                'output': 'json'
            }
            
            response = await self.http_client.get(self.archive_api, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for doc in data.get('response', {}).get('docs', []):
                results.append({
                    'source': 'Archive.org',
                    'title': doc.get('title', 'Untitled'),
                    'snippet': doc.get('description', 'No description')[:300],
                    'url': f"https://archive.org/details/{doc.get('identifier', '')}",
                    'type': 'archive'
                })
            
            logger.info(f"📦 Archive.org found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Archive.org search error: {e}")
            return []
    
    # ================== GOOGLE SCHOLAR (Academic) ==================
    
    async def search_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Google Scholar for academic papers"""
        try:
            # Using Semantic Scholar API (free alternative)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,url,authors,year,citationCount'
            }
            
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for paper in data.get('data', []):
                authors = ', '.join([a.get('name', '') for a in paper.get('authors', [])[:3]])
                results.append({
                    'source': 'Scholar',
                    'title': paper.get('title', 'Untitled'),
                    'snippet': paper.get('abstract', 'No abstract')[:300],
                    'url': paper.get('url', ''),
                    'metadata': f"Authors: {authors} | Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citationCount', 0)}",
                    'type': 'academic'
                })
            
            logger.info(f"🎓 Scholar found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Scholar search error: {e}")
            return []
    
    # ================== NEWS SEARCH ==================
    
    async def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for recent news articles"""
        try:
            # Using NewsAPI alternative - Bing News
            url = f"https://www.bing.com/news/search?q={quote_plus(query)}&count={max_results}"
            
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            # Basic extraction
            results = []
            text = response.text
            
            import re
            pattern = r'<a[^>]*href="([^"]+)"[^>]*class="title"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, text)
            
            for url, title in matches[:max_results]:
                results.append({
                    'source': 'Bing News',
                    'title': title.strip(),
                    'snippet': 'Recent news article',
                    'url': url,
                    'type': 'news'
                })
            
            logger.info(f"📰 Found {len(results)} news results")
            return results
        
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    # ================== UNIFIED SEARCH ==================
    
    async def unified_search(
        self,
        query: str,
        sources: List[str] = None,
        max_results_per_source: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Perform unified search across multiple sources
        
        Args:
            query: Search query
            sources: List of sources to search (default: all)
            max_results_per_source: Max results per source
        
        Returns:
            Dict with results from each source
        """
        if sources is None:
            sources = ['duckduckgo', 'wikipedia', 'bing', 'archive', 'scholar', 'news']
        
        tasks = []
        source_map = {}
        
        if 'duckduckgo' in sources:
            tasks.append(self.search_duckduckgo(query, max_results_per_source))
            source_map['duckduckgo'] = len(tasks) - 1
        
        if 'wikipedia' in sources:
            tasks.append(self.search_wikipedia(query, max_results_per_source))
            source_map['wikipedia'] = len(tasks) - 1
        
        if 'bing' in sources:
            tasks.append(self.search_bing(query, max_results_per_source))
            source_map['bing'] = len(tasks) - 1
        
        if 'archive' in sources:
            tasks.append(self.search_archive(query, max_results_per_source))
            source_map['archive'] = len(tasks) - 1
        
        if 'scholar' in sources:
            tasks.append(self.search_scholar(query, max_results_per_source))
            source_map['scholar'] = len(tasks) - 1
        
        if 'news' in sources:
            tasks.append(self.search_news(query, max_results_per_source))
            source_map['news'] = len(tasks) - 1
        
        # Execute all searches concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        organized_results = {}
        for source_name, idx in source_map.items():
            result = results_list[idx]
            if isinstance(result, Exception):
                logger.error(f"Error in {source_name}: {result}")
                organized_results[source_name] = []
            else:
                organized_results[source_name] = result
        
        return organized_results
    
    # ================== SEARCH SUMMARIZATION ==================
    
    def format_search_results(self, results: Dict[str, List[Dict]], max_total: int = 20) -> str:
        """Format search results for AI processing"""
        formatted = []
        total_count = 0
        
        for source, items in results.items():
            if not items or total_count >= max_total:
                continue
            
            formatted.append(f"\n{'='*60}")
            formatted.append(f"SOURCE: {source.upper()}")
            formatted.append(f"{'='*60}\n")
            
            for i, item in enumerate(items, 1):
                if total_count >= max_total:
                    break
                
                formatted.append(f"[{i}] {item['title']}")
                formatted.append(f"    {item['snippet'][:200]}...")
                formatted.append(f"    URL: {item['url']}")
                if 'metadata' in item:
                    formatted.append(f"    {item['metadata']}")
                formatted.append("")
                
                total_count += 1
        
        formatted.append(f"\n{'='*60}")
        formatted.append(f"TOTAL RESULTS: {total_count}")
        formatted.append(f"{'='*60}")
        
        return "\n".join(formatted)
    
    async def close(self):
        """Cleanup"""
        await self.http_client.aclose()
