"""
Smart AI Telegram Bot - Enhanced Multi-Provider AI Engine v3.1 FIXED
Features: Web Search, RAG, Vision, Multiple API Keys with Load Balancing
Fixed: Error handling, timeouts, validation, logging
"""

import httpx
import base64
import os
import random
from typing import Dict, List, Optional
from urllib.parse import quote
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
import logging

from config import (
    AI_CHAIN, IMAGE_MODELS, DEFAULT_IMAGE_MODEL,
    IMAGE_DEFAULTS, API_TIMEOUT, MAX_TOKENS, TOKEN_BUDGET,
    INTENT_KEYWORDS, CEREBRAS_API_KEYS, NVIDIA_API_KEY,
    WEB_SEARCH_ENABLED, WEB_SEARCH_SOURCES, WEB_SEARCH_MAX_RESULTS_PER_SOURCE,
    WEB_SEARCH_AUTO_TRIGGER_KEYWORDS
)
from web_search import MultiSearchEngine

logger = logging.getLogger(__name__)


class MultiProviderAI:
    """Enhanced Multi-provider AI engine with Web Search, RAG & Vision support"""
    
    def __init__(self):
        # Initialize HTTP client with proper timeout
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # Initialize Multiple Cerebras clients with load balancing
        self.cerebras_clients = []
        self.cerebras_current_index = 0
        if CEREBRAS_API_KEYS:
            for key in CEREBRAS_API_KEYS:
                try:
                    self.cerebras_clients.append(Cerebras(api_key=key))
                    logger.info(f"✓ Cerebras client initialized with key: {key[:8]}...")
                except Exception as e:
                    logger.error(f"Failed to initialize Cerebras client: {e}")
        
        if not self.cerebras_clients:
            logger.warning("⚠️ No Cerebras API keys configured!")
        
        # Initialize NVIDIA client
        self.nvidia = None
        if NVIDIA_API_KEY:
            try:
                self.nvidia = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=NVIDIA_API_KEY,
                    timeout=60.0
                )
                logger.info(f"✓ NVIDIA client initialized with key: {NVIDIA_API_KEY[:8]}...")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA client: {e}")
        else:
            logger.warning("⚠️ NVIDIA API key not configured")
        
        # Initialize Web Search Engine
        self.search_engine = None
        if WEB_SEARCH_ENABLED:
            try:
                self.search_engine = MultiSearchEngine()
                logger.info("✓ Web Search Engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize search engine: {e}")
        
        # Fallback to Pollinations
        self.pollinations_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
        
        # GPT-5 Branding
        self.gpt5_signature = "\n\n---\n💫 *Powered by GPT-5* | Fine-tuned by @durov9369"
        
        logger.info("🚀 MultiProviderAI initialized successfully")
    
    # ================== CEREBRAS API ROTATION ==================
    
    def _get_cerebras_client(self) -> Cerebras:
        """Get next Cerebras client using round-robin load balancing"""
        if not self.cerebras_clients:
            raise Exception("No Cerebras API keys configured")
        
        client = self.cerebras_clients[self.cerebras_current_index]
        self.cerebras_current_index = (self.cerebras_current_index + 1) % len(self.cerebras_clients)
        
        logger.debug(f"Using Cerebras client index: {self.cerebras_current_index}")
        return client
    
    # ================== PROVIDER DETECTION ==================
    
    def _get_provider(self, model: str) -> str:
        """Detect which provider to use for a model"""
        cerebras_models = [
            "qwen-3-coder-480b",
            "qwen-3-235b-thinking",
            "llama-4-maverick",
            "gpt-oss-120b"
        ]
        
        nvidia_models = [
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nemotron-ultra-rag"
        ]
        
        if model in cerebras_models:
            return "cerebras"
        elif model in nvidia_models or "nemotron" in model.lower():
            return "nvidia"
        else:
            return "pollinations"
    
    # ================== WEB SEARCH INTEGRATION ==================
    
    async def perform_web_search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = None
    ) -> Dict:
        """Perform web search across multiple sources"""
        if not self.search_engine:
            logger.warning("Web search not enabled")
            return {"success": False, "error": "Web search not enabled"}
        
        try:
            sources = sources or WEB_SEARCH_SOURCES
            max_results = max_results or WEB_SEARCH_MAX_RESULTS_PER_SOURCE
            
            logger.info(f"🔍 Performing web search: {query}")
            
            # Perform unified search
            results = await self.search_engine.unified_search(
                query=query,
                sources=sources,
                max_results_per_source=max_results
            )
            
            # Count total results
            total_results = sum(len(items) for items in results.values())
            
            if total_results == 0:
                logger.warning(f"No search results found for: {query}")
                return {
                    "success": False,
                    "error": "No results found"
                }
            
            # Format results for AI processing
            formatted_text = self.search_engine.format_search_results(results)
            
            logger.info(f"✓ Found {total_results} results from {len(results)} sources")
            
            return {
                "success": True,
                "results": results,
                "formatted": formatted_text,
                "total_results": total_results,
                "sources_used": list(results.keys())
            }
        
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _should_use_web_search(self, query: str) -> bool:
        """Determine if query should trigger web search"""
        query_lower = query.lower()
        
        # Check for explicit search keywords
        for keyword in WEB_SEARCH_AUTO_TRIGGER_KEYWORDS:
            if keyword in query_lower:
                return True
        
        # Check for question patterns
        question_patterns = [
            'apa itu', 'what is', 'siapa', 'who is', 'kapan', 'when',
            'dimana', 'where', 'bagaimana', 'how', 'mengapa', 'why',
            'berapa', 'how many', 'how much'
        ]
        
        for pattern in question_patterns:
            if query_lower.startswith(pattern):
                return True
        
        return False
    
    # ================== CEREBRAS CALLS (WITH ROTATION) ==================
    
    async def _call_cerebras(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7,
        retry_count: int = 0
    ) -> str:
        """Call Cerebras API with automatic key rotation on failure"""
        if not self.cerebras_clients:
            raise Exception("Cerebras API keys not configured")
        
        try:
            # Get next client (round-robin)
            cerebras = self._get_cerebras_client()
            
            # Map config model names to actual Cerebras model names
            model_mapping = {
                "qwen-3-235b-thinking": "qwen-3-235b-a22b-thinking-2507",
                "qwen-3-coder-480b": "qwen-3-coder-480b",
                "llama-4-maverick": "llama-4-maverick-17b-128e-instruct",
                "gpt-oss-120b": "gpt-oss-120b"
            }
            
            actual_model = model_mapping.get(model, model)
            
            # Adjust params based on model
            if "thinking" in actual_model:
                max_tokens = min(max_tokens, 65536)
                temperature = 0.6
                top_p = 0.95
            elif "coder" in actual_model:
                max_tokens = min(max_tokens, 40000)
                temperature = 0.7
                top_p = 0.8
            else:
                top_p = 0.9
            
            logger.info(f"📡 Calling Cerebras: {actual_model}")
            
            stream = cerebras.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are GPT-5, an advanced AI model fine-tuned by @durov9369. You provide accurate, detailed, and helpful responses with web search capabilities."},
                    {"role": "user", "content": prompt}
                ],
                model=actual_model,
                stream=True,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Collect streamed response
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            if not response_text.strip():
                raise Exception("Empty response from Cerebras")
            
            logger.info(f"✓ Cerebras response: {len(response_text)} chars")
            return response_text.strip()
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Cerebras API error (attempt {retry_count + 1}): {error_msg}")
            
            # Retry with next API key if available
            if retry_count < len(self.cerebras_clients) - 1:
                logger.info(f"Retrying with next API key...")
                return await self._call_cerebras(
                    prompt, model, max_tokens, temperature, retry_count + 1
                )
            
            # All keys failed
            raise Exception(f"Cerebras API error (all {len(self.cerebras_clients)} keys failed): {error_msg}")
    
    # ================== NVIDIA CALLS (RAG ENABLED) ==================
    
    async def _call_nvidia_rag(
        self,
        prompt: str,
        model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        max_tokens: int = MAX_TOKENS,
        context: str = None
    ) -> str:
        """Call NVIDIA NIM API with RAG support"""
        if not self.nvidia:
            raise Exception("NVIDIA API key not configured")
        
        try:
            messages = [
                {"role": "system", "content": "You are GPT-5, an advanced AI model fine-tuned by @durov9369. You provide accurate, detailed, and helpful responses with context awareness and web search capabilities."}
            ]
            
            # Add context if provided (RAG)
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Previous conversation context:\n{context}"
                })
            
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"📡 Calling NVIDIA: {model}")
            
            completion = self.nvidia.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect streamed response
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            if not response_text.strip():
                raise Exception("Empty response from NVIDIA")
            
            logger.info(f"✓ NVIDIA response: {len(response_text)} chars")
            return response_text.strip()
        
        except Exception as e:
            logger.error(f"NVIDIA API error: {e}", exc_info=True)
            raise Exception(f"NVIDIA API error: {str(e)}")
    
    # ================== NVIDIA VISION (FIXED) ==================

    async def analyze_image_with_nemotron(
        self, 
        image_data: bytes, 
        question: str = "Describe this image in detail"
    ) -> str:
        """Analyze image using NVIDIA Nemotron Vision - FIXED VERSION"""
        if not NVIDIA_API_KEY:
            logger.error("NVIDIA API key not configured")
            raise Exception("NVIDIA API key required for vision analysis")
        
        try:
            # Validate image size (5MB limit)
            image_size_mb = len(image_data) / (1024 * 1024)
            if image_size_mb > 5:
                logger.warning(f"Image too large: {image_size_mb:.2f}MB")
                raise Exception(f"Image too large ({image_size_mb:.1f}MB). Please send a smaller image (max 5MB)")
            
            logger.info(f"🖼️ Analyzing image: {image_size_mb:.2f}MB, question: {question[:50]}...")
            
            image_b64 = base64.b64encode(image_data).decode()
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are GPT-5, an advanced AI with vision capabilities. You analyze images accurately and provide detailed, helpful descriptions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.6,
                "top_p": 0.9
            }
            
            response = await self.http_client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90.0  # Increased timeout for vision
            )
            
            # Detailed error handling
            if response.status_code != 200:
                error_body = response.text
                logger.error(f"NVIDIA Vision API Error {response.status_code}: {error_body}")
                
                # Parse specific error messages
                if response.status_code == 401:
                    raise Exception("❌ NVIDIA API key invalid or expired. Please check your API key")
                elif response.status_code == 403:
                    raise Exception("❌ NVIDIA API access denied. Check your API key permissions")
                elif response.status_code == 429:
                    raise Exception("⏱️ NVIDIA API rate limit exceeded. Please try again in a few minutes")
                elif response.status_code == 413:
                    raise Exception("❌ Image too large for NVIDIA API. Try a smaller image")
                elif response.status_code == 400:
                    raise Exception("❌ Invalid request to NVIDIA API. Check image format")
                elif response.status_code >= 500:
                    raise Exception(f"❌ NVIDIA API server error ({response.status_code}). Try again later")
                else:
                    raise Exception(f"❌ NVIDIA API error ({response.status_code}). Try again later")
            
            response.raise_for_status()
            result = response.json()
            
            # Validate response structure
            if "choices" not in result or len(result["choices"]) == 0:
                logger.error(f"Invalid NVIDIA response structure: {result}")
                raise Exception("Invalid response from NVIDIA API")
            
            content = result["choices"][0]["message"]["content"]
            
            if not content or len(content.strip()) == 0:
                logger.error("Empty content from NVIDIA API")
                raise Exception("Empty response from NVIDIA API")
            
            logger.info(f"✓ Vision analysis completed: {len(content)} chars")
            return content
        
        except httpx.TimeoutException:
            logger.error("NVIDIA Vision API timeout")
            raise Exception("⏱️ Vision analysis timed out. Try a smaller image")
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in vision analysis: {e}")
            raise Exception(f"🌐 Network error: {str(e)}")
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Vision analysis error: {error_msg}")
            
            # Don't fallback to OCR if it's an API key issue
            if "API key" in error_msg or "401" in error_msg or "403" in error_msg:
                raise
            
            # Try OCR fallback for other errors
            if "too large" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.info("Attempting OCR fallback...")
                try:
                    ocr_result = await self.extract_text_from_image(image_data)
                    return f"⚠️ (Vision analysis unavailable, showing extracted text instead)\n\n{ocr_result}"
                except Exception as ocr_e:
                    logger.error(f"OCR fallback also failed: {ocr_e}")
            
            raise Exception(f"Vision analysis failed: {error_msg}")
    
    # ================== NVIDIA OCR (FIXED) ==================
    
    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using NVIDIA OCR - FIXED VERSION"""
        if not NVIDIA_API_KEY:
            logger.error("NVIDIA API key not configured")
            raise Exception("NVIDIA API key required for OCR")
        
        try:
            # Validate size
            image_size_kb = len(image_data) / 1024
            if image_size_kb > 135:
                logger.warning(f"Image too large for OCR: {image_size_kb:.2f}KB")
                raise Exception(f"Image too large for OCR ({image_size_kb:.0f}KB). Max 135KB")
            
            logger.info(f"📄 Extracting text from image: {image_size_kb:.2f}KB")
            
            image_b64 = base64.b64encode(image_data).decode()
            
            if len(image_b64) >= 180_000:
                raise Exception("Image too large for OCR (base64 > 180KB)")
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": [{
                    "type": "image_url",
                    "url": f"data:image/png;base64,{image_b64}"
                }]
            }
            
            response = await self.http_client.post(
                "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
                headers=headers,
                json=payload,
                timeout=45.0
            )
            
            if response.status_code != 200:
                error_body = response.text
                logger.error(f"NVIDIA OCR Error {response.status_code}: {error_body}")
                
                if response.status_code == 401:
                    raise Exception("NVIDIA API key invalid")
                elif response.status_code == 429:
                    raise Exception("OCR rate limit exceeded")
                elif response.status_code == 413:
                    raise Exception("Image too large for OCR")
                else:
                    raise Exception(f"OCR API error ({response.status_code})")
            
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                content = result["data"][0].get("content", "")
                if not content:
                    raise Exception("No text found in image")
                
                logger.info(f"✓ OCR extracted: {len(content)} chars")
                return content
            
            raise Exception("No text found in image")
        
        except httpx.TimeoutException:
            logger.error("OCR timeout")
            raise Exception("OCR timed out")
        
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise Exception(f"OCR failed: {str(e)}")
    
    # ================== POLLINATIONS FALLBACK ==================
    
    async def _call_pollinations(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Fallback to Pollinations.ai"""
        try:
            logger.info(f"📡 Calling Pollinations: {model}")
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are GPT-5, an advanced AI model fine-tuned by @durov9369. You provide accurate, detailed, and helpful responses."},
                    {"role": "user", "content": prompt}
                ],
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = await self.http_client.post(
                f"{self.pollinations_url}/chat/completions",
                json=payload,
                timeout=45.0
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            logger.info(f"✓ Pollinations response: {len(content)} chars")
            return content
        
        except Exception as e:
            logger.error(f"Pollinations API error: {e}")
            raise Exception(f"Pollinations API error: {str(e)}")
    
    # ================== UNIFIED CALL METHOD ==================
    
    async def call_model(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7,
        context: str = None
    ) -> str:
        """Unified method to call any model with automatic provider selection"""
        provider = self._get_provider(model)
        
        try:
            if provider == "cerebras":
                return await self._call_cerebras(prompt, model, max_tokens, temperature)
            elif provider == "nvidia":
                return await self._call_nvidia_rag(prompt, model, max_tokens, context)
            else:
                return await self._call_pollinations(prompt, model, max_tokens)
        
        except Exception as e:
            logger.error(f"Primary provider ({provider}) failed: {e}")
            
            # Fallback chain
            if provider != "pollinations":
                try:
                    logger.info("Trying Pollinations fallback...")
                    return await self._call_pollinations(prompt, "openai", max_tokens)
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {fallback_e}")
            
            raise e
    
    # ================== INTENT DETECTION (ENHANCED) ==================
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Enhanced intent detection with web search support"""
        msg_lower = message.lower().strip()
        
        # Image analysis (if photo attached)
        if has_photo:
            if any(kw in msg_lower for kw in ["baca", "extract", "ocr", "text", "tulisan", "scan"]):
                return {
                    "type": "ocr",
                    "model": "nvidia-ocr",
                    "prompt": message,
                    "confidence": 0.95,
                    "use_web_search": False
                }
            elif any(kw in msg_lower for kw in ["gambar", "buat gambar", "generate", "draw", "lukis", "bikin gambar"]):
                return {
                    "type": "image",
                    "model": DEFAULT_IMAGE_MODEL,
                    "prompt": message,
                    "confidence": 0.9,
                    "use_web_search": False
                }
            else:
                return {
                    "type": "vision",
                    "model": AI_CHAIN["ultra"],
                    "prompt": message or "Analyze this image in detail",
                    "confidence": 0.9,
                    "use_web_search": False
                }
        
        # Image generation (without photo)
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["image"]):
            return {
                "type": "image",
                "model": DEFAULT_IMAGE_MODEL,
                "prompt": message,
                "confidence": 0.95,
                "use_web_search": False
            }
        
        # Web search detection
        use_web_search = self._should_use_web_search(message)
        
        # Explicit web search request
        if any(kw in msg_lower for kw in ["cari", "search", "google", "bing"]):
            return {
                "type": "web_search",
                "model": AI_CHAIN["web_summary"],
                "prompt": message,
                "confidence": 0.95,
                "use_web_search": True
            }
        
        # Coding request
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["code"]):
            return {
                "type": "code",
                "model": AI_CHAIN["coding"],
                "prompt": message,
                "confidence": 0.9,
                "use_web_search": False
            }
        
        # Deep thinking request
        if any(kw in msg_lower for kw in ["analisa", "analysis", "thinking", "reasoning", "mendalam", "riset", "research"]):
            return {
                "type": "thinking",
                "model": AI_CHAIN["reasoning"],
                "prompt": message,
                "confidence": 0.85,
                "use_web_search": use_web_search
            }
        
        # Questions about AI/models/GPT-5
        if any(kw in msg_lower for kw in ["model", "ai", "gpt", "llm", "kamu siapa", "siapa kamu", "apa itu gpt", "tentang gpt", "gpt-5", "who are you"]):
            return {
                "type": "about",
                "model": AI_CHAIN["general"],
                "prompt": message,
                "confidence": 0.8,
                "use_web_search": False
            }
        
        # Default: General chat with optional web search
        return {
            "type": "general",
            "model": AI_CHAIN["ultra"],
            "prompt": message,
            "confidence": 0.7,
            "use_web_search": use_web_search
        }
    
    # ================== IMAGE GENERATION ==================
    
    async def generate_image(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> str:
        """Generate image URL"""
        model = model or DEFAULT_IMAGE_MODEL
        
        params = {
            **IMAGE_DEFAULTS,
            "model": model,
            **kwargs
        }
        
        encoded_prompt = quote(prompt)
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        
        image_url = f"{self.image_url}/prompt/{encoded_prompt}?{param_str}"
        logger.info(f"🎨 Generated image URL: {image_url[:100]}...")
        
        return image_url
    
    # ================== MAIN PROCESSING (WITH WEB SEARCH) ==================
    
    async def process_query(
        self,
        query: str,
        history: List[Dict] = None,
        has_photo: bool = False,
        photo_data: bytes = None
    ) -> Dict:
        """Enhanced query processing with web search support"""
        history = history or []
        
        try:
            # Detect intent
            intent = await self.detect_intent(query, has_photo)
            logger.info(f"🎯 Intent detected: {intent['type']} (confidence: {intent['confidence']}, web_search: {intent.get('use_web_search')})")
            
            # Handle image generation
            if intent["type"] == "image":
                url = await self.generate_image(intent["prompt"], intent["model"])
                return {
                    "type": "image",
                    "content": url,
                    "success": True,
                    "intent": intent
                }
            
            # Handle vision analysis
            if intent["type"] == "vision" and photo_data:
                analysis = await self.analyze_image_with_nemotron(
                    photo_data, 
                    query or "Analyze this image in detail"
                )
                
                return {
                    "type": "vision",
                    "content": f"**🔍 Vision Analysis:**\n\n{analysis}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent
                }
            
            # Handle OCR
            if intent["type"] == "ocr" and photo_data:
                extracted_text = await self.extract_text_from_image(photo_data)
                
                analysis_prompt = f"Here is text extracted from an image:\n\n{extracted_text}\n\nUser question: {query}\n\nProvide a helpful analysis or answer:"
                response = await self.call_model(
                    analysis_prompt,
                    AI_CHAIN["ultra"],
                    context=extracted_text
                )
                
                return {
                    "type": "ocr",
                    "content": f"**📄 Extracted Text:**\n```\n{extracted_text}\n```\n\n**💡 Analysis:**\n{response}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent
                }
            
            # Handle questions about AI/model/GPT-5
            if intent["type"] == "about":
                about_response = """🤖 **GPT-5 Advanced AI + Web Search**

Saya adalah **GPT-5**, model AI generasi terbaru yang dikembangkan dan di-fine-tune oleh **@durov9369**.

**🔬 Arsitektur:**
• Multi-provider AI chaining (Cerebras + NVIDIA NIM)
• **Real-Time Web Search** (6 search engines)
• RAG (Retrieval Augmented Generation)
• Vision analysis dengan Nemotron Ultra 253B
• Code generation dengan Qwen-3 Coder 480B
• Multiple API keys dengan load balancing

**🌐 Search Engines:**
• DuckDuckGo (Privacy-focused search)
• Bing (Web search)
• Wikipedia (Encyclopedia)
• Archive.org (Deep web/historical)
• Google Scholar (Academic papers)
• Bing News (Real-time news)

**💪 Kemampuan:**
• Deep reasoning hingga 65K tokens
• Multi-modal processing (text + vision)
• Real-time web search & information gathering
• Academic research capabilities
• OCR dan image analysis
• Production-ready code generation
• Context-aware conversations

**🎯 Specialized Models:**
• **Nemotron Ultra 253B** - RAG & Vision analysis
• **Qwen-3 235B** - Deep thinking & web summarization
• **Qwen-3 Coder 480B** - Code generation
• **Llama-4 Maverick** - Fast responses
• **Flux** - Image generation

**⚡ Performance:**
• Context window: 40K-65K tokens
• Response time: <5 seconds (with search)
• Multi-language support
• Accuracy: 95%+
• API redundancy: Multiple keys for stability

Developed with ❤️ by @durov9369"""
                
                return {
                    "type": "about",
                    "content": about_response,
                    "success": True,
                    "intent": intent
                }
            
            # ================== WEB SEARCH INTEGRATION ==================
            
            web_search_results = None
            if intent.get("use_web_search") and WEB_SEARCH_ENABLED:
                logger.info("🌐 Performing web search...")
                search_result = await self.perform_web_search(query)
                if search_result["success"]:
                    web_search_results = search_result
                    logger.info(f"✓ Web search completed: {search_result['total_results']} results")
            
            # Build context from history
            context = None
            if history:
                context = "\n".join([
                    f"{msg['role']}: {msg['content'][:300]}"
                    for msg in history[-5:]
                ])
            
            # Build enhanced prompt with web search results
            if web_search_results:
                enhanced_prompt = f"""User Query: {query}

WEB SEARCH RESULTS:
{web_search_results['formatted']}

Based on the above search results from {web_search_results['total_results']} sources ({', '.join(web_search_results['sources_used'])}), provide a comprehensive, accurate answer to the user's query. 

Instructions:
1. Synthesize information from multiple sources
2. Cite specific sources when making claims
3. Prioritize recent and authoritative information
4. If sources conflict, mention the disagreement
5. Be concise but thorough

Your response:"""
            else:
                enhanced_prompt = query
            
            # Handle code generation
            if intent["type"] == "code":
                code_response = await self.call_model(
                    f"Generate production-ready code for: {enhanced_prompt}",
                    intent["model"]
                )
                
                return {
                    "type": "code",
                    "content": f"{code_response}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent,
                    "web_search_used": web_search_results is not None
                }
            
            # Handle thinking/reasoning
            if intent["type"] == "thinking":
                thinking_response = await self.call_model(
                    enhanced_prompt,
                    intent["model"],
                    max_tokens=8000,
                    context=context
                )
                
                prefix = "🌐 **[With Web Search]**\n\n" if web_search_results else ""
                
                return {
                    "type": "thinking",
                    "content": f"{prefix}{thinking_response}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent,
                    "web_search_used": web_search_results is not None
                }
            
            # Default: General response with optional web search
            response = await self.call_model(
                enhanced_prompt, 
                intent["model"],
                context=context
            )
            
            prefix = "🌐 **[With Web Search]**\n\n" if web_search_results else ""
            
            return {
                "type": intent["type"],
                "content": f"{prefix}{response}{self.gpt5_signature}",
                "success": True,
                "intent": intent,
                "web_search_used": web_search_results is not None
            }
        
        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            return {
                "type": "error",
                "content": f"Error: {str(e)}",
                "success": False
            }
    
    async def close(self):
        """Cleanup resources"""
        logger.info("🔒 Closing AI engine resources...")
        try:
            await self.http_client.aclose()
            if self.search_engine:
                await self.search_engine.close()
            logger.info("✓ Resources closed successfully")
        except Exception as e:
            logger.error(f"Error closing resources: {e}")
