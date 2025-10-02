"""
Smart AI Telegram Bot - Multi-Provider AI Engine v2
Supports: Cerebras, NVIDIA NIM, and Pollinations.ai
"""

import httpx
import base64
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from openai import OpenAI, AsyncOpenAI
from cerebras.cloud.sdk import Cerebras

from config import (
    AI_CHAIN, IMAGE_MODELS, DEFAULT_IMAGE_MODEL,
    IMAGE_DEFAULTS, API_TIMEOUT, MAX_TOKENS, TOKEN_BUDGET,
    INTENT_KEYWORDS, CEREBRAS_API_KEY, NVIDIA_API_KEY
)


class MultiProviderAI:
    """Multi-provider AI engine with fallback support"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=API_TIMEOUT)
        
        # Initialize Cerebras client
        self.cerebras = Cerebras(api_key=CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None
        
        # Initialize NVIDIA client
        self.nvidia = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        ) if NVIDIA_API_KEY else None
        
        # Fallback to Pollinations
        self.pollinations_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
    
    # ================== PROVIDER DETECTION ==================
    
    def _get_provider(self, model: str) -> str:
        """Detect which provider to use for a model"""
        cerebras_models = [
            "qwen-3-coder-480b",
            "qwen-3-235b-a22b-instruct-2507", 
            "qwen-3-235b-a22b-thinking-2507",
            "llama-4-maverick-17b-128e-instruct",
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
    
    # ================== CEREBRAS CALLS ==================
    
    async def _call_cerebras(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7
    ) -> str:
        """Call Cerebras API"""
        if not self.cerebras:
            raise Exception("Cerebras API key not configured")
        
        try:
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
            
            stream = self.cerebras.chat.completions.create(
                messages=[
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
            
            return response_text.strip()
        
        except Exception as e:
            raise Exception(f"Cerebras API error: {str(e)}")
    
    # ================== NVIDIA CALLS ==================
    
    async def _call_nvidia(
        self,
        prompt: str,
        model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Call NVIDIA NIM API"""
        if not self.nvidia:
            raise Exception("NVIDIA API key not configured")
        
        try:
            completion = self.nvidia.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
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
            
            return response_text.strip()
        
        except Exception as e:
            raise Exception(f"NVIDIA API error: {str(e)}")
    
    # ================== NVIDIA OCR ==================
    
    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using NVIDIA OCR"""
        if not NVIDIA_API_KEY:
            raise Exception("NVIDIA API key required for OCR")
        
        try:
            image_b64 = base64.b64encode(image_data).decode()
            
            if len(image_b64) >= 180_000:
                raise Exception("Image too large for OCR (max ~135KB)")
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json"
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
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            # Extract text from OCR result
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0].get("content", "")
            return ""
        
        except Exception as e:
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
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = await self.http_client.post(
                f"{self.pollinations_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            raise Exception(f"Pollinations API error: {str(e)}")
    
    # ================== UNIFIED CALL METHOD ==================
    
    async def call_model(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7
    ) -> str:
        """Unified method to call any model with automatic provider selection"""
        provider = self._get_provider(model)
        
        try:
            if provider == "cerebras":
                return await self._call_cerebras(prompt, model, max_tokens, temperature)
            elif provider == "nvidia":
                return await self._call_nvidia(prompt, model, max_tokens)
            else:
                return await self._call_pollinations(prompt, model, max_tokens)
        
        except Exception as e:
            # Fallback chain
            if provider != "pollinations":
                try:
                    return await self._call_pollinations(prompt, "openai", max_tokens)
                except:
                    pass
            raise e
    
    # ================== INTENT DETECTION ==================
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Smart intent detection"""
        msg_lower = message.lower().strip()
        
        # Image generation
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["image"]) and not has_photo:
            return {
                "type": "image",
                "model": DEFAULT_IMAGE_MODEL,
                "prompt": message,
                "confidence": 0.95
            }
        
        # OCR request (if photo attached)
        if has_photo and any(kw in msg_lower for kw in ["baca", "extract", "ocr", "text"]):
            return {
                "type": "ocr",
                "model": "nvidia-ocr",
                "prompt": message,
                "confidence": 0.95
            }
        
        # Coding request
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["code"]):
            return {
                "type": "code",
                "model": AI_CHAIN["coding"],
                "prompt": message,
                "confidence": 0.9
            }
        
        # Deep thinking request
        if any(kw in msg_lower for kw in ["analisa", "analysis", "thinking", "reasoning"]):
            return {
                "type": "thinking",
                "model": AI_CHAIN["reasoning"],
                "prompt": message,
                "confidence": 0.85
            }
        
        # Default: General chat
        return {
            "type": "general",
            "model": AI_CHAIN["general"],
            "prompt": message,
            "confidence": 0.7
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
        
        return f"{self.image_url}/prompt/{encoded_prompt}?{param_str}"
    
    # ================== MAIN PROCESSING ==================
    
    async def process_query(
        self,
        query: str,
        history: List[Dict] = None,
        has_photo: bool = False,
        photo_data: bytes = None
    ) -> Dict:
        """Process query with multi-provider support"""
        history = history or []
        
        try:
            # Detect intent
            intent = await self.detect_intent(query, has_photo)
            
            # Handle image generation
            if intent["type"] == "image":
                url = await self.generate_image(intent["prompt"], intent["model"])
                return {
                    "type": "image",
                    "content": url,
                    "success": True
                }
            
            # Handle OCR
            if intent["type"] == "ocr" and photo_data:
                extracted_text = await self.extract_text_from_image(photo_data)
                
                if extracted_text:
                    # Now process the extracted text with AI
                    analysis_prompt = f"Text extracted from image:\n\n{extracted_text}\n\nUser question: {query}\n\nPlease analyze and respond:"
                    response = await self.call_model(
                        analysis_prompt,
                        AI_CHAIN["general"]
                    )
                    
                    return {
                        "type": "ocr",
                        "content": f"**Extracted Text:**\n{extracted_text}\n\n**Analysis:**\n{response}",
                        "success": True
                    }
                else:
                    return {
                        "type": "error",
                        "content": "Could not extract text from image",
                        "success": False
                    }
            
            # Handle code generation
            if intent["type"] == "code":
                code_response = await self.call_model(
                    f"Generate production-ready code for: {query}",
                    intent["model"]
                )
                
                return {
                    "type": "code",
                    "content": code_response,
                    "success": True,
                    "intent": intent
                }
            
            # Handle thinking/reasoning
            if intent["type"] == "thinking":
                thinking_response = await self.call_model(
                    query,
                    intent["model"],
                    max_tokens=8000
                )
                
                return {
                    "type": "thinking",
                    "content": thinking_response,
                    "success": True,
                    "intent": intent
                }
            
            # Default: General response
            response = await self.call_model(query, intent["model"])
            
            return {
                "type": intent["type"],
                "content": response,
                "success": True,
                "intent": intent
            }
        
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error: {str(e)}",
                "success": False
            }
    
    async def close(self):
        """Cleanup"""
        await self.http_client.aclose()
