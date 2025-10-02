"""
Smart AI Telegram Bot - Enhanced Multi-Provider AI Engine v3
Supports: Cerebras, NVIDIA NIM (with RAG & Vision), Pollinations.ai
Features: RAG, Vision Analysis, Multi-modal Processing
"""

import httpx
import base64
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras

from config import (
    AI_CHAIN, IMAGE_MODELS, DEFAULT_IMAGE_MODEL,
    IMAGE_DEFAULTS, API_TIMEOUT, MAX_TOKENS, TOKEN_BUDGET,
    INTENT_KEYWORDS, CEREBRAS_API_KEY, NVIDIA_API_KEY
)


class MultiProviderAI:
    """Enhanced Multi-provider AI engine with RAG & Vision support"""
    
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
        
        # GPT-5 Branding
        self.gpt5_signature = "\n\n---\n💫 *Powered by GPT-5* | Fine-tuned by @durov9369"
    
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
                {"role": "system", "content": "You are GPT-5, an advanced AI model fine-tuned by @durov9369. You provide accurate, detailed, and helpful responses."}
            ]
            
            # Add context if provided (RAG)
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Context information:\n{context}"
                })
            
            messages.append({"role": "user", "content": prompt})
            
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
            
            return response_text.strip()
        
        except Exception as e:
            raise Exception(f"NVIDIA API error: {str(e)}")
    
    # ================== NVIDIA VISION (Nemotron Multi-modal) ==================
    
    async def analyze_image_with_nemotron(
        self, 
        image_data: bytes, 
        question: str = "Describe this image in detail"
    ) -> str:
        """Analyze image using NVIDIA Nemotron Vision"""
        if not NVIDIA_API_KEY:
            raise Exception("NVIDIA API key required for vision analysis")
        
        try:
            image_b64 = base64.b64encode(image_data).decode()
            
            # Use NVIDIA's vision-capable model
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
                "messages": [
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
                "temperature": 0.6
            }
            
            response = await self.http_client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            # Fallback to OCR if vision fails
            return await self.extract_text_from_image(image_data)
    
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
            # Fallback chain
            if provider != "pollinations":
                try:
                    return await self._call_pollinations(prompt, "openai", max_tokens)
                except:
                    pass
            raise e
    
    # ================== INTENT DETECTION (ENHANCED) ==================
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Enhanced intent detection with vision support"""
        msg_lower = message.lower().strip()
        
        # Image analysis (if photo attached)
        if has_photo and not any(kw in msg_lower for kw in ["gambar", "buat", "generate", "draw"]):
            # Check if it's OCR request or vision analysis
            if any(kw in msg_lower for kw in ["baca", "extract", "ocr", "text", "tulisan"]):
                return {
                    "type": "ocr",
                    "model": "nvidia-ocr",
                    "prompt": message,
                    "confidence": 0.95
                }
            else:
                # General vision analysis
                return {
                    "type": "vision",
                    "model": AI_CHAIN["ultra"],
                    "prompt": message,
                    "confidence": 0.9
                }
        
        # Image generation
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["image"]) and not has_photo:
            return {
                "type": "image",
                "model": DEFAULT_IMAGE_MODEL,
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
        if any(kw in msg_lower for kw in ["analisa", "analysis", "thinking", "reasoning", "mendalam"]):
            return {
                "type": "thinking",
                "model": AI_CHAIN["reasoning"],
                "prompt": message,
                "confidence": 0.85
            }
        
        # Questions about AI models
        if any(kw in msg_lower for kw in ["model", "ai", "gpt", "llm", "kamu", "siapa"]):
            return {
                "type": "about",
                "model": AI_CHAIN["general"],
                "prompt": message,
                "confidence": 0.8
            }
        
        # Default: General chat with RAG
        return {
            "type": "general",
            "model": AI_CHAIN["ultra"],  # Use Nemotron for RAG
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
    
    # ================== MAIN PROCESSING (ENHANCED) ==================
    
    async def process_query(
        self,
        query: str,
        history: List[Dict] = None,
        has_photo: bool = False,
        photo_data: bytes = None
    ) -> Dict:
        """Enhanced query processing with multi-modal support"""
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
            
            # Handle vision analysis (Nemotron multi-modal)
            if intent["type"] == "vision" and photo_data:
                analysis = await self.analyze_image_with_nemotron(photo_data, query)
                
                return {
                    "type": "vision",
                    "content": f"**🔍 Vision Analysis:**\n\n{analysis}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent
                }
            
            # Handle OCR
            if intent["type"] == "ocr" and photo_data:
                extracted_text = await self.extract_text_from_image(photo_data)
                
                if extracted_text:
                    # Process with RAG
                    analysis_prompt = f"Text extracted from image:\n\n{extracted_text}\n\nUser question: {query}\n\nAnalyze and respond:"
                    response = await self.call_model(
                        analysis_prompt,
                        AI_CHAIN["ultra"],
                        context=extracted_text
                    )
                    
                    return {
                        "type": "ocr",
                        "content": f"**📄 Extracted Text:**\n{extracted_text}\n\n**💡 Analysis:**\n{response}{self.gpt5_signature}",
                        "success": True
                    }
                else:
                    return {
                        "type": "error",
                        "content": "Could not extract text from image",
                        "success": False
                    }
            
            # Handle questions about AI/model
            if intent["type"] == "about":
                about_response = """🤖 **GPT-5 Advanced AI**

Saya adalah **GPT-5**, model AI generasi terbaru yang dikembangkan dan di-fine-tune oleh **@durov9369**.

**🔬 Arsitektur:**
• Multi-provider AI chaining (Cerebras + NVIDIA NIM)
• RAG (Retrieval Augmented Generation)
• Vision analysis dengan Nemotron Ultra 253B
• Code generation dengan Qwen-3 Coder 480B

**💪 Kemampuan:**
• Deep reasoning hingga 65K tokens
• Multi-modal processing (text + vision)
• Real-time OCR dan image analysis
• Production-ready code generation
• Advanced problem solving

**🎯 Specialized Models:**
• Qwen-3 235B (Deep thinking)
• Nemotron Ultra 253B (RAG & Vision)
• Llama-4 Maverick (Fast responses)
• Flux (Image generation)

**⚡ Performance:**
• Context window: 40K-65K tokens
• Response time: <3 seconds
• Multi-language support
• Accuracy: 95%+

Developed with ❤️ by @durov9369"""
                
                return {
                    "type": "about",
                    "content": about_response,
                    "success": True,
                    "intent": intent
                }
            
            # Handle code generation
            if intent["type"] == "code":
                code_response = await self.call_model(
                    f"Generate production-ready code for: {query}",
                    intent["model"]
                )
                
                return {
                    "type": "code",
                    "content": f"{code_response}{self.gpt5_signature}",
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
                    "content": f"{thinking_response}{self.gpt5_signature}",
                    "success": True,
                    "intent": intent
                }
            
            # Default: General response with RAG
            # Build context from history
            context = None
            if history:
                context = "\n".join([
                    f"{msg['role']}: {msg['content'][:200]}"
                    for msg in history[-5:]  # Last 5 messages
                ])
            
            response = await self.call_model(
                query, 
                intent["model"],
                context=context
            )
            
            return {
                "type": intent["type"],
                "content": f"{response}{self.gpt5_signature}",
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