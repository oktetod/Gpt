"""
Smart AI Telegram Bot - AI Chain Engine
Implements 9-step AI chaining for optimal responses
"""

import httpx
import json
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from config import (
    AI_CHAIN, AI_ENDPOINTS, IMAGE_MODELS, DEFAULT_IMAGE_MODEL,
    IMAGE_DEFAULTS, API_TIMEOUT, MAX_TOKENS, CHUNK_SIZE, TOKEN_BUDGET,
    INTENT_KEYWORDS
)


class AIChainEngine:
    """Production-ready AI engine with intelligent chaining"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=API_TIMEOUT)
        self.text_url = AI_ENDPOINTS["text"]
        self.image_url = AI_ENDPOINTS["image"]
        self.history_cache = {}
    
    # ================== STEP 1: PRE-PROCESSING ==================
    def preprocess_query(self, text: str) -> Dict:
        """Clean and analyze query"""
        clean_text = text.strip()
        token_estimate = len(clean_text.split()) * 1.3  # Rough estimate
        
        return {
            "original": text,
            "clean": clean_text,
            "tokens": int(token_estimate),
            "length": len(clean_text)
        }
    
    # ================== STEP 2: INTENT DETECTION ==================
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Smart intent detection without menus"""
        msg_lower = message.lower().strip()
        
        # Image generation request
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["image"]) and not has_photo:
            model = self._detect_image_style(msg_lower)
            return {
                "type": "image",
                "model": model,
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
        
        # Analysis/Research request
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["analysis"]):
            return {
                "type": "analysis",
                "model": AI_CHAIN["reasoning"],
                "prompt": message,
                "confidence": 0.85
            }
        
        # Task/Creation request
        if any(kw in msg_lower for kw in INTENT_KEYWORDS["task"]):
            return {
                "type": "task",
                "model": AI_CHAIN["reasoning"],
                "prompt": message,
                "confidence": 0.8
            }
        
        # Default: General chat
        return {
            "type": "general",
            "model": AI_CHAIN["general"],
            "prompt": message,
            "confidence": 0.7
        }
    
    def _detect_image_style(self, text: str) -> str:
        """Detect desired image style from text"""
        if any(w in text for w in ["realistis", "nyata", "foto", "realistic", "photo"]):
            return "flux-realism"
        elif any(w in text for w in ["anime", "kartun", "manga", "cartoon"]):
            return "flux-anime"
        elif any(w in text for w in ["3d", "render", "cgi"]):
            return "flux-3d"
        elif any(w in text for w in ["gelap", "seram", "dark", "gothic", "horror"]):
            return "any-dark"
        elif any(w in text for w in ["cepat", "fast", "quick"]):
            return "turbo"
        return DEFAULT_IMAGE_MODEL
    
    # ================== STEP 3: RERANKING ==================
    async def rerank_context(self, query: str, contexts: List[str]) -> List[str]:
        """Use Qwen-3-235B-Thinking for context reranking"""
        if not contexts or len(contexts) <= 1:
            return contexts
        
        prompt = f"""Rank the following contexts by relevance to the query.
Return only the indices in order of relevance (e.g., 2,0,1).

Query: {query}

Contexts:
{chr(10).join(f'{i}. {c[:200]}...' for i, c in enumerate(contexts))}

Ranking:"""
        
        try:
            response = await self._call_model(prompt, AI_CHAIN["reranking"], max_tokens=100)
            indices = [int(x.strip()) for x in response.split(',') if x.strip().isdigit()]
            return [contexts[i] for i in indices if i < len(contexts)]
        except:
            return contexts
    
    # ================== STEP 4: CONTEXT ASSEMBLY ==================
    def assemble_context(self, history: List[Dict], current: str) -> str:
        """Chunk and assemble context within token budget"""
        context_parts = []
        token_count = 0
        
        # Add recent history (reversed for chronological order)
        for msg in reversed(history[-5:]):  # Last 5 messages
            chunk = f"{msg['role']}: {msg['content']}\n"
            chunk_tokens = len(chunk.split()) * 1.3
            
            if token_count + chunk_tokens < TOKEN_BUDGET * 0.3:  # 30% for history
                context_parts.append(chunk)
                token_count += chunk_tokens
            else:
                break
        
        # Add current query
        context_parts.append(f"user: {current}")
        
        return "\n".join(reversed(context_parts))
    
    # ================== STEP 5: REASONING/GENERATION ==================
    async def generate_reasoning(self, prompt: str, intent: Dict) -> str:
        """Main reasoning using Nemotron-Ultra RAG"""
        model = intent.get("model", AI_CHAIN["general"])
        
        system_prompt = self._get_system_prompt(intent["type"])
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        return await self._call_model(full_prompt, model)
    
    def _get_system_prompt(self, intent_type: str) -> str:
        """Get appropriate system prompt based on intent"""
        prompts = {
            "code": """You are an expert programmer. Provide clean, efficient, well-documented code.
Always include explanations and best practices.""",
            
            "analysis": """You are a research analyst. Provide thorough, well-structured analysis.
Break down complex topics into understandable parts.""",
            
            "task": """You are a helpful assistant. Complete tasks efficiently and thoroughly.
Provide step-by-step guidance when needed.""",
            
            "general": """You are a knowledgeable AI assistant. Provide accurate, helpful responses.
Be concise but complete."""
        }
        return prompts.get(intent_type, prompts["general"])
    
    # ================== STEP 6: CODE GENERATION ==================
    async def generate_code(self, task: str, language: str = "python") -> Dict:
        """Specialized code generation with Qwen-3-Coder-480B"""
        prompt = f"""Generate production-ready {language} code for:
{task}

Requirements:
- Clean, readable code
- Proper error handling
- Type hints (if applicable)
- Documentation
- No placeholders

Code:"""
        
        try:
            code = await self._call_model(prompt, AI_CHAIN["coding"])
            return {
                "code": self._extract_code(code),
                "language": language,
                "success": True
            }
        except Exception as e:
            # Fallback to alternative model
            try:
                code = await self._call_model(prompt, AI_CHAIN["coding_fallback"])
                return {
                    "code": self._extract_code(code),
                    "language": language,
                    "success": True
                }
            except:
                return {"code": None, "success": False, "error": str(e)}
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown fences"""
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else text.strip()
    
    # ================== STEP 7: VALIDATION ==================
    def validate_code(self, code: str, language: str) -> Dict:
        """Basic code validation"""
        issues = []
        
        # Check for common issues
        if not code or len(code.strip()) < 10:
            issues.append("Code too short or empty")
        
        if language == "python":
            # Basic Python checks
            if "import" not in code and "def" not in code and "class" not in code:
                issues.append("No functions or imports found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20)
        }
    
    # ================== STEP 8: POST-PROCESSING ==================
    def post_process(self, text: str) -> str:
        """Clean and format final output"""
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove markdown artifacts
        text = re.sub(r'^\s*```\w*\s*$', '', text, flags=re.MULTILINE)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    # ================== STEP 9: SUMMARIZATION ==================
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Create concise summary using GPT-OSS-120B"""
        if len(text) < max_length:
            return text
        
        prompt = f"""Summarize this in one clear sentence:

{text[:1000]}

Summary:"""
        
        try:
            return await self._call_model(prompt, AI_CHAIN["summarization"], max_tokens=50)
        except:
            return text[:max_length] + "..."
    
    # ================== IMAGE GENERATION ==================
    async def generate_image(
        self,
        prompt: str,
        model: str = None,
        negative_prompt: str = None,
        **kwargs
    ) -> str:
        """Generate image with positive and negative prompts"""
        model = model or DEFAULT_IMAGE_MODEL
        
        # Enhance prompt
        enhanced_prompt = self._enhance_image_prompt(prompt)
        
        # Default negative prompt if not provided
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
        
        # Build URL parameters
        params = {
            **IMAGE_DEFAULTS,
            "model": model,
            "negative": negative_prompt,
            **kwargs
        }
        
        # Encode prompt
        encoded_prompt = quote(enhanced_prompt)
        
        # Build URL
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.image_url}/prompt/{encoded_prompt}?{param_str}"
        
        return url
    
    def _enhance_image_prompt(self, prompt: str) -> str:
        """Enhance image prompt for better results"""
        enhancements = [
            "high quality",
            "detailed",
            "professional"
        ]
        
        # Add enhancements if not already present
        prompt_lower = prompt.lower()
        additions = [e for e in enhancements if e not in prompt_lower]
        
        if additions:
            return f"{prompt}, {', '.join(additions)}"
        return prompt
    
    # ================== CORE API CALL ==================
    async def _call_model(
        self,
        prompt: str,
        model: str,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Make API call to Pollinations.ai"""
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = await self.client.post(
                f"{self.text_url}/chat/completions",
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        except httpx.TimeoutException:
            raise Exception("Request timeout - please try again")
        except httpx.HTTPStatusError as e:
            raise Exception(f"API error: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    # ================== FULL CHAIN EXECUTION ==================
    async def process_query(
        self,
        query: str,
        history: List[Dict] = None,
        has_photo: bool = False
    ) -> Dict:
        """Execute full AI chain"""
        history = history or []
        
        try:
            # Step 1: Preprocess
            processed = self.preprocess_query(query)
            
            # Step 2: Detect intent
            intent = await self.detect_intent(processed["clean"], has_photo)
            
            # Handle image generation separately
            if intent["type"] == "image":
                url = await self.generate_image(intent["prompt"], intent["model"])
                return {
                    "type": "image",
                    "content": url,
                    "success": True
                }
            
            # Step 4: Assemble context
            context = self.assemble_context(history, processed["clean"])
            
            # Step 5 & 6: Generate response (with code if needed)
            if intent["type"] == "code":
                code_result = await self.generate_code(processed["clean"])
                if code_result["success"]:
                    explanation = await self.generate_reasoning(
                        f"Explain this code:\n{code_result['code']}",
                        {"type": "general", "model": AI_CHAIN["general"]}
                    )
                    content = f"{explanation}\n\n```{code_result['language']}\n{code_result['code']}\n```"
                else:
                    content = await self.generate_reasoning(processed["clean"], intent)
            else:
                content = await self.generate_reasoning(processed["clean"], intent)
            
            # Step 8: Post-process
            content = self.post_process(content)
            
            return {
                "type": intent["type"],
                "content": content,
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
        """Cleanup resources"""
        await self.client.aclose()
