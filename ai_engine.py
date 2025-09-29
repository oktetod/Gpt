# /proyek_bot/ai_engine.py

import httpx
import random
from typing import List, Dict
from urllib.parse import quote
from config import TEXT_MODELS, AUDIO_VOICES

class PollinationsAI:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0) # Timeout diperpanjang
        self.base_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Deteksi cerdas untuk memahami keinginan pengguna."""
        message_lower = message.lower()
        
        # Permintaan Audio
        if any(word in message_lower for word in ['suara', 'bicara', 'bilang', 'katakan', 'ngomong', 'speak', 'voice', 'audio', 'tts', 'say']):
            voice = 'alloy' # Default
            for v in AUDIO_VOICES:
                if v in message_lower:
                    voice = v
                    break
            return {'type': 'audio', 'voice': voice, 'text': message}
        
        # Permintaan Generasi Gambar
        image_keywords = ['gambar', 'buatkan', 'buat', 'lukis', 'generate', 'create', 'make', 'draw', 'image']
        if any(word in message_lower for word in image_keywords) and not has_photo:
            model = 'flux' # Default
            if any(word in message_lower for word in ['realistis', 'nyata', 'foto', 'realistic', 'photo', 'real']):
                model = 'flux-realism'
            elif any(word in message_lower for word in ['anime', 'kartun', 'manga', 'cartoon']):
                model = 'flux-anime'
            elif any(word in message_lower for word in ['3d', 'render', 'cgi']):
                model = 'flux-3d'
            elif any(word in message_lower for word in ['gelap', 'seram', 'dark', 'gothic', 'horror']):
                model = 'any-dark'
            elif any(word in message_lower for word in ['cepat', 'fast', 'quick']):
                model = 'turbo'
            
            return {'type': 'image', 'model': model, 'prompt': message}
        
        # Permintaan Transformasi Gambar (jika pengguna mengirim foto)
        if has_photo:
            transform_keywords = ['ubah', 'edit', 'transformasi', 'ganti', 'konversi', 'jadikan', 'jadi', 'transform', 'change', 'convert', 'into', 'style']
            if not message or not any(word in message_lower for word in transform_keywords):
                # Jika tidak ada teks atau kata kunci, defaultnya adalah analisis gambar
                return {'type': 'chat', 'model': 'gemini'}
            
            return {'type': 'image_transform', 'model': 'kontext', 'prompt': message}
        
        # Permintaan Coding/Pemrograman
        if any(word in message_lower for word in ['kode', 'coding', 'program', 'fungsi', 'skrip', 'debug', 'bug', 'code', 'function', 'script']):
            return {'type': 'chat', 'model': 'qwen-coder'}
        
        # Permintaan Penalaran/Analisis
        if any(word in message_lower for word in ['analisa', 'analisis', 'pikirkan', 'jelaskan', 'mengapa', 'kenapa', 'analyze', 'think', 'reason', 'explain', 'why']):
            return {'type': 'chat', 'model': 'deepseek-r1'}
        
        # Permintaan Pencarian/Informasi Terkini
        if any(word in message_lower for word in ['cari', 'berita', 'terbaru', 'informasi', 'search', 'find', 'latest', 'news']):
            return {'type': 'chat', 'model': 'gemini-search'}
        
        # Default: Chat biasa menggunakan GPT-5
        return {'type': 'chat', 'model': 'gpt-5'}
    
    async def chat(self, messages: List[Dict], model: str = "gpt-5") -> str:
        """Mengirim permintaan chat ke AI."""
        try:
            model_name = TEXT_MODELS.get(model, TEXT_MODELS['gpt-5'])
            
            response = await self.client.post(
                f"{self.base_url}/openai",
                json={"model": model_name, "messages": messages, "temperature": 0.7, "max_tokens": 4096},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "Maaf, terjadi kesalahan saat memproses permintaan Anda.")
        except httpx.HTTPStatusError as e:
            return f"❌ Error: Gagal terhubung ke server AI ({e.response.status_code}). Coba beberapa saat lagi."
        except Exception as e:
            return f"❌ Error: Terjadi kesalahan internal: {str(e)}"
    
    def generate_image_url(self, prompt: str, model: str = "flux", width: int = 1024, height: int = 1024, seed: int = None) -> str:
        """Membuat URL untuk generasi gambar."""
        if seed is None:
            seed = random.randint(1, 1000000)
        
        encoded_prompt = quote(prompt)
        return f"{self.image_url}/prompt/{encoded_prompt}?model={model}&width={width}&height={height}&seed={seed}&nologo=true"
    
    def transform_image_url(self, prompt: str, image_url: str, model: str = "kontext") -> str:
        """Membuat URL untuk transformasi gambar."""
        encoded_prompt = quote(prompt)
        encoded_image = quote(image_url)
        return f"{self.image_url}/prompt/{encoded_prompt}?model={model}&image={encoded_image}&nologo=true"
    
    def generate_audio_url(self, text: str, voice: str = "alloy") -> str:
        """Membuat URL untuk generasi audio."""
        encoded_text = quote(text)
        return f"{self.base_url}/{encoded_text}?model=openai-audio&voice={voice}"
    
    async def enhance_prompt(self, prompt: str) -> str:
        """Memperkaya prompt gambar agar lebih deskriptif."""
        if len(prompt.split()) > 25:
            return prompt
        
        enhance_request = f"Tingkatkan dan detailkan prompt gambar ini agar lebih artistik dan deskriptif (maksimal 50 kata, dalam bahasa Inggris): {prompt}"
        messages = [{"role": "user", "content": enhance_request}]
        enhanced = await self.chat(messages, "gpt-5-mini")
        return enhanced.strip().replace('"', '') # Hapus tanda kutip dari hasil
    
    async def close(self):
        """Menutup koneksi client httpx."""
        await self.client.aclose()
