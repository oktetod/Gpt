# 📄 ai_engine.py v2 (versi lengkap)

import httpx
import random
from typing import List, Dict
from urllib.parse import quote
from config import TEXT_MODELS, AUDIO_VOICES


class PollinationsAI:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self.base_url = "https://text.pollinations.ai"
        self.image_url = "https://image.pollinations.ai"
    
    async def detect_intent(self, message: str, has_photo: bool = False) -> Dict:
        """Deteksi cerdas untuk memahami keinginan pengguna."""
        message_lower = message.lower().strip() if message else ""

        # Permintaan Audio
        if any(word in message_lower for word in ['suara', 'bicara', 'bilang', 'katakan', 'ngomong', 'speak', 'voice', 'audio', 'tts', 'say']):
            voice = 'alloy'  # Default voice
            for v in AUDIO_VOICES:
                if v in message_lower:
                    voice = v
                    break
            return {'type': 'audio', 'voice': voice, 'text': message}

        # Permintaan Generasi Gambar
        image_keywords = ['gambar', 'buatkan', 'buat', 'lukis', 'generate', 'create', 'make', 'draw', 'image']
        if any(word in message_lower for word in image_keywords) and not has_photo:
            model = 'flux'  # Default model
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

        # Permintaan Transformasi Gambar (jika foto dikirim)
        if has_photo:
            transform_keywords = ['ubah', 'edit', 'transformasi', 'ganti', 'konversi', 'jadikan', 'jadi', 'transform', 'change', 'convert', 'into', 'style']
            if message and any(word in message_lower for word in transform_keywords):
                return {'type': 'image_transform', 'model': 'kontext', 'prompt': message}
            else:
                # Default: analisis gambar
                return {'type': 'chat', 'model': 'gemini'}

        # Permintaan Coding/Pemrograman
        if any(word in message_lower for word in ['kode', 'coding', 'program', 'fungsi', 'skrip', 'debug', 'bug', 'code', 'function', 'script']):
            return {'type': 'chat', 'model': 'qwen-coder'}

        # Permintaan Penalaran/Analisis
        if any(word in message_lower for word in ['analisa', 'analisis', 'pikirkan', 'jelaskan', 'reason', 'logika', 'berpikir', 'think', 'explain']):
            return {'type': 'chat', 'model': 'deepseek-r1'}

        # Default: Gunakan model utama
        return {'type': 'chat', 'model': 'gpt-oss'}

    async def generate_text(self, prompt: str, model: str = 'gpt-oss') -> str:
        """Generate teks menggunakan model tertentu."""
        try:
            payload = {"messages": [{"role": "user", "content": prompt}], "model": model}
            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error] Gagal menghasilkan teks: {str(e)}"

    async def generate_image(self, prompt: str, model: str = 'flux') -> str:
        """Generate gambar dari prompt."""
        try:
            img_prompt = quote(prompt)
            return f"{self.image_url}/prompt/{img_prompt}?model={model}"
        except Exception as e:
            return f"[Error] Gagal membuat URL gambar: {str(e)}"

    async def close(self):
        """Tutup klien HTTP."""
        await self.client.aclose()
