import json
import re
import threading
from typing import Callable, List, Tuple, Optional
import requests

from src.core.config import TranslatorConfig


class OllamaTranslator:
    """Async Ollama client for text translation."""

    def __init__(self, config: TranslatorConfig):
        self.config = config

    def check_connection(self) -> Tuple[bool, List[str]]:
        """Check if Ollama server is reachable and fetch available model list."""
        url = f"{self.config.ollama_url.rstrip('/')}/api/tags"
        try:
            resp = requests.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return True, models
        except Exception as e:
            print(f"[Ollama] Connection check failed: {e}")
        return False, []

    def translate_async(self, text: str, callback: Callable[[str, bool], None]):
        """Request translation in a background thread.
        Callback signature: callback(translated_text: str, success: bool)
        """
        if not text or not self.config.enabled:
            return

        thread = threading.Thread(
            target=self._translate_worker,
            args=(text, callback),
            daemon=True,
        )
        thread.start()

    def _translate_worker(self, text: str, callback: Callable[[str, bool], None]):
        url = f"{self.config.ollama_url.rstrip('/')}/api/generate"
        prompt = f"{self.config.system_prompt}\n\nJapanese: {text}\nEnglish:"

        model_name = self.config.model
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_sec)
            if resp.status_code == 200:
                result = resp.json().get("response", "").strip()
                result = self._clean_translation(result)
                callback(result, True)
                return
            elif resp.status_code == 404:
                print(f"[Ollama] Model '{model_name}' not found (404). Checking available local models...")
                ok, available_models = self.check_connection()
                if ok and available_models:
                    # Filter for text models (skip image-turbo or non-text models if possible)
                    text_models = [m for m in available_models if "image" not in m.lower()]
                    fallback_model = text_models[0] if text_models else available_models[0]
                    print(f"[Ollama] Falling back to available model: '{fallback_model}'")
                    payload["model"] = fallback_model
                    resp2 = requests.post(url, json=payload, timeout=self.config.timeout_sec)
                    if resp2.status_code == 200:
                        result = resp2.json().get("response", "").strip()
                        result = self._clean_translation(result)
                        callback(result, True)
                        return

            print(f"[Ollama] API returned status code {resp.status_code}: {resp.text}")
            callback("", False)
        except Exception as e:
            print(f"[Ollama] Translation request error: {e}")
            callback("", False)

    @staticmethod
    def _clean_translation(raw: str) -> str:
        """Strip preamble and markdown quotes if present."""
        cleaned = raw.strip()
        # Remove surrounded quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        # Strip common preambles
        cleaned = re.sub(r'^(Translation|English):\s*', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned
