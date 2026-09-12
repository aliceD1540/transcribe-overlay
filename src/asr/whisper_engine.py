import threading
from typing import Optional, Tuple
import numpy as np
from faster_whisper import WhisperModel

from src.core.config import ASRConfig


class WhisperASREngine:
    """ASR Engine wrapping faster-whisper with async model loading and fallback options."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self.model: Optional[WhisperModel] = None
        self.is_loading = False
        self.is_ready = False
        self._lock = threading.Lock()

    def load_model_async(self, on_loaded_callback: Optional[callable] = None):
        """Load model in background thread."""
        thread = threading.Thread(target=self._load_model, args=(on_loaded_callback,), daemon=True)
        thread.start()

    def _load_model(self, on_loaded_callback: Optional[callable] = None):
        with self._lock:
            if self.is_ready:
                return
            self.is_loading = True
            print(f"[ASR] Loading faster-whisper model '{self.config.model_size}' (device: {self.config.device})...")

            device = self.config.device
            compute_type = self.config.compute_type

            if device == "auto":
                device = "cpu"  # Direct default for compatibility if CUDA absent

            if compute_type == "default":
                compute_type = "int8" if device == "cpu" else "float16"

            try:
                self.model = WhisperModel(
                    model_size_or_path=self.config.model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=None,
                )
                self.is_ready = True
                self.is_loading = False
                print(f"[ASR] Whisper model '{self.config.model_size}' loaded successfully on {device}.")
                if on_loaded_callback:
                    on_loaded_callback(True, f"Loaded {self.config.model_size}")
            except Exception as e:
                print(f"[ASR] Error loading Whisper model '{self.config.model_size}': {e}")
                # Fallback attempt with cpu + float32
                try:
                    print("[ASR] Attempting fallback to cpu + float32...")
                    self.model = WhisperModel(
                        model_size_or_path=self.config.model_size,
                        device="cpu",
                        compute_type="float32",
                    )
                    self.is_ready = True
                    self.is_loading = False
                    print(f"[ASR] Fallback model '{self.config.model_size}' loaded successfully.")
                    if on_loaded_callback:
                        on_loaded_callback(True, f"Loaded {self.config.model_size} (CPU)")
                except Exception as ex:
                    print(f"[ASR] Fallback failed: {ex}")
                    self.is_loading = False
                    self.is_ready = False
                    if on_loaded_callback:
                        on_loaded_callback(False, str(ex))

    def transcribe(self, audio: np.ndarray, beam_size: int = 1, vad_filter: bool = True) -> str:
        """Transcribe 1D float32 audio array (16kHz)."""
        if not self.is_ready or self.model is None:
            return ""

        # Normalize audio array if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Require at least 0.25s of audio
        if len(audio) < 4000:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=400) if vad_filter else None,
                temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
            )
            text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
            full_text = " ".join(text_parts)
            return full_text
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
            return ""
