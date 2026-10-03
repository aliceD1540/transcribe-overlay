import threading
from typing import Optional
import numpy as np
import sherpa_onnx

from src.core.config import ASRConfig


class MoonshineASREngine:
    """ASR Engine using Sherpa-ONNX with Moonshine model (Japanese-optimized)."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self.recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        self.is_loading = False
        self.is_ready = False
        self._lock = threading.Lock()

    def load_model_async(self, on_loaded_callback: Optional[callable] = None):
        """Load model in background thread."""
        thread = threading.Thread(
            target=self._load_model, args=(on_loaded_callback,), daemon=True
        )
        thread.start()

    def _load_model(self, on_loaded_callback: Optional[callable] = None):
        """Load Moonshine model using Sherpa-ONNX."""
        with self._lock:
            if self.is_ready:
                return
            self.is_loading = True
            print(
                f"[ASR] Loading Moonshine model via Sherpa-ONNX (device: {self.config.device})..."
            )

            device = self.config.device
            if device == "auto":
                device = "cpu"  # Sherpa-ONNX defaults

            try:
                # Moonshine model configuration for Sherpa-ONNX
                # Using distilled-small model optimized for Japanese
                recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/moonshine-jp-en-distilled-small.onnx",
                        decoder="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/moonshine-jp-en-distilled-small-decoder.onnx",
                        joiner="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/moonshine-jp-en-distilled-small-joiner.onnx",
                    ),
                    num_threads=4,
                    providers=["CPUExecutionProvider"],
                    sample_rate=16000,
                    feature_extractor_type="linear_spectrogram",
                    enable_endpoint_detection=True,
                )

                self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    recognizer_config
                )
                self.is_ready = True
                self.is_loading = False
                print("[ASR] Moonshine model loaded successfully via Sherpa-ONNX.")
                if on_loaded_callback:
                    on_loaded_callback(True, "Loaded Moonshine")
            except Exception as e:
                print(f"[ASR] Error loading Moonshine model: {e}")
                self.is_loading = False
                self.is_ready = False
                if on_loaded_callback:
                    on_loaded_callback(False, str(e))

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 1D float32 audio array (16kHz)."""
        if not self.is_ready or self.recognizer is None:
            return ""

        # Normalize audio array if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Require at least 0.25s of audio
        if len(audio) < 4000:
            return ""

        try:
            # Create stream for online recognition
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio)

            # Run recognition
            while self.recognizer.is_ready(stream):
                self.recognizer.decode(stream)

            # Get result
            result = self.recognizer.get_result(stream)
            if result.text:
                return result.text.strip()
            return ""

        except Exception as e:
            print(f"[ASR] Error during Moonshine transcription: {e}")
            return ""

    def get_result(self, stream) -> str:
        """Get final transcription result from stream."""
        result = self.recognizer.get_result(stream)
        return result.text.strip() if result.text else ""
