import threading
from typing import Optional
import numpy as np
import os
from pathlib import Path
import tarfile

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

from src.core.config import ASRConfig


class MoonshineASREngine:
    """ASR Engine using Sherpa-ONNX with Moonshine model (Japanese-optimized)."""

    # Pre-trained Moonshine model URLs (v2 models - quantized versions)
    # Using base/tiny models with merged decoder
    MOONSHINE_MODELS = {
        "tiny": {
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27.tar.bz2",
            "dir_name": "sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27",
        },
        "base": {
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-ja-quantized-2026-02-27.tar.bz2",
            "dir_name": "sherpa-onnx-moonshine-base-ja-quantized-2026-02-27",
        },
    }

    def __init__(self, config: ASRConfig):
        self.config = config
        self.recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self.is_loading = False
        self.is_ready = False
        self._lock = threading.Lock()
        self.model_size = "tiny"  # Use smallest model by default for Japanese

    def load_model_async(self, on_loaded_callback: Optional[callable] = None):
        """Load model in background thread."""
        thread = threading.Thread(
            target=self._load_model, args=(on_loaded_callback,), daemon=True
        )
        thread.start()

    def _download_and_extract_model(self, model_size: str) -> str:
        """Download and extract model files. Returns path to model directory."""
        model_info = self.MOONSHINE_MODELS.get(model_size, self.MOONSHINE_MODELS["tiny"])
        
        model_cache_dir = Path.home() / ".cache" / "sherpa-onnx" / "moonshine"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = model_cache_dir / model_info["dir_name"]
        
        # Check if already extracted
        if model_dir.exists() and (model_dir / "encoder_model.ort").exists():
            print(f"[ASR] Model already cached at {model_dir}")
            return str(model_dir)
        
        # Download
        import urllib.request
        tar_path = model_cache_dir / f"{model_info['dir_name']}.tar.bz2"
        
        if not tar_path.exists():
            print(f"[ASR] Downloading Moonshine model from {model_info['url']}...")
            try:
                urllib.request.urlretrieve(model_info["url"], tar_path)
                print(f"[ASR] Downloaded to {tar_path}")
            except Exception as e:
                print(f"[ASR] Failed to download model: {e}")
                raise
        
        # Extract
        print(f"[ASR] Extracting model...")
        try:
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=model_cache_dir)
            print(f"[ASR] Extracted to {model_dir}")
            # Clean up tar file
            tar_path.unlink()
        except Exception as e:
            print(f"[ASR] Failed to extract model: {e}")
            raise
        
        return str(model_dir)

    def _load_model(self, on_loaded_callback: Optional[callable] = None):
        """Load Moonshine model using Sherpa-ONNX."""
        with self._lock:
            if self.is_ready:
                return
            self.is_loading = True
            print(
                f"[ASR] Loading Moonshine model via Sherpa-ONNX (size: {self.model_size})..."
            )

            if sherpa_onnx is None:
                error_msg = "sherpa_onnx module not installed. Please install: pip install sherpa-onnx"
                print(f"[ASR] {error_msg}")
                self.is_loading = False
                self.is_ready = False
                if on_loaded_callback:
                    on_loaded_callback(False, error_msg)
                return

            try:
                # Download and extract model
                print("[ASR] Preparing model files...")
                model_dir = self._download_and_extract_model(self.model_size)
                
                # Model file paths
                encoder_path = os.path.join(model_dir, "encoder_model.ort")
                decoder_merged_path = os.path.join(model_dir, "decoder_model_merged.ort")
                tokens_path = os.path.join(model_dir, "tokens.txt")
                
                # Verify files exist
                for path, name in [(encoder_path, "encoder"), (decoder_merged_path, "decoder"), (tokens_path, "tokens")]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"{name} not found at {path}")
                
                # Create recognizer using Moonshine model
                device = self.config.device
                if device == "auto":
                    device = "cpu"
                
                print(f"[ASR] Creating Moonshine v2 recognizer (device: {device})...")
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                    encoder=encoder_path,
                    decoder=decoder_merged_path,
                    tokens=tokens_path,
                    debug=False,
                )
                
                self.is_ready = True
                self.is_loading = False
                print("[ASR] Moonshine model loaded successfully via Sherpa-ONNX.")
                if on_loaded_callback:
                    on_loaded_callback(True, "Loaded Moonshine")
            except Exception as e:
                print(f"[ASR] Error loading Moonshine model: {e}")
                import traceback
                traceback.print_exc()
                self.is_loading = False
                self.is_ready = False
                if on_loaded_callback:
                    on_loaded_callback(False, str(e))

    def _is_confidence_high_enough(self, result) -> bool:
        """Check if recognition confidence is high enough to be trusted."""
        # Check log probabilities - main confidence indicator
        if hasattr(result, 'ys_log_probs') and len(result.ys_log_probs) > 0:
            # Average log probability - higher values = higher confidence
            avg_log_prob = sum(result.ys_log_probs) / len(result.ys_log_probs)
            # Log probability threshold (empirically tuned for Moonshine)
            # More negative values indicate lower confidence
            # Very low scores like < -5.0 typically indicate noise/silence
            if avg_log_prob < -3.5:
                return False
        else:
            # No log probs available - might indicate empty result
            # But also happens with valid short results
            # Only reject if text is extremely short AND no log probs
            text = result.text.strip()
            if len(text) <= 1 and not result.ys_log_probs:
                return False
        
        return True

    def _filter_extreme_noise_patterns(self, text: str) -> bool:
        """Filter only extreme noise patterns (single character with punctuation)."""
        # Only filter the most obvious noise: single character + punctuation
        # This is very conservative to avoid filtering actual speech
        extreme_noise_patterns = [
            "あ。",      # Single syllable
            "ん。",      # Single character
            "え。",
            "お。",
            "い。",
            "う。",
        ]
        
        return text not in extreme_noise_patterns

    def transcribe(self, audio: np.ndarray, beam_size: int = 1, vad_filter: bool = True) -> str:
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
            # Create stream for offline recognition
            stream = self.recognizer.create_stream()
            
            # Accept waveform (sample_rate=16000)
            stream.accept_waveform(16000, audio)
            
            # Decode the stream
            self.recognizer.decode_stream(stream)
            
            # Get result
            result = stream.result
            if result.text:
                text = result.text.strip()
                
                # Apply confidence checks
                if not self._is_confidence_high_enough(result):
                    return ""
                
                # Filter extreme noise patterns
                if not self._filter_extreme_noise_patterns(text):
                    return ""
                
                return text
            return ""

        except Exception as e:
            print(f"[ASR] Error during Moonshine transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""
