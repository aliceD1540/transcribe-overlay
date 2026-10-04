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
        self.debug_logging = True  # Enable detailed logging for debugging

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
        text = result.text.strip()
        text_length = len(text)
        
        # Reject empty text or only punctuation
        if not text or text.isspace():
            return False
        
        # Reject text that is only punctuation/brackets
        if all(c in '。？！、，，""""【】《》「」『』<<>>~・' for c in text):
            return False
        
        # Check log probabilities - main confidence indicator
        if hasattr(result, 'ys_log_probs') and len(result.ys_log_probs) > 0:
            # Average log probability - higher values = higher confidence
            avg_log_prob = sum(result.ys_log_probs) / len(result.ys_log_probs)
            min_log_prob = min(result.ys_log_probs)
            max_log_prob = max(result.ys_log_probs)
            
            # Much stricter threshold for very short results
            if text_length == 1:
                # Single character: extremely strict
                threshold = -5.0
            elif text_length <= 2:
                # 2 characters: very strict
                threshold = -4.5
            elif text_length <= 4:
                threshold = -4.0
            else:
                # Standard threshold for longer text
                threshold = -3.5
            
            if self.debug_logging:
                print(f"[ASR:DEBUG] Text: '{text}' | Len: {text_length} | "
                      f"AvgLogProb: {avg_log_prob:.3f} | MinLogProb: {min_log_prob:.3f} | "
                      f"MaxLogProb: {max_log_prob:.3f} | Threshold: {threshold:.3f} | "
                      f"Pass: {avg_log_prob >= threshold}")
            
            if avg_log_prob < threshold:
                return False
        else:
            # No log probs available - might indicate empty result
            # But also happens with valid short results
            # Only reject if text is extremely short AND no log probs
            if text_length <= 1 and not result.ys_log_probs:
                return False
        
        return True

    def _detect_repetition_pattern(self, text: str) -> bool:
        """
        Detect repetitive patterns that indicate misrecognition.
        Returns False if repetition detected (should be rejected).
        """
        text = text.strip()
        if not text or len(text) < 4:
            return True  # Not enough text to be repetitive
        
        # Split by common delimiters and check for repetition
        import re
        # Split by punctuation and common connectors
        # Note: exclude 'の' from delimiters as it's part of many words
        parts = re.split(r'[、。，，をで]', text)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) < 3:
            return True  # Not enough parts to be repetitive
        
        # Check if first few parts are identical or very similar
        # This catches "この私を、この私を、この私を..."
        if len(parts) >= 3:
            if parts[0] == parts[1] == parts[2]:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Repetition detected: '{text}'")
                return False  # Reject
        
        return True  # No repetition detected, accept

    def _filter_extreme_noise_patterns(self, text: str) -> bool:
        """Filter extreme noise patterns including empty brackets and keyboard noise."""
        # Patterns that are clearly not speech but keyboard/noise artifacts
        extreme_noise_patterns = [
            # Empty/minimal brackets
            "「」", "『』", "【】", "《》", "<<>>",
            "。", "？", "！", "、", "，",  # Standalone punctuation
            ".", "?", "!", ",",  # Latin punctuation
            
            # Single character variations (無音-keyboard click noise)
            "あ", "い", "う", "え", "お", "ん",
            "あ。", "い。", "う。", "え。", "お。", "ん。",
            "あ？", "い？", "う？", "え？", "お？", "ん？",
            "あ！", "い！", "う！", "え！", "お！", "ん！",
            "あ、", "い、", "う、", "え、", "お、", "ん、",
            
            # Double character keyboard noise patterns
            "あっ", "えっ", "おっ", "うっ", "いっ",
            "あっ。", "えっ。", "おっ。", "うっ。", "いっ。",
            "あっ？", "えっ？", "おっ？", "うっ？", "いっ？",
            "あっ！", "えっ！", "おっ！", "うっ！", "いっ！",
            "あっ、", "えっ、", "おっ、", "うっ、", "いっ、",
            
            # Common filler/interjection noise (from mechanical vibration)
            "うん", "うん。", "うん？", "うん！", "うん、",
            "えっと",  # Often follows keyboard click
            
            # Doubled sounds (keyboard glitch)
            "ああ", "いい", "うう", "ええ", "おお", "んん",
            "ああ。", "いい。", "うう。", "ええ。", "おお。", "んん。",
        ]
        
        return text not in extreme_noise_patterns

    def _is_likely_keyboard_noise(self, text: str, audio_length_ms: float) -> bool:
        """
        Detect keyboard noise based on characteristics.
        Returns True if likely keyboard noise (should be rejected).
        
        Keyboard noise characteristics:
        - Very short duration (typically < 300ms for single key press, < 600ms for double click)
        - Limited vocabulary (single sounds like 'ん', 'あ', etc., or mechanical tones)
        - Often followed by punctuation
        - Very low confidence from ASR model
        """
        text = text.strip()
        if not text:
            return False
        
        # Pattern-based detection for very short audio (mechanical noise)
        if audio_length_ms < 300:
            keyboard_patterns = [
                # Single vowel characters
                "あ", "い", "う", "え", "お", "ん",
                # Doubled vowels (mechanical resonance)
                "ああ", "いい", "うう", "ええ", "おお",
                # With punctuation (all combinations)
                "あ。", "い。", "う。", "え。", "お。", "ん。",
                "あ？", "い？", "う？", "え？", "お？", "ん？",
                "あ！", "い！", "う！", "え！", "お！", "ん！",
                "あ、", "い、", "う、", "え、", "お、", "ん、",
                "あ.", "い.", "う.", "え.", "お.", "ん.",
            ]
            
            if text in keyboard_patterns:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Keyboard noise (single key): '{text}' (duration: {audio_length_ms:.0f}ms)")
                return True
        
        # Extended pattern-based detection for short audio (double-click or sustained key)
        if audio_length_ms < 600:
            extended_patterns = [
                # Double character (key press + release)
                "あっ", "いっ", "うっ", "えっ", "おっ",
                "ああ", "いい", "うう", "ええ", "おお",
                # With punctuation
                "あっ。", "いっ。", "うっ。", "えっ。", "おっ。",
                "あっ？", "いっ？", "うっ？", "えっ？", "おっ？",
                "あっ！", "いっ！", "うっ！", "えっ！", "おっ！",
                "あっ、", "いっ、", "うっ、", "えっ、", "おっ、",
                # Common interjections from mechanical noise
                "うん", "うん。", "うん？", "うん！", "うん、",
                "んぅ", "んぅ。",
            ]
            
            if text in extended_patterns:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Keyboard noise (double key): '{text}' (duration: {audio_length_ms:.0f}ms)")
                return True
        
        return False

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
            # Calculate audio duration in milliseconds
            audio_length_ms = (len(audio) / 16000.0) * 1000
            
            # Moonshine has limits on audio length processing
            # Split long audio into chunks to avoid ONNX runtime errors
            # Maximum chunk: ~10 seconds to be safe (Moonshine may have decoder limits)
            max_chunk_samples = int(16000 * 10)  # 10 seconds
            
            if len(audio) > max_chunk_samples:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Audio too long ({audio_length_ms:.0f}ms), processing in chunks")
                
                # Process in overlapping chunks with generous overlap
                results = []
                chunk_size = max_chunk_samples
                overlap = int(16000 * 1.0)  # 1.0s overlap for better continuity
                
                start = 0
                chunk_num = 0
                while start < len(audio):
                    end = min(start + chunk_size, len(audio))
                    chunk = audio[start:end]
                    chunk_duration_ms = ((end - start) / 16000.0) * 1000
                    
                    if len(chunk) >= 4000:
                        try:
                            chunk_num += 1
                            if self.debug_logging:
                                print(f"[ASR:DEBUG] Processing chunk {chunk_num}: {start}-{end} ({chunk_duration_ms:.0f}ms)")
                            
                            result_text = self._process_audio_chunk(chunk, chunk_duration_ms)
                            if result_text:
                                results.append(result_text)
                        except Exception as e:
                            if self.debug_logging:
                                print(f"[ASR:DEBUG] Error processing chunk {chunk_num} at sample {start}: {e}")
                    
                    start += chunk_size - overlap
                
                final_result = " ".join(results) if results else ""
                if self.debug_logging and len(results) > 0:
                    print(f"[ASR:DEBUG] Concatenated {len(results)} chunks: '{final_result}'")
                return final_result
            else:
                # Process single chunk
                return self._process_audio_chunk(audio, audio_length_ms)

        except Exception as e:
            print(f"[ASR] Error during Moonshine transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _process_audio_chunk(self, audio: np.ndarray, audio_length_ms: float) -> str:
        """Process a single audio chunk."""
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
                
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Raw recognition result: '{text}'")
                
                # Apply confidence checks
                if not self._is_confidence_high_enough(result):
                    return ""
                
                # Detect keyboard noise based on duration and pattern
                if self._is_likely_keyboard_noise(text, audio_length_ms):
                    return ""
                
                # Detect and filter repetition patterns
                if not self._detect_repetition_pattern(text):
                    return ""
                
                # Filter extreme noise patterns
                if not self._filter_extreme_noise_patterns(text):
                    return ""
                
                if self.debug_logging:
                    print(f"[ASR:DEBUG] Final result after filtering: '{text}'")
                
                return text
            else:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] No recognition result (empty text). Samples: {len(audio)}, Duration: {audio_length_ms:.0f}ms")
                return ""

        except RuntimeError as e:
            # ONNX Runtime errors - log but don't crash
            error_str = str(e)
            if "Attempting to broadcast" in error_str or "Non-zero status" in error_str:
                if self.debug_logging:
                    print(f"[ASR:DEBUG] ONNX Runtime shape error (recoverable): {e}")
                # Return empty result instead of propagating error
                return ""
            else:
                # Other runtime errors - log and propagate
                raise
        except Exception as e:
            if self.debug_logging:
                print(f"[ASR:DEBUG] Error in chunk processing: {type(e).__name__}: {e}")
            # Return empty on error instead of crashing
            return ""
