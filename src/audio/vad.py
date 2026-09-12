from collections import deque
import os
import urllib.request
import numpy as np
import onnxruntime as ort

from src.core.config import VADConfig


SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")


class SileroVADDetector:
    """Voice Activity Detector using Silero VAD (ONNX Runtime)."""

    def __init__(self, config: VADConfig, model_path: str = DEFAULT_MODEL_PATH):
        self.config = config
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)

        self.is_speaking = False
        self.silence_samples = 0
        self.min_silence_samples = int(self.config.min_silence_duration_ms * 16)  # 16 samples per ms at 16kHz
        self.speech_buffer = []

        # Pre-speech padding ring buffer (e.g. 200ms = 3200 samples = ~6 chunks of 512)
        padding_chunks = max(1, int((self.config.speech_pad_ms * 16) / 512))
        self.pre_speech_buffer = deque(maxlen=padding_chunks)

        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.model_path):
            print(f"[VAD] Downloading Silero VAD ONNX model to {self.model_path}...")
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                urllib.request.urlretrieve(SILERO_VAD_URL, self.model_path)
                print("[VAD] Silero VAD ONNX model downloaded successfully.")
            except Exception as e:
                print(f"[VAD] Error downloading Silero VAD model: {e}")

        if os.path.exists(self.model_path):
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
                input_names = [inp.name for inp in self.session.get_inputs()]
                self.uses_state = "state" in input_names
                if self.uses_state:
                    self._state = np.zeros((2, 1, 128), dtype=np.float32)
                else:
                    self._h = np.zeros((2, 1, 64), dtype=np.float32)
                    self._c = np.zeros((2, 1, 64), dtype=np.float32)
                print(f"[VAD] Silero VAD ONNX Session initialized (uses_state={self.uses_state}).")
            except Exception as e:
                print(f"[VAD] Failed to initialize ONNX Runtime session for VAD: {e}")
                self.session = None

    def reset_state(self):
        if hasattr(self, "uses_state") and self.uses_state:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.is_speaking = False
        self.silence_samples = 0
        self.speech_buffer.clear()
        self.pre_speech_buffer.clear()

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """Calculate speech probability for an audio chunk at 16kHz with Noise Gate and AGC."""
        rms = float(np.sqrt(np.mean(chunk**2)))

        # Noise Gate: ignore background ambient/static noise below RMS 0.002
        if rms < 0.002:
            return 0.0

        # Automatic Gain Control (AGC) for quiet USB microphones
        target_rms = 0.04
        gain = 1.0
        if 0.002 <= rms < target_rms:
            gain = min(6.0, target_rms / rms)

        boosted_chunk = np.clip(chunk * gain, -1.0, 1.0)
        onnx_prob = 0.0

        if self.session is not None and len(boosted_chunk) == 512:
            try:
                audio_input = np.expand_dims(boosted_chunk.astype(np.float32), axis=0)
                if self.uses_state:
                    inputs = {
                        "input": audio_input,
                        "sr": self.sr,
                        "state": self._state,
                    }
                    out, self._state = self.session.run(None, inputs)
                else:
                    inputs = {
                        "input": audio_input,
                        "sr": self.sr,
                        "h": self._h,
                        "c": self._c,
                    }
                    out, self._h, self._c = self.session.run(None, inputs)
                onnx_prob = float(out[0][0])
            except Exception as e:
                print(f"[VAD] ONNX inference error, resetting state: {e}")
                self.reset_state()

        # Hybrid energy-based speech probability
        energy_prob = min(1.0, float((rms - 0.002) / 0.003)) if rms >= 0.0025 else 0.0
        return max(onnx_prob, energy_prob)

    def process_chunk(self, chunk: np.ndarray):
        """Process incoming chunk and return state tuple:
        (is_speaking: bool, speech_started: bool, speech_ended: bool, full_utterance: Optional[np.ndarray])
        """
        prob = self.get_speech_prob(chunk)
        speech_started = False
        speech_ended = False
        full_utterance = None

        if prob >= self.config.threshold:
            if not self.is_speaking:
                self.is_speaking = True
                speech_started = True
                # Prepend buffered audio right before speech start so initial consonants are preserved
                self.speech_buffer.extend(list(self.pre_speech_buffer))
                self.pre_speech_buffer.clear()
            self.silence_samples = 0
            self.speech_buffer.append(chunk)
        else:
            if self.is_speaking:
                self.speech_buffer.append(chunk)
                self.silence_samples += len(chunk)
                if self.silence_samples >= self.min_silence_samples:
                    self.is_speaking = False
                    speech_ended = True
                    if self.speech_buffer:
                        full_utterance = np.concatenate(self.speech_buffer)
                    self.speech_buffer.clear()
                    self.pre_speech_buffer.clear()
                    self.silence_samples = 0
            else:
                self.pre_speech_buffer.append(chunk)

        return self.is_speaking, speech_started, speech_ended, full_utterance
