import queue
import time
from typing import Optional
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.config import AppConfig
from src.audio.capture import AudioCapturer
from src.audio.vad import SileroVADDetector
from src.asr.whisper_engine import WhisperASREngine
from src.translator.ollama_client import OllamaTranslator


class TranscriptionPipeline(QThread):
    """Background worker thread coordinating Audio -> VAD -> ASR -> Translation -> PySide6 Signals."""

    asr_updated = Signal(str, bool)  # (text, is_final)
    translation_updated = Signal(str)  # (translated_text)
    status_changed = Signal(str)  # (status_message)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.running = False

        self.audio_capturer: Optional[AudioCapturer] = None
        self.vad_detector: Optional[SileroVADDetector] = None
        self.asr_engine: Optional[WhisperASREngine] = None
        self.translator: Optional[OllamaTranslator] = None

    def initialize_components(self):
        """Initialize all sub-modules."""
        self.status_changed.emit("設定を読み込み中...")
        self.audio_capturer = AudioCapturer(self.config.audio)
        self.vad_detector = SileroVADDetector(self.config.vad)
        self.asr_engine = WhisperASREngine(self.config.asr)
        self.translator = OllamaTranslator(self.config.translator)

        # Load ASR model asynchronously
        def on_asr_loaded(success: bool, msg: str):
            if success:
                self.status_changed.emit(f"音声認識モデル準備完了 ({msg})")
            else:
                self.status_changed.emit(f"音声認識モデルのロード失敗: {msg}")

        self.asr_engine.load_model_async(on_loaded_callback=on_asr_loaded)

        # Check Ollama connection
        if self.config.translator.enabled:
            ok, models = self.translator.check_connection()
            if ok:
                print(f"[Pipeline] Ollama connected. Available models: {models}")
            else:
                print("[Pipeline] Ollama not detected at start. Will retry when translating.")

    def run(self):
        self.running = True
        self.initialize_components()

        if not self.audio_capturer.start():
            self.status_changed.emit("マイク入力の開始に失敗しました。")
            return

        self.status_changed.emit("リアルタイム文字起こし開始")

        speech_chunks = []
        last_interim_time = 0.0
        interim_interval = 0.35  # seconds between real-time interim ASR updates

        while self.running:
            try:
                chunk = self.audio_capturer.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None or len(chunk) == 0:
                continue

            is_speaking, speech_started, speech_ended, full_utterance = self.vad_detector.process_chunk(chunk)

            if speech_started:
                self.status_changed.emit("🎤 音声検出中...")

            if is_speaking:
                speech_chunks.append(chunk)
                now = time.time()
                # Run interim ASR periodically if speaking
                if now - last_interim_time >= interim_interval and len(speech_chunks) > 0:
                    last_interim_time = now
                    current_audio = np.concatenate(speech_chunks)
                    if len(current_audio) >= 4800 and self.asr_engine and self.asr_engine.is_ready:  # At least 0.3s
                        interim_text = self.asr_engine.transcribe(current_audio, beam_size=1)
                        if interim_text:
                            self.asr_updated.emit(interim_text, False)

            if speech_ended and full_utterance is not None:
                speech_chunks.clear()
                self.status_changed.emit("⚡ 文字起こし処理中...")
                if self.asr_engine:
                    if not self.asr_engine.is_ready:
                        print("[Pipeline] ASR Engine loading in background, waiting for completion...")
                        # Wait up to 10 seconds for ASR model to finish loading if needed
                        for _ in range(100):
                            if self.asr_engine.is_ready or not self.running:
                                break
                            time.sleep(0.1)

                    if self.asr_engine.is_ready:
                        final_text = self.asr_engine.transcribe(full_utterance, beam_size=2)
                        if final_text:
                            print(f"[Pipeline] Confirmed ASR: {final_text}")
                            self.asr_updated.emit(final_text, True)

                            # Trigger translation if enabled
                            if self.config.translator.enabled and self.translator:
                                def _on_translated(translation: str, success: bool):
                                    if success and translation:
                                        print(f"[Pipeline] Translation: {translation}")
                                        self.translation_updated.emit(translation)

                                self.translator.translate_async(final_text, _on_translated)

    def update_config(self, new_config: AppConfig):
        """Update runtime configuration."""
        self.config = new_config
        if self.audio_capturer:
            self.audio_capturer.stop()
            self.audio_capturer.queue = queue.Queue()
            self.audio_capturer.config = new_config.audio
            self.audio_capturer.start()

        if self.vad_detector:
            self.vad_detector.config = new_config.vad
            self.vad_detector.reset_state()

        if self.translator:
            self.translator.config = new_config.translator

        if self.asr_engine and self.asr_engine.config.model_size != new_config.asr.model_size:
            self.asr_engine.config = new_config.asr
            self.asr_engine.is_ready = False
            self.asr_engine.load_model_async()

    def stop(self):
        self.running = False
        if self.audio_capturer:
            self.audio_capturer.stop()
        self.wait()
        print("[Pipeline] Pipeline thread stopped.")
