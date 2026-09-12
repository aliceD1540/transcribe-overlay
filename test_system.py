import sys
import numpy as np

print("=== Transcribe Overlay System Verification ===")

# Test 1: Config module
print("[Test 1/5] Testing Config module...")
from src.core.config import AppConfig
config = AppConfig.load("config.json")
assert config.audio.sample_rate == 16000
print(" -> Config module OK.")

# Test 2: Audio Capture & VAD module
print("[Test 2/5] Testing Audio Devices & VAD...")
from src.audio.capture import AudioCapturer
from src.audio.vad import SileroVADDetector

devices = AudioCapturer.get_input_devices()
print(f" -> Found {len(devices)} input devices.")

vad = SileroVADDetector(config.vad)
dummy_audio = np.zeros(512, dtype=np.float32)
prob = vad.get_speech_prob(dummy_audio)
print(f" -> VAD speech probability for silence: {prob:.4f}")
assert prob < 0.5
print(" -> Audio & VAD module OK.")

# Test 3: Ollama Client module
print("[Test 3/5] Testing Ollama Client...")
from src.translator.ollama_client import OllamaTranslator
translator = OllamaTranslator(config.translator)
connected, models = translator.check_connection()
print(f" -> Ollama connected: {connected}, Models: {models}")
print(" -> Ollama module OK.")

# Test 4: ASR Whisper Engine module
print("[Test 4/5] Testing Whisper ASR Engine initialization...")
from src.asr.whisper_engine import WhisperASREngine
asr = WhisperASREngine(config.asr)
print(" -> Whisper Engine module OK.")

# Test 5: PySide6 GUI imports & instantiation
print("[Test 5/5] Testing PySide6 UI components instantiation...")
from PySide6.QtWidgets import QApplication
from src.ui.overlay_window import OverlayWindow
from src.ui.tray_icon import SystemTrayIcon
from src.ui.settings_dialog import SettingsDialog

app = QApplication.instance() or QApplication(sys.argv)
win = OverlayWindow(config)
tray = SystemTrayIcon(win, config)
dlg = SettingsDialog(config)
print(" -> UI components, OverlayWindow, SystemTrayIcon, & SettingsDialog instantiation OK.")

print("\n=== ALL SYSTEM TESTS PASSED SUCCESSFULLY! ===")
