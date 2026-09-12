import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AudioConfig:
    device_index: Optional[int] = None  # None uses default input device
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512  # VAD sample chunk (e.g., 512 samples = 32ms at 16kHz)


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200


@dataclass
class ASRConfig:
    model_size: str = "base"  # tiny, base, small, medium, turbo, large-v3-turbo
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "default"  # default, float32, int8, float16
    language: str = "ja"


@dataclass
class TranslatorConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma2:2b"
    system_prompt: str = (
        "You are a professional live stream translator. "
        "Translate the following Japanese speech transcript into natural, spoken English. "
        "Output ONLY the English translation without any preamble, explanation, or quotes."
    )
    temperature: float = 0.3
    timeout_sec: float = 8.0


@dataclass
class UIConfig:
    font_family: str = "Segoe UI"
    asr_font_size: int = 24
    translation_font_size: int = 20
    status_font_size: int = 14
    asr_color: str = "#FFFFFF"
    translation_color: str = "#FFD700"
    status_color: str = "#00E5FF"
    bg_color: str = "rgba(0, 0, 0, 160)"
    border_radius: int = 8
    padding: int = 10
    window_x: int = 100
    window_y: int = 100
    window_width: int = 800
    window_height: int = 200
    click_through: bool = True
    always_on_top: bool = True
    show_status: bool = True
    display_mode: str = "transparent"  # "transparent" or "window"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, file_path: str = "config.json") -> "AppConfig":
        if not os.path.exists(file_path):
            config = cls()
            config.save(file_path)
            return config
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                audio=AudioConfig(**data.get("audio", {})),
                vad=VADConfig(**data.get("vad", {})),
                asr=ASRConfig(**data.get("asr", {})),
                translator=TranslatorConfig(**data.get("translator", {})),
                ui=UIConfig(**data.get("ui", {})),
            )
        except Exception as e:
            print(f"[Config] Error loading {file_path}, using defaults: {e}")
            return cls()

    def save(self, file_path: str = "config.json") -> None:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            print(f"[Config] Saved to {file_path}")
        except Exception as e:
            print(f"[Config] Error saving {file_path}: {e}")
