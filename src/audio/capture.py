import queue
import threading
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import sounddevice as sd

from src.core.config import AudioConfig


class AudioCapturer:
    """Captures microphone audio using sounddevice and feeds PCM chunks into a thread-safe Queue."""

    def __init__(self, config: AudioConfig, audio_callback: Optional[Callable[[np.ndarray], None]] = None):
        self.config = config
        self.audio_callback = audio_callback
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.queue: queue.Queue[np.ndarray] = queue.Queue()

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        """Return list of available audio input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_input = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    api_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else ""
                    is_default = (idx == default_input)
                    default_tag = " [デフォルト]" if is_default else ""
                    devices.append({
                        "index": idx,
                        "name": f"{dev['name']} ({api_name}){default_tag}",
                        "channels": dev["max_input_channels"],
                        "default_samplerate": dev["default_samplerate"],
                        "is_default": is_default,
                    })
        except Exception as e:
            print(f"[AudioCapturer] Error querying input devices: {e}")
        return devices

    def _audio_stream_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
        if status:
            print(f"[AudioCapturer] Buffer status warning: {status}")
        # indata is shape (frames, channels), float32 in range [-1.0, 1.0]
        # Flatten to 1D mono float32 array
        audio_chunk = indata[:, 0].astype(np.float32).copy()
        
        self.queue.put(audio_chunk)
        if self.audio_callback:
            try:
                self.audio_callback(audio_chunk)
            except Exception as e:
                print(f"[AudioCapturer] Callback error: {e}")

    def start(self) -> bool:
        if self.is_running:
            return True

        device = self.config.device_index
        if device is not None:
            try:
                dev_info = sd.query_devices(device)
                if dev_info.get("max_input_channels", 0) <= 0:
                    print(f"[AudioCapturer] Warning: Device {device} has no input channels. Falling back to default.")
                    device = None
            except Exception as e:
                print(f"[AudioCapturer] Invalid device index {device}: {e}. Falling back to default.")
                device = None

        try:
            self.stream = sd.InputStream(
                device=device,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.chunk_size,
                callback=self._audio_stream_callback,
            )
            self.stream.start()
            self.is_running = True
            dev_str = f"index {device}" if device is not None else "System Default"
            print(f"[AudioCapturer] Started listening on device: {dev_str} @ {self.config.sample_rate}Hz")
            return True
        except Exception as e:
            print(f"[AudioCapturer] Failed to start audio stream on device {device}: {e}")
            if device is not None:
                try:
                    print("[AudioCapturer] Retrying with System Default device...")
                    self.stream = sd.InputStream(
                        device=None,
                        samplerate=self.config.sample_rate,
                        channels=self.config.channels,
                        dtype="float32",
                        blocksize=self.config.chunk_size,
                        callback=self._audio_stream_callback,
                    )
                    self.stream.start()
                    self.is_running = True
                    print("[AudioCapturer] Started listening on System Default device.")
                    return True
                except Exception as ex:
                    print(f"[AudioCapturer] Fallback to default device also failed: {ex}")
            self.is_running = False
            return False

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[AudioCapturer] Error stopping stream: {e}")
            self.stream = None
        print("[AudioCapturer] Audio stream stopped")
