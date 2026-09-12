from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QTextEdit, QPushButton, QMessageBox, QLabel, QProgressBar
)
import sounddevice as sd
import numpy as np
from typing import Optional

from src.core.config import AppConfig
from src.audio.capture import AudioCapturer
from src.translator.ollama_client import OllamaTranslator


class SettingsDialog(QDialog):
    """Configuration Dialog for Audio, ASR, Ollama Translation, and UI display options."""

    settings_saved = Signal(AppConfig)
    mic_level_signal = Signal(int)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("リアルタイム文字起こし・翻訳 設定")
        self.resize(580, 520)

        self.test_stream: Optional[sd.InputStream] = None
        self.mic_level_signal.connect(self._update_mic_meter)

        self._init_ui()
        self._load_current_values()
        self._start_mic_test()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget(self)

        # Tab 1: Audio
        audio_tab = QWidget()
        audio_layout = QFormLayout(audio_tab)
        self.device_combo = QComboBox()
        self._populate_audio_devices()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        self.mic_meter = QProgressBar()
        self.mic_meter.setRange(0, 100)
        self.mic_meter.setValue(0)
        self.mic_meter.setTextVisible(False)
        self.mic_meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #222;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E5FF, stop:0.8 #00FF66, stop:1.0 #FFD700);
                border-radius: 3px;
            }
        """)

        refresh_btn = QPushButton("デバイス一覧を更新")
        refresh_btn.clicked.connect(self._populate_audio_devices)

        dev_box = QHBoxLayout()
        dev_box.addWidget(self.device_combo)
        dev_box.addWidget(refresh_btn)

        self.vad_threshold_spin = QDoubleSpinBox()
        self.vad_threshold_spin.setRange(0.05, 0.95)
        self.vad_threshold_spin.setSingleStep(0.05)
        self.vad_threshold_spin.setDecimals(2)
        self.vad_threshold_spin.setToolTip("音声検出感度（値が小さいほど感度が高くなります）")

        audio_layout.addRow("マイク入力デバイス:", dev_box)
        audio_layout.addRow("マイク入力レベル:", self.mic_meter)
        audio_layout.addRow("VAD音声検出閾値:", self.vad_threshold_spin)

        self.tab_widget.addTab(audio_tab, "音声入力")

        # Tab 2: ASR (Whisper)
        asr_tab = QWidget()
        asr_layout = QFormLayout(asr_tab)
        self.asr_model_combo = QComboBox()
        self.asr_model_combo.addItems(["tiny", "base", "small", "medium", "turbo", "large-v3-turbo"])

        self.asr_device_combo = QComboBox()
        self.asr_device_combo.addItems(["auto", "cpu", "cuda"])

        self.asr_compute_combo = QComboBox()
        self.asr_compute_combo.addItems(["default", "int8", "float32", "float16"])

        asr_layout.addRow("Whisperモデルサイズ:", self.asr_model_combo)
        asr_layout.addRow("推論デバイス:", self.asr_device_combo)
        asr_layout.addRow("演算精度 (Compute Type):", self.asr_compute_combo)
        self.tab_widget.addTab(asr_tab, "文字起こし (Whisper)")

        # Tab 3: Translation (Ollama)
        trans_tab = QWidget()
        trans_layout = QFormLayout(trans_tab)
        self.trans_enabled_cb = QCheckBox("翻訳を有効化する")
        self.ollama_url_edit = QLineEdit()
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)

        test_btn_layout = QHBoxLayout()
        test_btn = QPushButton("Ollama接続テスト")
        test_btn.clicked.connect(self._test_ollama_connection)
        test_btn_layout.addWidget(self.ollama_model_combo, 1)
        test_btn_layout.addWidget(test_btn)

        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setMaximumHeight(90)

        trans_layout.addRow(self.trans_enabled_cb)
        trans_layout.addRow("Ollama URL:", self.ollama_url_edit)
        trans_layout.addRow("翻訳LLMモデル:", test_btn_layout)
        trans_layout.addRow("システムプロンプト:", self.system_prompt_edit)
        self.tab_widget.addTab(trans_tab, "翻訳 (Ollama)")

        # Tab 4: UI / Overlay
        ui_tab = QWidget()
        ui_layout = QFormLayout(ui_tab)

        self.show_status_cb = QCheckBox("ステータス表示枠を表示する")

        self.status_font_size_spin = QSpinBox()
        self.status_font_size_spin.setRange(8, 48)

        self.asr_font_size_spin = QSpinBox()
        self.asr_font_size_spin.setRange(12, 72)

        self.trans_font_size_spin = QSpinBox()
        self.trans_font_size_spin.setRange(12, 72)

        self.status_color_edit = QLineEdit()
        self.asr_color_edit = QLineEdit()
        self.trans_color_edit = QLineEdit()
        self.bg_color_edit = QLineEdit()

        self.click_through_cb = QCheckBox("位置固定（マウス透過モード）")

        ui_layout.addRow(self.show_status_cb)
        ui_layout.addRow("ステータスフォントサイズ (px):", self.status_font_size_spin)
        ui_layout.addRow("文字起こしフォントサイズ (px):", self.asr_font_size_spin)
        ui_layout.addRow("翻訳フォントサイズ (px):", self.trans_font_size_spin)
        ui_layout.addRow("ステータス文字色 (HEX):", self.status_color_edit)
        ui_layout.addRow("文字起こし文字色 (HEX):", self.asr_color_edit)
        ui_layout.addRow("翻訳文字色 (HEX):", self.trans_color_edit)
        ui_layout.addRow("背景色 (rgba / HEX):", self.bg_color_edit)
        ui_layout.addRow(self.click_through_cb)
        self.tab_widget.addTab(ui_tab, "画面表示・オーバーレイ")

        main_layout.addWidget(self.tab_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    def _populate_audio_devices(self):
        self.device_combo.blockSignals(True)
        current_data = self.device_combo.currentData() if self.device_combo.count() > 0 else None
        self.device_combo.clear()
        self.device_combo.addItem("システムデフォルト マイク", None)
        devices = AudioCapturer.get_input_devices()
        for dev in devices:
            self.device_combo.addItem(dev["name"], dev["index"])
        
        idx = self.device_combo.findData(current_data)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.device_combo.blockSignals(False)

    def _start_mic_test(self):
        self._stop_mic_test()
        device_idx = self.device_combo.currentData()
        try:
            def audio_cb(indata, frames, time_info, status):
                rms = np.sqrt(np.mean(indata[:, 0] ** 2))
                level = min(100, int(rms * 1500))
                self.mic_level_signal.emit(level)

            self.test_stream = sd.InputStream(
                device=device_idx,
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=audio_cb,
            )
            self.test_stream.start()
        except Exception as e:
            print(f"[SettingsDialog] Could not start mic test stream: {e}")

    def _stop_mic_test(self):
        if self.test_stream is not None:
            try:
                self.test_stream.stop()
                self.test_stream.close()
            except Exception:
                pass
            self.test_stream = None

    def _on_device_changed(self):
        self._start_mic_test()

    @Slot(int)
    def _update_mic_meter(self, level: int):
        self.mic_meter.setValue(level)

    def closeEvent(self, event):
        self._stop_mic_test()
        super().closeEvent(event)

    def reject(self):
        self._stop_mic_test()
        super().reject()

    def _load_current_values(self):
        # Audio
        idx = self.device_combo.findData(self.config.audio.device_index)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        else:
            self.device_combo.setCurrentIndex(0)

        self.vad_threshold_spin.setValue(self.config.vad.threshold)

        # ASR
        idx = self.asr_model_combo.findText(self.config.asr.model_size)
        if idx >= 0:
            self.asr_model_combo.setCurrentIndex(idx)

        idx = self.asr_device_combo.findText(self.config.asr.device)
        if idx >= 0:
            self.asr_device_combo.setCurrentIndex(idx)

        idx = self.asr_compute_combo.findText(self.config.asr.compute_type)
        if idx >= 0:
            self.asr_compute_combo.setCurrentIndex(idx)

        # Translation
        self.trans_enabled_cb.setChecked(self.config.translator.enabled)
        self.ollama_url_edit.setText(self.config.translator.ollama_url)
        self.system_prompt_edit.setPlainText(self.config.translator.system_prompt)
        self._populate_ollama_models(select_model=self.config.translator.model)

        # UI
        self.show_status_cb.setChecked(getattr(self.config.ui, "show_status", True))
        self.status_font_size_spin.setValue(getattr(self.config.ui, "status_font_size", 14))
        self.asr_font_size_spin.setValue(self.config.ui.asr_font_size)
        self.trans_font_size_spin.setValue(self.config.ui.translation_font_size)
        self.status_color_edit.setText(getattr(self.config.ui, "status_color", "#00E5FF"))
        self.asr_color_edit.setText(self.config.ui.asr_color)
        self.trans_color_edit.setText(self.config.ui.translation_color)
        self.bg_color_edit.setText(self.config.ui.bg_color)
        self.click_through_cb.setChecked(self.config.ui.click_through)

    def _populate_ollama_models(self, select_model: str = ""):
        url = self.ollama_url_edit.text().strip()
        temp_config = AppConfig()
        temp_config.translator.ollama_url = url
        translator = OllamaTranslator(temp_config.translator)

        ok, models = translator.check_connection()
        self.ollama_model_combo.clear()
        if ok and models:
            for m in models:
                self.ollama_model_combo.addItem(m)
            target = select_model or self.config.translator.model
            idx = self.ollama_model_combo.findText(target)
            if idx >= 0:
                self.ollama_model_combo.setCurrentIndex(idx)
            else:
                self.ollama_model_combo.setEditText(target)
        else:
            self.ollama_model_combo.setEditText(select_model or self.config.translator.model)

    def _test_ollama_connection(self):
        url = self.ollama_url_edit.text().strip()
        temp_config = AppConfig()
        temp_config.translator.ollama_url = url
        translator = OllamaTranslator(temp_config.translator)
        
        ok, models = translator.check_connection()
        if ok:
            self._populate_ollama_models()
            msg = f"Ollamaサーバーへの接続成功！\n検出されたモデル: {', '.join(models) if models else 'なし'}"
            QMessageBox.information(self, "接続成功", msg)
        else:
            QMessageBox.warning(self, "接続失敗", f"URL {url} に接続できませんでした。\nOllamaが起動しているか確認してください。")

    def _on_save(self):
        self._stop_mic_test()

        # Update config object
        self.config.audio.device_index = self.device_combo.currentData()
        self.config.vad.threshold = self.vad_threshold_spin.value()

        self.config.asr.model_size = self.asr_model_combo.currentText()
        self.config.asr.device = self.asr_device_combo.currentText()
        self.config.asr.compute_type = self.asr_compute_combo.currentText()

        self.config.translator.enabled = self.trans_enabled_cb.isChecked()
        self.config.translator.ollama_url = self.ollama_url_edit.text().strip()
        self.config.translator.model = self.ollama_model_combo.currentText().strip()
        self.config.translator.system_prompt = self.system_prompt_edit.toPlainText().strip()

        self.config.ui.show_status = self.show_status_cb.isChecked()
        self.config.ui.status_font_size = self.status_font_size_spin.value()
        self.config.ui.asr_font_size = self.asr_font_size_spin.value()
        self.config.ui.translation_font_size = self.trans_font_size_spin.value()
        self.config.ui.status_color = self.status_color_edit.text().strip()
        self.config.ui.asr_color = self.asr_color_edit.text().strip()
        self.config.ui.translation_color = self.trans_color_edit.text().strip()
        self.config.ui.bg_color = self.bg_color_edit.text().strip()
        self.config.ui.click_through = self.click_through_cb.isChecked()

        self.config.save()
        self.settings_saved.emit(self.config)
        self.accept()
