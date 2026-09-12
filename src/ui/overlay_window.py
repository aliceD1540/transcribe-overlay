from PySide6.QtCore import Qt, Slot, QPoint, Signal
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PySide6.QtGui import QColor, QFont, QMouseEvent

from src.core.config import UIConfig, AppConfig


class OverlayWindow(QWidget):
    """Transparent overlay window for live stream transcription and translation."""

    config_changed = Signal(AppConfig)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.app_config = config
        self.ui_config: UIConfig = config.ui

        self.drag_position = QPoint()
        self.is_dragging = False

        self._init_ui()
        self._init_window_flags()
        self.apply_styles()

    def _init_window_flags(self):
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.set_click_through(self.ui_config.click_through)

    def set_click_through(self, enabled: bool):
        self.ui_config.click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        # Update window flags to refresh mouse event passthrough state on Windows
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.ui_config.always_on_top)
        self.show()
        if hasattr(self, "asr_label"):
            self.apply_styles()

    def _init_ui(self):
        self.setGeometry(
            self.ui_config.window_x,
            self.ui_config.window_y,
            self.ui_config.window_width,
            self.ui_config.window_height,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Status Indicator Label
        self.status_label = QLabel("待機中...", self)
        self.status_label.setWordWrap(True)

        # ASR Text Label
        self.asr_label = QLabel("（文字起こし結果がここに表示されます）", self)
        self.asr_label.setWordWrap(True)

        # Translation Text Label
        self.translation_label = QLabel("Translation will appear here...", self)
        self.translation_label.setWordWrap(True)

        # Drop shadows for crisp text visibility over any background
        shadow0 = QGraphicsDropShadowEffect(self)
        shadow0.setBlurRadius(8)
        shadow0.setColor(QColor(0, 0, 0, 220))
        shadow0.setOffset(1, 1)
        self.status_label.setGraphicsEffect(shadow0)

        shadow1 = QGraphicsDropShadowEffect(self)
        shadow1.setBlurRadius(8)
        shadow1.setColor(QColor(0, 0, 0, 220))
        shadow1.setOffset(1, 1)
        self.asr_label.setGraphicsEffect(shadow1)

        shadow2 = QGraphicsDropShadowEffect(self)
        shadow2.setBlurRadius(8)
        shadow2.setColor(QColor(0, 0, 0, 220))
        shadow2.setOffset(1, 1)
        self.translation_label.setGraphicsEffect(shadow2)

        layout.addWidget(self.status_label)
        layout.addWidget(self.asr_label)
        layout.addWidget(self.translation_label)
        self.setLayout(layout)

    def apply_styles(self):
        """Apply CSS styling based on UIConfig and position lock state."""
        border_style = "2px dashed #00E5FF;" if not self.ui_config.click_through else "none;"

        status_style = f"""
            QLabel {{
                color: {getattr(self.ui_config, 'status_color', '#00E5FF')};
                font-family: '{self.ui_config.font_family}';
                font-size: {getattr(self.ui_config, 'status_font_size', 14)}px;
                font-weight: bold;
                background-color: {self.ui_config.bg_color};
                border-radius: {self.ui_config.border_radius}px;
                padding: {max(4, self.ui_config.padding // 2)}px {self.ui_config.padding}px;
                border: {border_style}
            }}
        """

        asr_style = f"""
            QLabel {{
                color: {self.ui_config.asr_color};
                font-family: '{self.ui_config.font_family}';
                font-size: {self.ui_config.asr_font_size}px;
                font-weight: bold;
                background-color: {self.ui_config.bg_color};
                border-radius: {self.ui_config.border_radius}px;
                padding: {self.ui_config.padding}px;
                border: {border_style}
            }}
        """

        translation_style = f"""
            QLabel {{
                color: {self.ui_config.translation_color};
                font-family: '{self.ui_config.font_family}';
                font-size: {self.ui_config.translation_font_size}px;
                font-weight: bold;
                background-color: {self.ui_config.bg_color};
                border-radius: {self.ui_config.border_radius}px;
                padding: {self.ui_config.padding}px;
                border: {border_style}
            }}
        """

        self.status_label.setStyleSheet(status_style)
        self.status_label.setVisible(getattr(self.ui_config, "show_status", True))
        self.asr_label.setStyleSheet(asr_style)
        self.translation_label.setStyleSheet(translation_style)

    @Slot(str, bool)
    def update_asr_text(self, text: str, is_final: bool):
        """Update ASR transcription text slot."""
        if not text:
            return
        if not is_final:
            # Interim (unconfirmed): translucent / italic representation
            self.asr_label.setText(f"<i>... {text}</i>")
        else:
            self.asr_label.setText(text)

    @Slot(str)
    def update_translation_text(self, text: str):
        """Update translation text slot."""
        if text:
            self.translation_label.setText(text)

    @Slot(str)
    def update_status(self, status: str):
        """Update status message slot."""
        self.status_label.setText(status)

    # Mouse drag support when position lock / click-through is turned OFF
    def mousePressEvent(self, event: QMouseEvent):
        if not self.ui_config.click_through and event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.ui_config.click_through and self.is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.ui_config.click_through and self.is_dragging:
            self.is_dragging = False
            # Save updated position to config
            self.ui_config.window_x = self.x()
            self.ui_config.window_y = self.y()
            self.ui_config.window_width = self.width()
            self.ui_config.window_height = self.height()
            self.app_config.save()
            event.accept()
