from PySide6.QtCore import Qt, QObject, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

from src.core.config import AppConfig
from src.ui.overlay_window import OverlayWindow


class SystemTrayIcon(QObject):
    """System Tray Icon controlling Overlay Window mode and application settings."""

    open_settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, overlay_window: OverlayWindow, config: AppConfig):
        super().__init__()
        self.overlay_window = overlay_window
        self.config = config

        self.tray_icon = QSystemTrayIcon(self._generate_default_icon())
        self.tray_icon.setToolTip("リアルタイム文字起こし・翻訳オーバーレイ")

        self._create_menu()
        self.tray_icon.show()

    def _generate_default_icon(self) -> QIcon:
        """Dynamically generate a clean 32x32 icon."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        painter.setBrush(QColor(30, 144, 255))
        painter.setPen(QColor(255, 255, 255))
        painter.drawEllipse(2, 2, 28, 28)

        # "T" text symbol
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "T")
        painter.end()

        return QIcon(pixmap)

    def _create_menu(self):
        menu = QMenu()

        # Action: Toggle Click-Through / Lock Position
        self.click_through_action = menu.addAction("位置固定（マウス透過）")
        self.click_through_action.setCheckable(True)
        self.click_through_action.setChecked(self.config.ui.click_through)
        self.click_through_action.triggered.connect(self._on_toggle_click_through)

        # Action: Toggle Translation
        self.translation_action = menu.addAction("翻訳を有効化")
        self.translation_action.setCheckable(True)
        self.translation_action.setChecked(self.config.translator.enabled)
        self.translation_action.triggered.connect(self._on_toggle_translation)

        menu.addSeparator()

        # Action: Reset Position
        reset_action = menu.addAction("表示位置リセット")
        reset_action.triggered.connect(self._on_reset_position)

        # Action: Open Settings
        settings_action = menu.addAction("設定...")
        settings_action.triggered.connect(self.open_settings_requested.emit)

        menu.addSeparator()

        # Action: Quit
        quit_action = menu.addAction("終了")
        quit_action.triggered.connect(self.quit_requested.emit)

        self.tray_icon.setContextMenu(menu)

    @Slot(bool)
    def _on_toggle_click_through(self, checked: bool):
        self.config.ui.click_through = checked
        self.config.save()
        self.overlay_window.set_click_through(checked)

    @Slot(bool)
    def _on_toggle_translation(self, checked: bool):
        self.config.translator.enabled = checked
        self.config.save()
        if hasattr(self.overlay_window, "config_changed"):
            self.overlay_window.config_changed.emit(self.config)

    def _on_reset_position(self):
        self.overlay_window.move(100, 100)
        self.overlay_window.resize(800, 160)
        self.config.ui.window_x = 100
        self.config.ui.window_y = 100
        self.config.ui.window_width = 800
        self.config.ui.window_height = 160
        self.config.save()
