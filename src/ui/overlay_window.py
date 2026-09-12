from PySide6.QtCore import Qt, Slot, QPoint, Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QColor, QFont, QMouseEvent

from src.core.config import UIConfig, AppConfig


class DraggableTitleBar(QWidget):
    """Custom title bar that allows dragging the parent window."""

    def __init__(self, content: QWidget, parent_window: QWidget):
        super().__init__()
        self.parent_window = parent_window
        self.is_dragging = False
        self.drag_position = QPoint()

        # Use the content's layout
        layout = content.layout()
        if layout:
            # Create a new layout for this wrapper
            wrapper_layout = QVBoxLayout()
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(0)

            # Copy all items from content layout to wrapper
            for i in range(layout.count()):
                item = layout.itemAt(0)
                if item.widget():
                    wrapper_layout.addWidget(item.widget())
                elif item.layout():
                    wrapper_layout.addLayout(item.layout())

            self.setLayout(wrapper_layout)

        # Copy styling from content
        self.setStyleSheet(content.styleSheet())
        self.setFixedHeight(36)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_position = (
                event.globalPosition().toPoint()
                - self.parent_window.frameGeometry().topLeft()
            )
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.parent_window.move(
                event.globalPosition().toPoint() - self.drag_position
            )
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self.is_dragging = False
            # Save position
            self.parent_window.ui_config.window_x = self.parent_window.x()
            self.parent_window.ui_config.window_y = self.parent_window.y()
            self.parent_window.ui_config.window_width = self.parent_window.width()
            self.parent_window.ui_config.window_height = self.parent_window.height()
            self.parent_window.app_config.save()
            event.accept()
        else:
            event.ignore()


class OverlayWindow(QWidget):
    """Transparent overlay window for live stream transcription and translation."""

    config_changed = Signal(AppConfig)
    display_mode_changed = Signal(str)  # Signal to notify mode changes

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
        # Apply window flags based on display mode
        self._apply_display_mode_flags()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.set_click_through(self.ui_config.click_through)

    def _apply_display_mode_flags(self):
        """Apply window flags based on current display mode."""
        is_transparent = self.ui_config.display_mode == "transparent"

        if is_transparent:
            # Transparent overlay mode: frameless, on top, tool window
            flags = (
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
        else:
            # Normal window mode: standard window with frame
            flags = Qt.WindowType.WindowStaysOnTopHint

        self.setWindowFlags(flags)

    def set_click_through(self, enabled: bool):
        self.ui_config.click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        # Update window flags to refresh mouse event passthrough state on Windows
        self.setWindowFlag(
            Qt.WindowType.FramelessWindowHint,
            self.ui_config.display_mode == "transparent",
        )
        self.setWindowFlag(
            Qt.WindowType.WindowStaysOnTopHint, self.ui_config.always_on_top
        )
        self.show()
        if hasattr(self, "asr_label"):
            self.apply_styles()

    def set_display_mode(self, mode: str):
        """Switch between 'transparent' and 'window' display modes."""
        if mode not in ("transparent", "window"):
            print(f"[OverlayWindow] Invalid display mode: {mode}")
            return

        if self.ui_config.display_mode == mode:
            return  # No change needed

        print(
            f"[OverlayWindow] Switching display mode from '{self.ui_config.display_mode}' to '{mode}'"
        )
        self.ui_config.display_mode = mode

        # Re-apply window flags for the new mode
        self._apply_display_mode_flags()

        # Adjust size when switching modes
        if mode == "window":
            # Increase window size for window mode
            if self.ui_config.window_height < 400:
                self.ui_config.window_height = 400
            self.resize(self.ui_config.window_width, self.ui_config.window_height)

            # Create or show title bar
            if not self.title_bar_widget:
                # Create title bar if it doesn't exist
                self.title_bar_widget = self._create_title_bar()
                # Get the main layout and insert title bar at the beginning
                main_layout = self.layout()
                if main_layout:
                    main_layout.insertWidget(0, self.title_bar_widget)
            else:
                self.title_bar_widget.show()

            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self.ui_config.click_through = False
        else:
            # Hide title bar in transparent mode
            if self.title_bar_widget:
                self.title_bar_widget.hide()
            # In transparent mode, re-enable click-through
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                self.ui_config.click_through,
            )

        # Refresh styles and show
        self.apply_styles()
        self.show()

        # Emit signal to notify others
        self.display_mode_changed.emit(mode)

    def _init_ui(self):
        # Adjust window size based on display mode
        window_height = self.ui_config.window_height
        if self.ui_config.display_mode == "window":
            # Window mode needs more space for titlebar and content
            window_height = max(400, window_height)

        self.setGeometry(
            self.ui_config.window_x,
            self.ui_config.window_y,
            self.ui_config.window_width,
            window_height,
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create title bar for window mode
        self.title_bar_widget = None
        if self.ui_config.display_mode == "window":
            self.title_bar_widget = self._create_title_bar()
            main_layout.addWidget(self.title_bar_widget)

        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(8)

        # Status Indicator Label
        self.status_label = QLabel("待機中...", self)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(30)

        # ASR Text Label
        self.asr_label = QLabel("（文字起こし結果がここに表示されます）", self)
        self.asr_label.setWordWrap(True)
        self.asr_label.setMinimumHeight(60)

        # Translation Text Label
        self.translation_label = QLabel("Translation will appear here...", self)
        self.translation_label.setWordWrap(True)
        self.translation_label.setMinimumHeight(60)

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

        content_layout.addWidget(self.status_label)
        content_layout.addWidget(self.asr_label)
        content_layout.addWidget(self.translation_label)
        content_layout.addStretch()

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def _create_title_bar(self) -> QWidget:
        """Create a custom title bar for window mode."""
        title_bar = QWidget()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-bottom: 1px solid #444;
            }
        """)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(10, 0, 10, 0)
        title_layout.setSpacing(10)

        title_label = QLabel("リアルタイム文字起こし・翻訳")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI';
                font-size: 12px;
                font-weight: bold;
            }
        """)

        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_bar.setLayout(title_layout)

        # Wrap title bar with a custom draggable title bar widget
        draggable_title_bar = DraggableTitleBar(title_bar, self)
        return draggable_title_bar

    def apply_styles(self):
        """Apply CSS styling based on UIConfig, display mode, and position lock state."""
        # Determine border and background based on mode and position lock
        is_transparent_mode = self.ui_config.display_mode == "transparent"
        should_show_border = not self.ui_config.click_through and is_transparent_mode
        border_style = "2px dashed #00E5FF;" if should_show_border else "none;"

        # In window mode, use more opaque background; in transparent mode, use semi-transparent
        if is_transparent_mode:
            bg_color = self.ui_config.bg_color
            # In transparent mode, set window background to transparent
            self.setStyleSheet("QWidget { background-color: transparent; }")
        else:
            # Window mode: use more opaque background (replace alpha value)
            # Convert rgba(..., alpha) to rgba(..., 220) for window mode
            bg = self.ui_config.bg_color
            if "rgba" in bg:
                # Extract RGB values and set higher alpha
                bg_color = bg.rsplit(",", 1)[0] + ", 220)"
                window_bg = bg_color
            else:
                bg_color = bg
                window_bg = bg
            # Set main window background
            self.setStyleSheet(f"QWidget {{ background-color: {window_bg}; }}")

        status_style = f"""
            QLabel {{
                color: {getattr(self.ui_config, 'status_color', '#00E5FF')};
                font-family: '{self.ui_config.font_family}';
                font-size: {getattr(self.ui_config, 'status_font_size', 14)}px;
                font-weight: bold;
                background-color: {bg_color};
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
                background-color: {bg_color};
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
                background-color: {bg_color};
                border-radius: {self.ui_config.border_radius}px;
                padding: {self.ui_config.padding}px;
                border: {border_style}
            }}
        """

        self.status_label.setStyleSheet(status_style)
        self.status_label.setVisible(getattr(self.ui_config, "show_status", True))
        self.asr_label.setStyleSheet(asr_style)
        self.translation_label.setStyleSheet(translation_style)

        # Show/hide title bar based on display mode
        if self.title_bar_widget:
            self.title_bar_widget.setVisible(self.ui_config.display_mode == "window")

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

    # Mouse drag support for transparent mode
    def mousePressEvent(self, event: QMouseEvent):
        # Only handle left-click in transparent mode
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        # In transparent mode, allow dragging when click-through is OFF
        if (
            self.ui_config.display_mode == "transparent"
            and not self.ui_config.click_through
        ):
            self.is_dragging = True
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            self.ui_config.display_mode == "transparent"
            and self.is_dragging
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self.is_dragging = False
            # Save updated position to config
            self.ui_config.window_x = self.x()
            self.ui_config.window_y = self.y()
            self.ui_config.window_width = self.width()
            self.ui_config.window_height = self.height()
            self.app_config.save()
            event.accept()
        else:
            event.ignore()
