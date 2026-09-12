import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Slot

from src.core.config import AppConfig
from src.core.pipeline import TranscriptionPipeline
from src.ui.overlay_window import OverlayWindow
from src.ui.tray_icon import SystemTrayIcon
from src.ui.settings_dialog import SettingsDialog


def main():
    # Ensure correct working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("TranscribeOverlay")

    # Load configuration
    config = AppConfig.load("config.json")

    # Create Overlay Window
    overlay_window = OverlayWindow(config)
    overlay_window.show()

    # Create Pipeline Thread
    pipeline = TranscriptionPipeline(config)
    pipeline.asr_updated.connect(overlay_window.update_asr_text)
    pipeline.translation_updated.connect(overlay_window.update_translation_text)
    pipeline.status_changed.connect(overlay_window.update_status)

    # Create System Tray Icon
    tray_icon = SystemTrayIcon(overlay_window, config)

    # Settings dialog handler
    settings_dialog = None

    def show_settings():
        nonlocal settings_dialog
        if settings_dialog is None:
            settings_dialog = SettingsDialog(config, parent=None)

            @Slot(AppConfig)
            def on_settings_saved(new_config: AppConfig):
                overlay_window.app_config = new_config
                overlay_window.ui_config = new_config.ui
                overlay_window.set_display_mode(new_config.ui.display_mode)
                overlay_window.apply_styles()
                overlay_window.set_click_through(new_config.ui.click_through)
                tray_icon.click_through_action.setChecked(new_config.ui.click_through)
                tray_icon.translation_action.setChecked(new_config.translator.enabled)
                pipeline.update_config(new_config)

            settings_dialog.settings_saved.connect(on_settings_saved)

        settings_dialog.exec()

    tray_icon.open_settings_requested.connect(show_settings)

    def quit_application():
        print("[Main] Shutting down application...")
        pipeline.stop()
        app.quit()

    overlay_window.quit_requested.connect(quit_application)
    tray_icon.quit_requested.connect(quit_application)
    app.aboutToQuit.connect(pipeline.stop)

    # Start audio pipeline thread
    pipeline.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
