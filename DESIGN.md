# 配信用リアルタイム文字起こし・翻訳アプリ 設計書

## 1. システム概要

本アプリケーションは、マイク音声からリアルタイムで文字起こしおよび翻訳を行い、配信画面上に透過表示させるオーバーレイツールです。クラウドAPIを使用せず、完全にローカル環境（0円運用）で動作し、OBS Studio等の配信ソフトウェアと組み合わせて運用することを前提とします。

### 構成のポイント

* **完全ローカル運用:** 音声認識・翻訳の全処理をローカルモデルで行い、通信遅延や利用コストを排除。
* **低負荷・リソース最適化:** CPU（Ryzen 7 8700G）と内蔵GPU（Radeon 780M）へ処理を適切に分散し、配信への影響を最小化。
* **UI透過表示:** 背景透過およびマウス操作のクロマキー/クリックスルー対応。

---

## 2. 技術スタック

| 区分 | 選定技術 / ライブラリ | 役割・選定理由 |
| --- | --- | --- |
| **言語 / GUI** | Python 3.11+ / PySide6 (Qt) | プロトタイピングの迅速性、透過UI（`WA_TranslucentBackground`）およびマウス透過（`WA_TransparentForMouseEvents`）の標準サポート |
| **音声キャプチャ** | `sounddevice` または `PyAudio` | マイクからのPCM音声ストリーム取得 |
| **音声活動検出** | Silero VAD | 発話・息継ぎ（ポーズ）の判定。CPU上で超軽量動作 |
| **文字起こし (ASR)** | `sherpa-onnx` または `faster-whisper`<br>

<br>*(DirectML / ONNX Runtime)* | Whisper (`base` または `small` 日本語モデル) を使用。内蔵GPU (Radeon 780M) 加速により超低遅延で処理 |
| **翻訳 (LLM)** | Ollama (`gemma2:2b` または `qwen2.5:3b`) | ローカルAPI接続 (`http://localhost:11434`)。確定テキストの自然な口語翻訳 |

---

## 3. システムアーキテクチャ・データフロー

```text
[マイク入力 (PCM)]
       │
       ▼
[Silero VAD (発話検知)]
       │
       ├─► 【発話中】────► 短区間ストリーミング推論 ──► 未確定テキスト ──► [UI: リアルタイム描画 (白文字)]
       │                   (Whisper / DirectML)
       │
       └─► 【ポーズ検知】──► 確定区間推論 ──────────► 確定テキスト ───► [UI: 確定描画]
                           (Whisper / DirectML)             │
                                                            ▼
                                                     [Ollama (翻訳)]
                                                            │
                                                            ▼
                                                     翻訳テキスト ────► [UI: 翻訳描画 (黄文字)]

```

---

## 4. UI / UX 仕様

1. **ウィンドウ属性:**
* 枠なし（`FramelessWindowHint`）
* 最前面固定（`WindowStaysOnTopHint`）
* タスクバー非表示（`Tool`）
* 背景透明（`WA_TranslucentBackground`）
* マウスイベント透過（`WA_TransparentForMouseEvents`）


2. **描画レイアウト:**
* **上段（文字起こし表示エリア）:** リアルタイムで入力される文字列。未確定時は薄い文字/下線表示、確定時に全表示。
* **下段（翻訳表示エリア）:** 確定イベント発生後に翻訳結果を表示（視認性向上のため黄色・半透明黒背景を標準化）。



---

## 5. 推奨される動作環境・ハードウェア設定

* **OS:** Windows 11
* **CPU:** AMD Ryzen 7 8700G (8コア / 16スレッド)
* **RAM:** 32GB (DDR5)
* **BIOS設定:** **UMA Frame Buffer Size を 4GB ～ 8GB に固定割り当て**
* 内蔵GPU（Radeon 780M）がWhisperおよびOllamaのVRAM領域を確実に確保できるようにするため。



---

## 6. プロトタイプコード (Python + PySide6)

以下は、画面透過およびレイアウトの基本構造となるスケルトンコードです。

```python
import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class OverlayWindow(QWidget):

    def __init__(self):
        super().__init__()

        # ウィンドウ属性の設定
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # UIレイアウト
        layout = QVBoxLayout()

        # 音声認識テキスト表示用ラベル
        self.asr_label = QLabel("文字起こし待機中...", self)
        self.asr_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 24px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 160);
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # 翻訳テキスト表示用ラベル
        self.translation_label = QLabel("Translation will appear here...", self)
        self.translation_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 20px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 160);
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout.addWidget(self.asr_label)
        layout.addWidget(self.translation_label)
        self.setLayout(layout)
        self.resize(800, 150)

    @Slot(str, bool)
    def update_asr_text(self, text: str, is_final: bool):
        """文字起こしテキストの更新用スロット"""
        # is_final が False の場合は未確定（リアルタイム）表示
        self.asr_label.setText(text)

    @Slot(str)
    def update_translation_text(self, text: str):
        """翻訳テキストの更新用スロット"""
        self.translation_label.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    sys.exit(app.exec())

```

---

## 7. 実装ステップ・ロードマップ

1. **フェーズ1（UI・環境構築）:**
* Python環境構築、PySide6での透過ウィンドウ作成。
* Ollamaのセットアップ（`gemma2:2b` のインストールと動作確認）。


2. **フェーズ2（音声キャプチャとVAD）:**
* `sounddevice` でマイク入力を受け取り、Silero VADで発話/ポーズの検知テスト。


3. **フェーズ3（ASR組み込み）:**
* ONNX Runtime (DirectML) 経由で Whisper モデルを読み込み、Radeon 780M 上での推論速度を検証・最適化。


4. **フェーズ4（パイプライン結合）:**
* VAD・Whisper・Ollama・PySide6 UIを非同期処理（`QThread` / `asyncio`）で結合し、描画遅延のない状態へ調整。