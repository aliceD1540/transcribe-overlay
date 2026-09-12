# 🎤 Transcribe Overlay

> **配信用 リアルタイムローカル文字起こし・翻訳オーバーレイツール**  
> クラウドAPI不要・完全0円運用。マイク音声からリアルタイムで文字起こしおよび翻訳を行い、配信画面上に透過表示させるオーバーレイアプリケーションです。

---

## ✨ 主な特徴

- 🔒 **完全ローカル運用 & 低レイテンシ**
  - 音声認識（ASR）およびLLM翻訳の全処理をローカルモデルで行うため、通信遅延や利用料金を気にせず運用可能。
- 🖼️ **OBS/配信向け 透過＆マウスイベント通過**
  - 枠なし・背景透明・最前面固定（`WindowStaysOnTopHint`）のスタイリッシュな字幕UI。
  - 通常時は**マウス透過モード**で動作し、配信画面やゲーム操作を妨げません。
- 🖱️ **タスクトレイ操作 & 直感的な位置調整**
  - タスクトレイ常駐アイコンのメニューから「**位置固定（マウス透過）**」のON/OFFを切替可能。
  - 位置固定を解除すると破線枠が表示され、ドラッグ＆ドロップおよびリサイズで位置調整が容易に行えます。
- ⚡ **超軽量 VAD + 高精度 Whisper ASR**
  - **Silero VAD** (ONNX Runtime) によるリアルタイム発話・息継ぎ判定。
  - **faster-whisper** による未確定テキストの高速リアルタイム描画と、ポーズ検出時の高精度確定描画。
- 🌐 **Ollama ローカル LLM 翻訳**
  - 確定テキストを **Ollama** (`gemma2:2b`, `qwen2.5:3b` 等) 経由で自然な英語・多言語にリアルタイム翻訳。
  - Ollama が未起動の場合でもアプリがクラッシュせず「文字起こし単体」で動作する安全設計。

---

## 🛠️ 技術スタック

| 区分 | 技術 / ライブラリ | 役割 |
| --- | --- | --- |
| **言語 / GUI** | Python 3.11+ / PySide6 | GUI・透過オーバーレイ・設定画面・トレイアイコン |
| **音声キャプチャ** | `sounddevice` | PCMマイク音声ストリームのリアルタイム取得 |
| **発話検知 (VAD)** | Silero VAD (ONNX Runtime) | 発話・ポーズの超軽量・高速判定 |
| **文字起こし (ASR)** | `faster-whisper` | Whisper 日本語モデル (`base` / `small` / `turbo` 等) |
| **翻訳 (LLM)** | Ollama API (`http://localhost:11434`) | 確定テキストのローカル翻訳 |

---

## 🚀 クイックスタート

### 1. 動作要件

- **OS**: Windows 10 / 11
- **Python**: 3.11 以上
- **(任意) Ollama**: 翻訳機能を利用する場合、[Ollama](https://ollama.com/) をインストールして起動してください。
  ```bash
  # 翻訳用モデルのダウンロード例
  ollama pull gemma2:2b
  ```

### 2. セットアップ

本リポジトリをクローンし、必要な依存ライブラリをインストールします。

```bash
git clone https://github.com/project-grimoire-dev/transcribe-overlay.git
cd transcribe-overlay
pip install -r requirements.txt
```

### 3. アプリケーションの起動

```bash
python main.py
```

---

## 📖 使い方

### 1. 位置調整とサイズ変更
1. タスクトレイ（画面右下の通知領域）にある青い「**T**」アイコンを右クリックします。
2. メニューから「**位置固定（マウス透過）**」のチェックを外します。
3. 画面上の字幕枠に青い破線が表示されます。マウスでドラッグして移動、または枠の端をドラッグしてサイズを変更します。
4. 調整が完了したら、再度トレイメニューの「**位置固定（マウス透過）**」にチェックを入れて操作不可（透過）モードに戻します。

### 2. 設定ダイアログ
トレイアイコンを右クリック ➔ 「**設定...**」を選択すると、設定画面が開きます。

- **音声入力タブ**: 使用するマイクデバイスの選択
- **文字起こしタブ**: Whisperモデルサイズ (`tiny`, `base`, `small`, `turbo` 等)、推論デバイス (`cpu`, `cuda`, `auto`)、演算精度の設定
- **翻訳タブ**: 翻訳機能のON/OFF、Ollama URL、LLMモデル名、システムプロンプトの調整および接続テスト
- **画面表示タブ**: フォントサイズ、文字色、背景色の変更

---

## ⚙️ 設定ファイル (`config.json`)

初回起動時に自動生成される `config.json` を直接編集して設定を変更することも可能です。

```json
{
  "audio": {
    "device_index": null,
    "sample_rate": 16000
  },
  "asr": {
    "model_size": "base",
    "device": "auto",
    "language": "ja"
  },
  "translator": {
    "enabled": true,
    "ollama_url": "http://localhost:11434",
    "model": "gemma2:2b"
  },
  "ui": {
    "asr_font_size": 24,
    "translation_font_size": 20,
    "click_through": true
  }
}
```

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) の下で公開されています。

Copyright (c) 2026 **project-grimoire.dev**
