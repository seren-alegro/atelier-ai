# Atelier AI

[한국어](README.md) · [English](README.en.md)

画像生成専用のデスクトップツール。初心者でもすぐ使える直感的なUIを目指しています。

- 完全ローカル動作（インターネットなしで実行可能）
- Linux + macOS + Windows (Tauri)
- Anima系モデルに特化
- Danbooru式タグ優先のUX
- すべての設定に説明＋推奨値を表示（日・韓・英の3言語対応）

📖 [使い方ガイド](docs/GUIDE.ja.md) — このツールが何か、何ができるのか、初めての方へ

## インストール（初めての方）

### 1. 必要なものをインストール

| 必要なもの | 説明 | 入手先 |
|---|---|---|
| Python 3.11 以上 | バックエンド実行用 | https://www.python.org/downloads/ |
| Node.js 18 以上 | 画面ビルド用（npm 含む） | https://nodejs.org/ |
| Rust | アプリビルド用（`cargo` コマンド） | https://rustup.rs/ |

OS別の注意点：

- **Ubuntu/Debian**: `sudo apt install python3 python3-venv` の後、Node.js・Rustは上記リンクから
- **Arch/CachyOS**: `sudo pacman -S python nodejs npm` の後、Rustは上記リンクから
- **macOS**: `xcode-select --install` の後、[Homebrew](https://brew.sh/)で `brew install python node`
- **Windows**: 上記リンクからそれぞれインストール＋ [C++ Build Tools](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)（「C++によるデスクトップ開発」にチェック）

### 2. ワンクリックインストール

```bash
git clone https://github.com/seren-alegro/atelier-ai.git
cd atelier-ai
```

Linux / macOS:
```bash
./install.sh
```

Windows (PowerShell):
```powershell
.\install.ps1
```

スクリプトがバックエンドのインストール → 画面のビルド → デスクトップアプリのビルドを順番に実行します。
初回は依存関係のダウンロードのため数分かかります。

### 3. モデルファイルを配置

モデルフォルダ：`~/.atelier-ai/models/`（Windows: `%USERPROFILE%\.atelier-ai\models\`）

- チェックポイント（`.safetensors` / `.ckpt`）→ フォルダ直下
- ControlNet → `controlnet/` サブフォルダ
- LoRA → `lora/` サブフォルダ

モデルがなくてもアプリは起動します。（デフォルト値モードで動作し、モデルタブで状態を確認できます）

### 4. 実行

Linux / macOS:
```bash
./src-tauri/target/release/atelier-ai
```

Windows:
```powershell
.\src-tauri\target\release\atelier-ai.exe
```

アプリを起動するとバックエンドも自動で起動します。終了時はアプリを閉じるだけです。

## LoRA学習

作業タブの「LoRA学習」から、自分の画像でキャラクター・画風LoRAを直接作れます。

1. ベースチェックポイントを選択（AnimaなどSDXL系）
2. 学習画像フォルダを指定（15〜30枚推奨）
3. 「一括自動キャプション」で.txtキャプションを生成（WD14 v3 Danbooruタグ / BLIP自然文）
4. トリガーワード・rank・学習率などを設定して「学習開始」

結果はモデルフォルダの `lora/` に保存され、すぐ使えます。
学習には12GB以上のVRAMが必要なため、GPUのない環境ではRunPodなどの外部GPUを使います。

## 開発者向け

### 構成

```
atelier-ai/
├── backend/        # Python + FastAPI（生成エンジン）
├── frontend/       # Svelte 5 + Vite Web UI
├── src-tauri/      # Tauriデスクトップラッパー
├── install.sh      # ワンクリックインストール（Linux/macOS）
├── install.ps1     # ワンクリックインストール（Windows）
└── docs/           # 企画スペック
```

### ビルドなしで実行（フロント開発用）

ターミナルを2つ開きます。

**1. バックエンド** (http://127.0.0.1:8079)

```bash
cd backend
uv sync
uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
```

**2. フロントエンド** (http://localhost:5173)

```bash
cd frontend
npm install
npm run dev
```

### テスト

```bash
cd backend && uv run pytest tests/ -q
cd frontend && npm test && npm run build
```

## リリース（自動更新）

タグをプッシュするとGitHub ActionsがLinux/Windows/macOS用インストーラをビルドし、
リリースを発行します。アプリは起動時に新バージョンを確認し、設定タブから
ダウンロード・インストール・再起動できます。（Tauriアップデータ＋署名）

### 初回のみ：署名キーの登録

アップデータ署名用の秘密鍵はリポジトリに上げません。
鍵ファイル（`updater.key`）を受け取った後、GitHubリポジトリの
Settings → Secrets and variables → Actionsに登録してください。

| Secret | 値 |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | `updater.key` ファイルの内容全体 |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | 空の値（鍵にパスワードなし） |

```sh
cat updater.key | base64 -w0 | gh secret set TAURI_SIGNING_PRIVATE_KEY
printf '' | gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD
```

### リリースする

```sh
./scripts/bump-version.sh 0.2.0   # 3か所のバージョンを一括更新
git add -A && git commit -m "v0.2.0"
git tag v0.2.0 && git push origin main v0.2.0
```

ワークフローはバックエンドの凍結（PyInstaller）→ Tauriビルド・署名 → draftリリース作成 →
`latest.json` 検証 → 公開の順に進みます。公開後にアプリが更新を検知します。

### 環境変数

| 変数 | 説明 | デフォルト |
|---|---|---|
| `ATELIER_MODELS_DIR` | モデルフォルダのパス | `~/.atelier-ai/models` |
| `ATELIER_DATA_DIR` | 作業履歴の保存パス | `~/.atelier-ai` |
| `ATELIER_BACKEND_CMD` | バックエンド実行コマンドを直接指定（Tauri） | 自動探索 |
| `ATELIER_BACKEND_DIR` | バックエンド作業ディレクトリ（Tauri） | 自動探索 |

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| UIに「デフォルト値使用中」バッジが出る | バックエンドがまだ起動中の可能性があります。最大3分まで待って、ダメならバッジをクリックして再接続 |
| 「モデルがロードされていません」（503） | モデルフォルダにチェックポイント（`.safetensors`）があるか確認。モデルタブでも確認可能 |
| 「Pythonが見つかりません」 | Python 3.11+をインストール。Windowsはpython.orgからインストール後 `py -3 --version` を確認 |
| ポート8079が使用中というエラー | すでにバックエンドが動いています。既存プロセスを終了するか、そのまま使います |
| 生成中にエラー／画面が固まる | VRAM不足の可能性があります。解像度やバッチを下げてみてください |
| `./install.sh`がTauri依存関係で止まる | `src-tauri/README.md`のOS別システムパッケージ一覧を確認 |
| `npm run dev`が動かない | Node.js 18+を確認後 `rm -rf node_modules && npm install` |

## 動作要件

NVIDIA GPU（CUDA）またはApple Silicon（Metal/MPS）が必要です。CPU専用環境は
サポートしておらず、生成リクエスト時にエラーが発生します。

## ライセンス

MIT

## 謝辞

- タグ日本語・韓国語ラベル：[Danbooru Wiki韓国語翻訳](https://localsmile.github.io/danbooru_KR_wiki_tag_search/)（localsmileさん）— 初回実行時にダウンロードして使用します。
