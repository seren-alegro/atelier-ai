# Atelier AI

[한국어](README.md)

A desktop tool dedicated to image generation, with an intuitive UI that beginners can use right away.

- Fully local (runs without internet)
- Linux + macOS + Windows (Tauri)
- Specialized for Anima-series models
- Danbooru-style tag-first UX
- Every setting shows a Korean explanation + recommended value

## Installation (first time)

### 1. Prerequisites

| Requirement | Purpose | Download |
|---|---|---|
| Python 3.11+ | Backend runtime | https://www.python.org/downloads/ |
| Node.js 18+ | UI build (includes npm) | https://nodejs.org/ |
| Rust | App build (`cargo` command) | https://rustup.rs/ |

OS-specific notes:

- **Ubuntu/Debian**: `sudo apt install python3 python3-venv`, then Node.js/Rust from the links above
- **Arch/CachyOS**: `sudo pacman -S python nodejs npm`, then Rust from the link above
- **macOS**: `xcode-select --install`, then `brew install python node` via [Homebrew](https://brew.sh/)
- **Windows**: install each from the links above + [C++ Build Tools](https://visualstudio.microsoft.com/vs/visual-cpp-build-tools/) (check "Desktop development with C++")

### 2. One-click install

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

The script installs the backend → builds the UI → builds the desktop app, in order.
The first run takes several minutes due to dependency downloads.

### 3. Add model files

Model folder: `~/.atelier-ai/models/` (Windows: `%USERPROFILE%\.atelier-ai\models\`)

- Checkpoints (`.safetensors` / `.ckpt`) → directly in the folder
- ControlNet → `controlnet/` subfolder
- LoRA → `lora/` subfolder

The app opens even without models. (It runs in default mode; check status in the Models tab.)

### 4. Run

Linux / macOS:
```bash
./src-tauri/target/release/atelier-ai
```

Windows:
```powershell
.\src-tauri\target\release\atelier-ai.exe
```

The backend starts automatically with the app. To quit, just close the app.

## LoRA training

The "LoRA training" tab lets you train your own character/style LoRAs from your images.

1. Pick a base checkpoint (Anima or other SDXL-based)
2. Point to a folder of training images (15–30 recommended)
3. Generate .txt captions with "batch auto-caption" (WD14 v3 Danbooru tags / BLIP natural language)
4. Set trigger word, rank, learning rate, etc., then "start training"

Results are saved to the `lora/` model folder and ready to use immediately.
Training needs 12GB+ VRAM, so on machines without a GPU use an external GPU (e.g. RunPod).

## For developers

### Structure

```
atelier-ai/
├── backend/        # Python + FastAPI (generation engine)
├── frontend/       # Svelte 5 + Vite web UI
├── src-tauri/      # Tauri desktop wrapper
├── install.sh      # one-click install (Linux/macOS)
├── install.ps1     # one-click install (Windows)
└── docs/           # design spec
```

### Run without building (frontend development)

Open two terminals.

**1. Backend** (http://127.0.0.1:8079)

```bash
cd backend
uv sync
uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
```

**2. Frontend** (http://localhost:5173)

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
cd backend && uv run pytest tests/ -q
cd frontend && npm test && npm run build
```

## Releases (auto-update)

Pushing a tag makes GitHub Actions build installers for Linux/Windows/macOS and
publish a release. The app checks for new versions on startup, and you can
download, install, and restart from the Settings tab. (Tauri updater + signing)

### One-time setup: signing key

The updater signing private key is never committed to the repo.
After receiving the key file (`updater.key`), register it in the GitHub repo's
Settings → Secrets and variables → Actions.

| Secret | Value |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | Entire contents of the `updater.key` file |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Empty (the key has no password) |

```sh
cat updater.key | base64 -w0 | gh secret set TAURI_SIGNING_PRIVATE_KEY
printf '' | gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD
```

### Making a release

```sh
./scripts/bump-version.sh 0.2.0   # bumps the version in all three places
git add -A && git commit -m "v0.2.0"
git tag v0.2.0 && git push origin main v0.2.0
```

The workflow runs: backend freeze (PyInstaller) → Tauri build & sign → draft release →
`latest.json` verification → publish, in order. Apps detect the update after publishing.

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `ATELIER_MODELS_DIR` | Model folder path | `~/.atelier-ai/models` |
| `ATELIER_DATA_DIR` | History storage path | `~/.atelier-ai` |
| `ATELIER_BACKEND_CMD` | Backend launch command override (Tauri) | auto-detect |
| `ATELIER_BACKEND_DIR` | Backend working directory (Tauri) | auto-detect |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Using defaults" badge in the UI | The backend may still be starting. Wait up to 3 minutes; if it persists, click the badge to reconnect |
| "Model not loaded" (503) | Check that the model folder contains a checkpoint (`.safetensors`). Also visible in the Models tab |
| "Python not found" | Install Python 3.11+. On Windows, install from python.org and verify with `py -3 --version` |
| "Port 8079 already in use" | A backend is already running. Kill the existing process or just use it |
| Error during generation / frozen UI | May be VRAM shortage. Try lowering resolution or batch size |
| `./install.sh` stalls on Tauri dependencies | See the OS-specific system package list in `src-tauri/README.md` |
| `npm run dev` fails | Verify Node.js 18+, then `rm -rf node_modules && npm install` |

## Requirements

An NVIDIA GPU (CUDA) or Apple Silicon (Metal/MPS) is required. CPU-only machines are not supported — generation requests will fail with an error.

## License

MIT
