#!/usr/bin/env bash
# Atelier AI 원클릭 설치 스크립트 (Linux / macOS)
# 사용법: ./install.sh
set -euo pipefail

info() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m오류:\033[0m %s\n' "$*"; exit 1; }

OS="$(uname -s)"
info "Atelier AI 설치를 시작해요. (OS: $OS)"

# ---------- 1. 필수 도구 확인 ----------
check_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' 명령을 찾지 못했어요.\n$2"
}

check_cmd python3 "Python 3.11 이상을 먼저 설치해주세요: https://www.python.org/downloads/"
check_cmd node    "Node.js 18 이상을 먼저 설치해주세요: https://nodejs.org/"
check_cmd npm     "Node.js를 설치하면 npm도 함께 들어있어요: https://nodejs.org/"
check_cmd cargo   "Rust를 먼저 설치해주세요: https://rustup.rs/ (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)"

python3 -c "import sys; assert sys.version_info >= (3, 11), 'old'" 2>/dev/null \
  || die "Python 3.11 이상이 필요해요. (현재: $(python3 --version))"

# ---------- 2. Linux 시스템 의존성 안내 ----------
if [ "$OS" = "Linux" ]; then
  if ! pkg-config --exists webkit2gtk-4.1 2>/dev/null; then
    warn "Tauri 빌드에 필요한 webkit2gtk-4.1이 없어 보여요. 먼저 설치해주세요."
    echo ""
    echo "  Ubuntu/Debian:"
    echo "    sudo apt update && sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev"
    echo ""
    echo "  Fedora:"
    echo "    sudo dnf install webkit2gtk4.1-devel curl wget file libxdo-devel openssl-devel libayatana-appindicator-gtk3-devel librsvg2-devel"
    echo ""
    echo "  Arch / CachyOS / Manjaro:"
    echo "    sudo pacman -S --needed webkit2gtk-4.1 base-devel curl wget file openssl appmenu-gtk-module libappindicator-gtk3 librsvg"
    echo ""
    die "시스템 의존성을 설치한 뒤 다시 실행해주세요."
  fi
fi

if [ "$OS" = "Darwin" ]; then
  xcode-select -p >/dev/null 2>&1 \
    || die "Xcode Command Line Tools가 필요해요: xcode-select --install"
fi

# ---------- 3. 백엔드 ----------
info "백엔드 설치 중... (처음엔 수 분 걸려요)"
cd backend
if command -v uv >/dev/null 2>&1; then
  uv sync
  PYBIN=".venv/bin/python"
else
  warn "'uv'가 없어서 Python 기본 venv로 설치해요. (uv를 쓰면 더 빨라요: https://docs.astral.sh/uv/)"
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install -e .
  PYBIN=".venv/bin/python"
fi
cd ..
"$PYBIN" -c "import fastapi, uvicorn; print('백엔드 OK')" 2>/dev/null \
  || warn "백엔드 의존성 확인에 실패했어요. backend/에서 'uv sync'를 다시 해보세요."

# ---------- 4. 프론트엔드 ----------
info "프론트엔드 빌드 중..."
cd frontend
npm install --no-audit --no-fund
npm run build
cd ..

# ---------- 5. 데스크톱 앱 ----------
info "데스크톱 앱 빌드 중... (몇 분 걸려요)"
cd src-tauri
cargo build --release
cd ..

# ---------- 완료 ----------
info "설치 완료! 🎉"
echo ""
echo "  실행:  ./src-tauri/target/release/atelier-ai"
echo ""
echo "  모델 파일은 ~/.atelier-ai/models/ 에 넣으세요."
echo "    - 체크포인트(.safetensors/.ckpt)는 바로 그 폴더에"
echo "    - ControlNet은 controlnet/ 하위 폴더에"
echo "    - LoRA는 lora/ 하위 폴더에"
echo ""
echo "  모델이 없어도 앱은 열려요. (기본값 모드로 동작)"
