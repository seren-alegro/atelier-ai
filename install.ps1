# Atelier AI 원클릭 설치 스크립트 (Windows)
# 사용법: PowerShell에서 .\install.ps1
$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "`n==> $msg" -ForegroundColor Blue }
function Die($msg)  { Write-Host "오류: $msg" -ForegroundColor Red; exit 1 }

Info "Atelier AI 설치를 시작해요. (OS: Windows)"

# ---------- 1. 필수 도구 확인 ----------
foreach ($cmd in @("python", "node", "npm", "cargo")) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Die "'$cmd' 명령을 찾지 못했어요.`npython: https://www.python.org/downloads/`nnode/npm: https://nodejs.org/`ncargo: https://rustup.rs/ (rustup-init.exe)"
  }
}

$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVersion -lt [version]"3.11") {
  Die "Python 3.11 이상이 필요해요. (현재: $pyVersion)"
}

# WebView2 안내 (보통은 이미 있음)
Info "WebView2 확인 중..."
$wv = Get-ItemProperty "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" -ErrorAction SilentlyContinue
if (-not $wv) {
  Write-Host "주의: WebView2 런타임이 없어 보여요. 없다면 설치해주세요: https://developer.microsoft.com/ko-kr/microsoft-edge/webview2/" -ForegroundColor Yellow
}

# ---------- 2. 백엔드 ----------
Info "백엔드 설치 중... (처음엔 수 분 걸려요)"
Push-Location backend
try {
  if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv sync
  } else {
    Write-Host "주의: 'uv'가 없어서 Python 기본 venv로 설치해요." -ForegroundColor Yellow
    if (-not (Test-Path .venv)) { python -m venv .venv }
    .\.venv\Scripts\pip install --quiet --upgrade pip
    .\.venv\Scripts\pip install -e .
  }
} finally { Pop-Location }

# ---------- 3. 프론트엔드 ----------
Info "프론트엔드 빌드 중..."
Push-Location frontend
try {
  npm install --no-audit --no-fund
  npm run build
} finally { Pop-Location }

# ---------- 4. 데스크톱 앱 ----------
Info "데스크톱 앱 빌드 중... (몇 분 걸려요)"
Push-Location src-tauri
try {
  cargo build --release
} finally { Pop-Location }

# ---------- 완료 ----------
Info "설치 완료! 🎉"
Write-Host ""
Write-Host "  실행:  .\src-tauri\target\release\atelier-ai.exe"
Write-Host ""
Write-Host "  모델 파일은 $env:USERPROFILE\.atelier-ai\models\ 에 넣으세요."
Write-Host "    - 체크포인트(.safetensors/.ckpt)는 바로 그 폴더에"
Write-Host "    - ControlNet은 controlnet\ 하위 폴더에"
Write-Host "    - LoRA는 lora\ 하위 폴더에"
Write-Host ""
Write-Host "  모델이 없어도 앱은 열려요. (기본값 모드로 동작)"
