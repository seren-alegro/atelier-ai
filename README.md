# Atelier AI

[English](README.en.md)

그림 생성 전용 데스크톱 도구. 초보자도 바로 쓸 수 있는 직관적인 UI를 지향한다.

- 완전 로컬 동작 (인터넷 없이도 실행)
- Linux + macOS + Windows (Tauri)
- Anima 계열 모델 특화
- Danbooru식 태그 우선 UX
- 모든 설정에 한국어 설명 + 추천값 표시

## 설치 (처음 설치하는 분)

### 1. 준비물 설치

| 준비물 | 설명 | 받는 곳 |
|---|---|---|
| Python 3.11 이상 | 백엔드 실행용 | https://www.python.org/downloads/ |
| Node.js 18 이상 | 화면 빌드용 (npm 포함) | https://nodejs.org/ |
| Rust | 앱 빌드용 (`cargo` 명령) | https://rustup.rs/ |

OS별 참고:

- **Ubuntu/Debian**: `sudo apt install python3 python3-venv` 후 Node.js·Rust는 위 링크에서
- **Arch/CachyOS**: `sudo pacman -S python nodejs npm` 후 Rust는 위 링크에서
- **macOS**: `xcode-select --install` 후 [Homebrew](https://brew.sh/)로 `brew install python node`
- **Windows**: 위 링크에서 각각 설치 + [C++ Build Tools](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/) ("C++를 사용한 데스크톱 개발" 체크)

### 2. 원클릭 설치

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

스크립트가 백엔드 설치 → 화면 빌드 → 데스크톱 앱 빌드를 순서대로 해준다.
처음엔 의존성 다운로드 때문에 수 분 걸린다.

### 3. 모델 파일 넣기

모델 폴더: `~/.atelier-ai/models/` (Windows: `%USERPROFILE%\.atelier-ai\models\`)

- 체크포인트(`.safetensors` / `.ckpt`) → 폴더 바로 아래
- ControlNet → `controlnet/` 하위 폴더
- LoRA → `lora/` 하위 폴더

모델이 없어도 앱은 열린다. (기본값 모드로 동작하며, 모델 탭에서 상태 확인 가능)

### 4. 실행

Linux / macOS:
```bash
./src-tauri/target/release/atelier-ai
```

Windows:
```powershell
.\src-tauri\target\release\atelier-ai.exe
```

앱을 켜면 백엔드가 자동으로 따라 켜진다. 끌 때는 앱만 끄면 된다.

## 개발자용

### 구조

```
atelier-ai/
├── backend/        # Python + FastAPI (생성 엔진)
├── frontend/       # Svelte 5 + Vite 웹 UI
├── src-tauri/      # Tauri 데스크톱 래퍼
├── install.sh      # 원클릭 설치 (Linux/macOS)
├── install.ps1     # 원클릭 설치 (Windows)
└── docs/           # 기획 스펙
```

### 빌드 없이 실행 (프론트 개발용)

터미널 두 개를 연다.

**1. 백엔드** (http://127.0.0.1:8079)

```bash
cd backend
uv sync
uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
```

**2. 프론트엔드** (http://localhost:5173)

```bash
cd frontend
npm install
npm run dev
```

### 테스트

```bash
cd backend && uv run pytest tests/ -q
cd frontend && npm test && npm run build
```

## 릴리즈 (자동 업데이트)

태그를 푸시하면 GitHub Actions가 Linux/Windows/macOS용 설치 파일을 빌드하고
릴리즈를 발행합니다. 앱은 시작 시 새 버전을 확인하고, 설정 탭에서
다운로드·설치·재시작할 수 있습니다. (Tauri 업데이터 + 서명)

### 처음 한 번만: 서명 키 등록

업데이터 서명용 비밀키는 저장소에 올리지 않습니다.
키 파일(`updater.key`)을 받은 뒤 GitHub 저장소의
Settings → Secrets and variables → Actions에 등록하세요.

| Secret | 값 |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | `updater.key` 파일 전체 내용 |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | 빈 값 (키에 비밀번호 없음) |

```sh
cat updater.key | base64 -w0 | gh secret set TAURI_SIGNING_PRIVATE_KEY
printf '' | gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD
```

### 릴리즈 하기

```sh
./scripts/bump-version.sh 0.2.0   # 세 곳의 버전을 한 번에 올림
git add -A && git commit -m "v0.2.0"
git tag v0.2.0 && git push origin main v0.2.0
```

워크플로우가 백엔드 동결(PyInstaller) → Tauri 빌드·서명 → draft 릴리즈 생성 →
`latest.json` 검증 → 공개 순서로 진행합니다. 공개 후에 앱들이 업데이트를 감지합니다.

### 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `ATELIER_MODELS_DIR` | 모델 폴더 경로 | `~/.atelier-ai/models` |
| `ATELIER_DATA_DIR` | 작업내역 저장 경로 | `~/.atelier-ai` |
| `ATELIER_BACKEND_CMD` | 백엔드 실행 명령 직접 지정 (Tauri) | 자동 탐색 |
| `ATELIER_BACKEND_DIR` | 백엔드 작업 디렉토리 (Tauri) | 자동 탐색 |

## 문제 해결

| 증상 | 대처 |
|---|---|
| UI에 "기본값 사용 중" 배지가 뜬다 | 백엔드가 아직 켜지는 중일 수 있다. 최대 3분까지 기다려보고, 안 되면 배지를 클릭해서 재연결 |
| "모델이 로드되지 않았어요" (503) | 모델 폴더에 체크포인트(`.safetensors`)가 있는지 확인. 모델 탭에서도 확인 가능 |
| "Python을 찾지 못했습니다" | Python 3.11+ 설치. Windows는 python.org에서 설치 후 `py -3 --version` 확인 |
| 포트 8079가 이미 사용 중이라는 오류 | 이미 백엔드가 돌고 있다. 기존 프로세스를 끄거나 그대로 쓴다 |
| 생성 중 오류 / 화면이 멈춘다 | VRAM 부족일 수 있다. 해상도나 배치를 낮춰본다 |
| `./install.sh`가 Tauri 의존성에서 멈춘다 | `src-tauri/README.md`의 OS별 시스템 패키지 목록 확인 |
| `npm run dev`가 안 된다 | Node.js 18+ 확인 후 `rm -rf node_modules && npm install` |

## 요구 사항

NVIDIA GPU(CUDA) 또는 Apple Silicon(Metal/MPS)이 필요합니다. CPU 전용 환경은
지원하지 않으며, 생성 요청 시 오류가 발생합니다.

## 라이선스

MIT
