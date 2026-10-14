# Atelier AI — Tauri 데스크톱 셸

`src-tauri/`는 Atelier AI의 데스크톱 앱 껍데기다. 웹 UI(`frontend/dist/`)를
네이티브 창에 띄우고, Python 백엔드(FastAPI)를 자식 프로세스로 함께 실행한다.

## 백엔드 실행 전략 (MVP)

앱 시작 시 `src/main.rs`의 `spawn_backend()`가 백엔드를 찾는다. 우선순위:

1. **`ATELIER_BACKEND_CMD` 환경 변수** — 실행할 전체 명령을 직접 지정.
   예: `ATELIER_BACKEND_CMD="/opt/atelier/run-backend.sh"`
   작업 디렉토리는 `ATELIER_BACKEND_DIR`로 지정할 수 있다.
2. **사이드카 바이너리** — 실행 파일 옆의 `atelier-backend`
   (향후 PyInstaller 등으로 얼린 배포용 바이너리 자리.
   인자 없이 실행하면 `127.0.0.1:8079`에서 서빙한다고 가정한다.)
3. **개발 레이아웃** — 실행 파일 기준 `../../backend`에 `pyproject.toml`이 있으면
   OS에 맞는 Python으로 `uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079` 실행.
   (Windows: `py -3` → `python` → `python3`, Unix: `python3` → `python` 순으로 탐색.)
   (`backend/`에서 `uv sync`로 의존성을 먼저 설치해야 한다.)

어떤 방법으로도 백엔드를 띄우지 못하면 앱은 그대로 실행되고,
프론트엔드는 내장 기본값(폴백) 모드로 동작한다.
헤더의 연결 상태 배지로 백엔드 연결 여부를 확인할 수 있다.

앱 종료 시 백엔드 자식 프로세스를 함께 종료(`kill`)한다.

## 프론트엔드-백엔드 주소

Tauri 안에서는 백엔드가 같은 머신에서 뜨므로
`http://127.0.0.1:8079`를 그대로 사용한다 (`frontend/src/lib/api.js`).

## 빌드

```sh
cd src-tauri
cargo build        # 디버그 빌드
```

`tauri build`로 설치 패키지를 만들 때는 Tauri CLI가 필요하다.

### 디버그 빌드에서는 dev 서버가 필요하다

`cargo build`(디버그)는 개발 모드라서, Tauri가 `tauri.conf.json`의
`devUrl`(http://localhost:5173)로 프론트엔드를 불러온다.
그래서 앱을 실행하기 전에 프론트 dev 서버를 먼저 띄워야 한다.
(안 띄우면 "Could not connect to localhost"가 뜬다.)

```sh
# 터미널 1: 프론트 dev 서버
cd frontend && npm run dev
# 터미널 2: 백엔드 (앱이 자동 실행을 시도하지만, 직접 띄우는 게 확실하다)
cd backend && uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
# 터미널 3: 앱
cd src-tauri && ./target/debug/atelier-ai
```

### 독립 실행 바이너리가 필요하면

`--release`로 빌드하면 프론트엔드(`frontend/dist`)가 바이너리에 내장된다.
배포용은 이쪽이다.

```sh
cd frontend && npm run build
cd ../src-tauri && cargo build --release
./target/release/atelier-ai
```

### 필요한 시스템 패키지 (OS별 설치 명령어)

Tauri 2 앱을 빌드하려면 OS마다 준비물이 조금씩 다르다.
아래 명령어를 터미널에 한 줄씩 붙여넣으면 된다.

**공통: Rust 설치** (아직 없다면, Windows 제외)

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Ubuntu / Debian**

```sh
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

**Fedora**

```sh
sudo dnf check-update
sudo dnf install webkit2gtk4.1-devel openssl-devel curl wget file \
  libappindicator-gtk3-devel librsvg2-devel
sudo dnf group install "c-development" "development-tools"
```

**Arch / CachyOS / Manjaro**

```sh
sudo pacman -Syu
sudo pacman -S --needed webkit2gtk-4.1 base-devel curl wget file openssl \
  appmenu-gtk-module libappindicator-gtk3 librsvg
```

**macOS**

Xcode Command Line Tools만 설치하면 된다.

```sh
xcode-select --install
```

**Windows**

1. [rustup-init.exe](https://rustup.rs/)를 받아 Rust를 설치한다.
2. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)를 설치한다.
   설치할 때 "C++를 사용한 데스크톱 개발" 워크로드를 꼭 체크한다.
3. WebView2는 Windows 10/11에 보통 이미 들어 있다.
   없다면 [WebView2 런타임](https://developer.microsoft.com/ko-kr/microsoft-edge/webview2/)을 설치한다.

**BSD (FreeBSD 등)**

솔직히 말하면 Tauri가 BSD를 공식 지원하지 않는다.
FreeBSD ports에 `webkit2gtk`가 있긴 하지만, Tauri 자체가 BSD 빌드를
테스트하지 않아 중간에 막힐 가능성이 높다.
꼭 시도해보고 싶다면 Linux용 의존성을 참고해서 ports에서 맞춰보고,
안 되면 Linux 가상머신이나 WSL2에서 빌드하는 쪽이 빠르다.

## 알려진 제한

- 백엔드는 `torch`/`diffusers` 임포트 때문에 시작이 느리다 (수십 초).
  프론트엔드가 먼저 뜨면 "기본값 사용 중"으로 표시될 수 있고,
  백엔드가 올라온 뒤 새로고침하면 연결된다.
- 활성 모델 선택은 현재 인메모리라 앱(백엔드) 재시작 시 초기화된다.
