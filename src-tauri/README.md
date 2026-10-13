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
   `python3 -m uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079` 실행.
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

### 필요한 시스템 패키지

Linux에서 Tauri 2 앱을 빌드하려면 WebKit/GTK 개발 패키지가 필요하다.
설치하지 않으면 `webkit2gtk-sys` 컴파일 단계에서 실패한다.

- **Ubuntu/Debian**: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`,
  `libayatana-appindicator3-dev`, `librsvg2-dev`, `libssl-dev`, `build-essential`
- **Fedora**: `webkit2gtk4.1-devel`, `gtk3-devel`,
  `libayatana-appindicator-gtk3-devel`, `librsvg2-devel`, `openssl-devel`
- **macOS**: Xcode Command Line Tools만 있으면 된다.

## 알려진 제한

- 백엔드는 `torch`/`diffusers` 임포트 때문에 시작이 느리다 (수십 초).
  프론트엔드가 먼저 뜨면 "기본값 사용 중"으로 표시될 수 있고,
  백엔드가 올라온 뒤 새로고침하면 연결된다.
- 활성 모델 선택은 현재 인메모리라 앱(백엔드) 재시작 시 초기화된다.
