# Atelier AI

그림 생성 전용 데스크톱 도구. 초보자도 바로 쓸 수 있는 직관적인 UI를 지향한다.

-  완전 로컬 동작
- Linux + macOS + Windows (Tauri)
- Anima 계열 모델 특화
- Danbooru식 태그 우선 UX
- 모든 설정에 한국어 설명 + 추천값 표시

## 구조

```
atelier-ai/
├── backend/        # Python + FastAPI (생성 엔진)
├── frontend/       # Svelte 5 + Vite 웹 UI
├── src-tauri/      # Tauri 데스크톱 래퍼
└── docs/           # 기획 스펙
```

## 실행 방법 (빌드 없이)

Tauri로 묶지 않고 바로 실행할 수 있다. 터미널 두 개를 연다.

**1. 백엔드** (http://127.0.0.1:8079)

```bash
cd backend
uv sync
uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
```

모델 파일은 `~/.atelier-ai/models/` (또는 `ATELIER_MODELS_DIR`)에 넣는다.
체크포인트(`.safetensors`)가 없어도 UI는 기본값 모드로 열린다.

**2. 프론트엔드** (http://localhost:5173)

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 http://localhost:5173 을 열면 된다.
백엔드가 꺼져 있어도 설정 패널은 기본값으로 동작한다.

## 데스크톱 앱으로 빌드

Tauri 네이티브 빌드(`.deb` / `.dmg` / `.msi`)는 개발 머신에서 직접 한다.
순서와 필요 패키지는 `src-tauri/README.md`에 정리되어 있다.

## 라이선스

MIT
