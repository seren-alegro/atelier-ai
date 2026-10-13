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

## 문제 해결

| 증상 | 대처 |
|---|---|
| UI에 "기본값 사용 중" 배지가 뜬다 | 백엔드가 꺼져 있다. 터미널에서 백엔드를 먼저 실행한다 |
| "모델이 로드되지 않았어요" (503) | `~/.atelier-ai/models/`에 체크포인트(`.safetensors`)가 있는지 확인. 모델 탭에서도 확인 가능 |
| "Python을 찾지 못했습니다" | Python 3.10+ 설치. Windows는 python.org에서 설치 후 `py -3 --version` 확인 |
| 포트 8079가 이미 사용 중이라는 오류 | 이미 백엔드가 돌고 있다. 기존 프로세스를 끄거나 그대로 쓴다 |
| 생성 중 오류 / 화면이 멈춘다 | VRAM 부족일 수 있다. 해상도나 배치를 낮춰본다 |
| `npm run dev`가 안 된다 | Node.js 18+ 확인 후 `rm -rf node_modules && npm install` |
| Tauri 빌드가 의존성에서 실패한다 | `src-tauri/README.md`의 시스템 패키지 목록 확인 |

## 라이선스

MIT
