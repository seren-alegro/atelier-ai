# Atelier AI

그림 생성 전용 데스크톱 도구. 초보자도 바로 쓸 수 있는 직관적인 UI를 지향한다.

- 이용권 없음, 완전 로컬 동작
- Linux + macOS 네이티브 (Tauri)
- Anima 계열 모델 특화
- Danbooru식 태그 우선 UX
- 모든 설정에 한국어 설명 + 추천값 표시

## 구조

```
atelier-ai/
├── backend/        # Python + FastAPI (생성 엔진)
├── frontend/       # 웹 UI (프레임워크 미정)
├── src-tauri/      # Tauri 데스크톱 래퍼
└── docs/           # 기획 스펙
```

## 개발 시작

```bash
cd backend
uv sync
uv run uvicorn atelier.api.app:app --host 127.0.0.1 --port 8079
```

## 라이선스

MIT
