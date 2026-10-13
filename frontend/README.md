# Atelier AI — 프론트엔드

Svelte 5 + Vite 웹 UI. Tauri 데스크톱 래퍼 안에서 동작한다.

## 구조

```
src/
├── App.svelte              # 탭 레이아웃 (생성 / 인페인트 / 설정)
├── app.css                 # 다크 테마
└── lib/
    ├── api.js              # 백엔드 API 클라이언트 (실패 시 폴백)
    ├── fallbackSettings.js # /api/settings 폴백 (settings.py와 동기화)
    ├── store.svelte.js     # 설정 메타 + 현재값 공유 스토어
    ├── TagInput.svelte     # Danbooru 태그 칩 입력
    ├── SettingField.svelte # kind별 입력 위젯
    ├── SettingsPanel.svelte# /api/settings 동적 렌더링
    ├── GenerateTab.svelte  # 생성 버튼 + 결과 갤러리 (목업)
    └── InpaintTab.svelte   # 인페인트 캔버스 자리
```

## 백엔드 API

- `GET /api/settings` — 설정 목록 (한국어 설명 + 추천값)
- `GET /api/settings/defaults` — 추천값 딕셔너리
- `GET /api/health` — 헬스체크
- `POST /api/generate` — 미구현 (TODO)

백엔드(기본 `http://127.0.0.1:8079`)에 연결 실패해도 폴백 설정으로 UI가 정상 동작한다.

## 개발

```bash
npm install
npm run dev     # http://localhost:5173
npm run build   # dist/ 생성 (Tauri가 읽음)
```

## TODO

- `POST /api/generate` 연동 (GenerateTab.svelte)
- 태그별 한국어 라벨 데이터 (TagInput.svelte)
- 인페인트 캔버스: 이미지 업로드 + 마스크 브러시
