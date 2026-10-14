#!/usr/bin/env bash
# 백엔드를 PyInstaller로 얼려 src-tauri/backend-dist/atelier-backend/ 에 넣는다.
# 릴리즈 워크플로우와 로컬 배포용 빌드에서 함께 쓴다.
#
# 환경 변수:
#   TORCH_INDEX_URL — 설정하면 torch를 해당 인덱스에서 강제 재설치
#                     (예: https://download.pytorch.org/whl/cu128)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 필요해요: https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "==> 백엔드 의존성 설치 (uv sync)"
uv sync --frozen

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  echo "==> torch를 $TORCH_INDEX_URL 에서 재설치"
  uv pip install --force-reinstall torch --index-url "$TORCH_INDEX_URL"
fi

echo "==> PyInstaller 설치"
uv pip install pyinstaller

echo "==> 백엔드 동결 중... (수 분 걸려요)"
uv run pyinstaller atelier-backend.spec --noconfirm --distpath dist

OUT="$ROOT/src-tauri/backend-dist/atelier-backend"
rm -rf "$OUT"
mkdir -p "$ROOT/src-tauri/backend-dist"
cp -r dist/atelier-backend "$OUT"

EXE="$OUT/atelier-backend"
case "${OS:-}$(uname -s)" in
  *Windows*|*MINGW*|*MSYS*) EXE="$EXE.exe" ;;
esac
if [ ! -f "$EXE" ]; then
  echo "동결 실패: $OUT 에 실행 파일이 없어요" >&2
  exit 1
fi

echo "==> 완료: $OUT"
du -sh "$OUT"
