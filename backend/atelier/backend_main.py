"""Frozen 백엔드 진입점 (PyInstaller용).

릴리즈 번들에 포함되는 얼린 백엔드는 인자 없이 실행하면
127.0.0.1:8079 에서 서빙한다 (Tauri 셸의 spawn_backend 가정).
"""

from __future__ import annotations

import uvicorn

from atelier.api.app import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8079, log_level="info")


if __name__ == "__main__":
    main()
