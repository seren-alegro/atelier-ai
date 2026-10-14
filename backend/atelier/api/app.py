"""FastAPI 앱."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atelier.api.routes.generate import router as generate_router
from atelier.api.routes.history import router as history_router
from atelier.api.routes.models import router as models_router
from atelier.api.routes.pose import router as pose_router
from atelier.api.routes.tags import router as tags_router
from atelier.api.routes.train import router as train_router
from atelier.api.routes.upscale import router as upscale_router
from atelier.settings import SETTINGS, defaults


def create_app() -> FastAPI:
    app = FastAPI(title="Atelier AI", version="0.1.0")

    # Tauri 웹뷰·Vite dev 등 다른 오리진에서 127.0.0.1:8079로 fetch할 때
    # 브라우저가 응답을 막지 않도록. (로컬 루프백 전용 서버라 * 허용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(generate_router)
    app.include_router(history_router)
    app.include_router(models_router)
    app.include_router(pose_router)
    app.include_router(tags_router)
    app.include_router(train_router)
    app.include_router(upscale_router)

    @app.get("/api/settings")
    def list_settings():
        """설정 목록: 한국어 설명 + 추천값 포함."""
        return [
            {
                "key": s.key,
                "label": s.label,
                "description": s.description,
                "recommended": s.recommended,
                "kind": s.kind,
                "min": s.min,
                "max": s.max,
                "options": s.options,
                "unit": s.unit,
            }
            for s in SETTINGS
        ]

    @app.get("/api/settings/defaults")
    def setting_defaults():
        return defaults()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    return app


app = create_app()
