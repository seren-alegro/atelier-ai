"""FastAPI 앱."""

from fastapi import FastAPI

from atelier.api.routes.generate import router as generate_router
from atelier.settings import SETTINGS, defaults


def create_app() -> FastAPI:
    app = FastAPI(title="Atelier AI", version="0.1.0")

    app.include_router(generate_router)

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
