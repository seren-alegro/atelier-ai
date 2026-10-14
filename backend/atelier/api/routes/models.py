"""모델 매니저 API.

- GET /api/models: 모델 디렉토리 스캔 (체크포인트 + LoRA)
- POST /api/models/active: 활성 체크포인트 전환 (파이프라인 재로드)
- POST /api/models/refresh: 재스캔
- DELETE /api/models/{name}: 모델 파일 삭제

모델 디렉토리: ATELIER_MODELS_DIR 환경 변수 우선, 없으면 ~/.atelier-ai/models.
활성 모델 기록은 인메모리 (서버 재시작 시 사라짐, MVP).
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.engine.pipeline import Pipeline

router = APIRouter()

CHECKPOINT_EXTS = (".safetensors", ".ckpt")
LORA_DIR_NAMES = ("lora", "loras")

_active_name: str | None = None


def get_active_name() -> str | None:
    """현재 활성 체크포인트 이름. 작업내역 기록용."""
    return _active_name


def models_dir() -> str:
    """모델 디렉토리 경로. 없으면 만들어 둔다."""
    d = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    return d


def _scan() -> list[dict]:
    """모델 디렉토리를 스캔해 체크포인트/LoRA 목록 반환."""
    base = models_dir()
    items: list[dict] = []
    if not os.path.isdir(base):
        return items
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if f.startswith("."):
                continue
            if os.path.splitext(f)[1].lower() not in CHECKPOINT_EXTS:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            parts = rel.split("/")
            is_lora = any(
                p.lower() in LORA_DIR_NAMES for p in parts[:-1]
            ) or "lora" in f.lower()
            st = os.stat(full)
            items.append(
                {
                    "name": rel,
                    "kind": "lora" if is_lora else "checkpoint",
                    "size_mb": round(st.st_size / (1024 * 1024), 1),
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                }
            )
    return items


def _resolve(name: str) -> str:
    """모델 이름을 디렉토리 안의 절대 경로로 바꾼다. 경로 조작 시 400."""
    if not name or name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="잘못된 모델 이름이에요.")
    base = os.path.realpath(models_dir())
    full = os.path.realpath(os.path.join(base, name.replace("/", os.sep)))
    if full != base and not full.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="잘못된 모델 이름이에요.")
    if not os.path.isfile(full):
        raise HTTPException(
            status_code=404, detail="모델 파일을 찾을 수 없어요."
        )
    return full


@router.get("/api/models")
def list_models():
    """모델 목록 + 활성 체크포인트 + 모델 디렉토리 경로."""
    return {
        "models_dir": models_dir(),
        "active": _active_name,
        "models": _scan(),
    }


@router.post("/api/models/refresh")
def refresh_models():
    """모델 디렉토리 재스캔."""
    return list_models()


class ActiveRequest(BaseModel):
    name: str


@router.post("/api/models/active")
def set_active(req: ActiveRequest):
    """활성 체크포인트 전환. 파이프라인을 새 모델로 재로드한다.

    로드 실패 시 기존 파이프라인을 유지하고 500을 반환한다.
    """
    # generate 모듈과 순환 import를 피하려고 함수 안에서 가져온다.
    from atelier.api.routes import generate

    items = _scan()
    match = next(
        (m for m in items if m["name"] == req.name and m["kind"] == "checkpoint"),
        None,
    )
    if match is None:
        raise HTTPException(
            status_code=404, detail="체크포인트를 찾을 수 없어요."
        )
    path = _resolve(req.name)

    try:
        current = generate.get_pipeline()
    except HTTPException:
        current = None
    output_dir = getattr(current, "output_dir", "outputs")

    new_pipe = Pipeline(model_path=path, output_dir=output_dir)
    try:
        new_pipe.load()
    except Exception as e:  # noqa: BLE001 - 메시지를 그대로 전달
        raise HTTPException(
            status_code=500,
            detail=f"모델 로드에 실패했어요. 기존 모델을 유지합니다: {e}",
        )
    generate.set_pipeline(new_pipe)

    global _active_name
    _active_name = req.name
    return {"active": _active_name}


@router.delete("/api/models/{name:path}")
def delete_model(name: str):
    """모델 파일 삭제. 사용 중인 활성 모델은 삭제할 수 없다."""
    if name == _active_name:
        raise HTTPException(
            status_code=409,
            detail="사용 중인 모델은 삭제할 수 없어요. "
                   "먼저 다른 모델로 전환하세요.",
        )
    path = _resolve(name)
    os.remove(path)
    return {"deleted": name}
