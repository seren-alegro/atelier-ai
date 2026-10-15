"""프롬프트 즐겨찾기 + 네거티브 프리셋 API.

- GET/POST/DELETE /api/prompt-favorites
- GET/POST/DELETE /api/negative-presets (내장 3개는 builtin=true, 삭제 불가)

저장: ATELIER_DATA_DIR 환경 변수 우선, 없으면 ~/.atelier-ai/
(history.py의 data_dir() 패턴 재사용)
- prompt_favorites.json: [{id, name, prompt, created_at}]
- negative_presets.json: [{id, name, text, builtin}]
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.api.routes.history import data_dir

router = APIRouter()

# JSON 파일 동시 쓰기 방지
_presets_lock = threading.Lock()

#: 내장 네거티브 프리셋 3개 (삭제 불가)
BUILTIN_NEGATIVE_PRESETS = [
    {
        "id": "builtin-basic",
        "name": "기본 정리",
        "text": "lowres, bad anatomy, bad hands, missing fingers, "
                "extra fingers, fewer fingers, cropped, worst quality, "
                "jpeg artifacts, signature, watermark, username, blurry",
        "builtin": True,
    },
    {
        "id": "builtin-hands",
        "name": "손 보정",
        "text": "bad hands, missing fingers, extra fingers, fewer fingers, "
                "mutated hands, poorly drawn hands, malformed hands",
        "builtin": True,
    },
    {
        "id": "builtin-no-text",
        "name": "글자 제거",
        "text": "text, watermark, signature, username, logo, speech bubble",
        "builtin": True,
    },
]


def _favorites_path() -> str:
    return os.path.join(data_dir(), "prompt_favorites.json")


def _negpresets_path() -> str:
    return os.path.join(data_dir(), "negative_presets.json")


def _load(path: str) -> list[dict]:
    """JSON 리스트를 읽는다. 파일이 없거나 깨졌으면 빈 리스트."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save(path: str, items: list[dict]) -> None:
    with _presets_lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


class FavoriteCreate(BaseModel):
    name: str
    prompt: str


class NegativePresetCreate(BaseModel):
    name: str
    text: str


# -- 프롬프트 즐겨찾기 ----------------------------------------------------


@router.get("/api/prompt-favorites")
def list_favorites():
    """즐겨찾기 목록."""
    return _load(_favorites_path())


@router.post("/api/prompt-favorites", status_code=201)
def create_favorite(req: FavoriteCreate):
    """즐겨찾기 추가. name/prompt가 비어 있으면 422."""
    name = req.name.strip()
    prompt = req.prompt.strip()
    if not name or not prompt:
        raise HTTPException(
            status_code=422,
            detail="이름과 프롬프트를 모두 입력해주세요.",
        )
    items = _load(_favorites_path())
    item = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "prompt": prompt,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items.append(item)
    _save(_favorites_path(), items)
    return item


@router.delete("/api/prompt-favorites/{fav_id}")
def delete_favorite(fav_id: str):
    """즐겨찾기 삭제. 없으면 404."""
    path = _favorites_path()
    items = _load(path)
    kept = [i for i in items if i.get("id") != fav_id]
    if len(kept) == len(items):
        raise HTTPException(
            status_code=404, detail="즐겨찾기를 찾을 수 없어요."
        )
    _save(path, kept)
    return {"deleted": fav_id}


# -- 네거티브 프리셋 ------------------------------------------------------


@router.get("/api/negative-presets")
def list_negative_presets():
    """내장 3개 + 사용자 프리셋 전체."""
    return [*BUILTIN_NEGATIVE_PRESETS, *_load(_negpresets_path())]


@router.post("/api/negative-presets", status_code=201)
def create_negative_preset(req: NegativePresetCreate):
    """사용자 프리셋 추가. name/text가 비어 있으면 422."""
    name = req.name.strip()
    text = req.text.strip()
    if not name or not text:
        raise HTTPException(
            status_code=422,
            detail="이름과 내용을 모두 입력해주세요.",
        )
    items = _load(_negpresets_path())
    item = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "text": text,
        "builtin": False,
    }
    items.append(item)
    _save(_negpresets_path(), items)
    return item


@router.delete("/api/negative-presets/{preset_id}")
def delete_negative_preset(preset_id: str):
    """프리셋 삭제. 내장이면 400, 없으면 404."""
    if any(p["id"] == preset_id for p in BUILTIN_NEGATIVE_PRESETS):
        raise HTTPException(
            status_code=400, detail="내장 프리셋은 삭제할 수 없어요."
        )
    path = _negpresets_path()
    items = _load(path)
    kept = [i for i in items if i.get("id") != preset_id]
    if len(kept) == len(items):
        raise HTTPException(
            status_code=404, detail="프리셋을 찾을 수 없어요."
        )
    _save(path, kept)
    return {"deleted": preset_id}
