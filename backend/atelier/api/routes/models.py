"""모델 매니저 API.

- GET /api/models: 모델 디렉토리 스캔 (체크포인트 + LoRA)
- POST /api/models/active: 활성 체크포인트 전환 (파이프라인 재로드)
- POST /api/models/refresh: 재스캔
- DELETE /api/models/{name}: 모델 파일 삭제

모델 디렉토리: ATELIER_MODELS_DIR 환경 변수 우선, 없으면 ~/.atelier-ai/models.
활성 모델 선택은 active_model.json에 저장되어 서버 재시작 후에도 유지된다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.engine.pipeline import Pipeline

router = APIRouter()

CHECKPOINT_EXTS = (".safetensors", ".ckpt")
LORA_DIR_NAMES = ("lora", "loras")
CONTROLNET_DIR_NAMES = ("controlnet", "controlnets")
VAE_DIR_NAMES = ("vae", "vaes")
TEXT_ENCODER_DIR_NAMES = ("text_encoder", "text_encoders")
#: 팩 판정에서 제외할 특수 폴더 (이 안의 파일은 kind별로만 분류)
_SPECIAL_DIR_NAMES = (LORA_DIR_NAMES + CONTROLNET_DIR_NAMES
                      + VAE_DIR_NAMES + TEXT_ENCODER_DIR_NAMES)

_active_name: str | None = None


def get_active_name() -> str | None:
    """현재 활성 체크포인트 이름. 작업내역 기록용."""
    return _active_name


def _active_file() -> str:
    """활성 모델 저장 파일. 작업내역과 같은 데이터 디렉토리 사용."""
    from atelier.api.routes import history as history_module
    return os.path.join(history_module.data_dir(), "active_model.json")


def _load_saved_active() -> str | None:
    try:
        with open(_active_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def _save_active(name: str | None) -> None:
    try:
        with open(_active_file(), "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
    except OSError:
        pass  # 저장 실패해도 동작은 계속


def get_saved_active_path() -> str | None:
    """저장된 활성 모델의 절대 경로. 파일/팩이 없으면 None."""
    name = _load_saved_active()
    if not name:
        return None
    for resolver in (_resolve, _resolve_dir):
        try:
            return resolver(name)
        except HTTPException:
            continue
    return None


def models_dir() -> str:
    """모델 디렉토리 경로. 없으면 만들어 둔다."""
    d = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    return d


def _scan() -> list[dict]:
    """모델 디렉토리를 스캔해 체크포인트/LoRA/Anima 3.8b 팩 목록 반환."""
    from atelier.engine.anima38 import looks_like_pack_dir, pack_status

    base = models_dir()
    items: list[dict] = []
    if not os.path.isdir(base):
        return items
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rel_root = (os.path.relpath(root, base).replace(os.sep, "/")
                    if root != base else "")
        top_dir = rel_root.split("/")[0].lower() if rel_root else ""
        # Anima 3.8b 팩 디렉토리: 파일 목록에 넣지 않고 팩 항목으로만 표시
        # (text_encoders/·vae/ 등 특수 폴더는 팩 판정에서 제외)
        if (root != base and top_dir not in _SPECIAL_DIR_NAMES
                and looks_like_pack_dir(root)):
            rel = os.path.relpath(root, base).replace(os.sep, "/")
            st = pack_status(root)
            total = 0
            newest = 0.0
            for f in files:
                if f.startswith("."):
                    continue
                fp = os.path.join(root, f)
                try:
                    fst = os.stat(fp)
                except OSError:
                    continue
                total += fst.st_size
                newest = max(newest, fst.st_mtime)
            items.append(
                {
                    "name": rel,
                    "kind": "anima38pack",
                    "size_mb": round(total / (1024 * 1024), 1),
                    "modified": datetime.fromtimestamp(newest).isoformat(
                        timespec="seconds"
                    ) if newest else "",
                    "pack_status": st,
                }
            )
            dirs[:] = []  # 팩 내부는 더 내려가지 않음
            continue
        for f in sorted(files):
            if f.startswith("."):
                continue
            if os.path.splitext(f)[1].lower() not in CHECKPOINT_EXTS:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            parts = rel.split("/")
            is_controlnet = any(
                p.lower() in CONTROLNET_DIR_NAMES for p in parts[:-1]
            )
            is_vae = any(
                p.lower() in VAE_DIR_NAMES for p in parts[:-1]
            )
            is_text_encoder = any(
                p.lower() in TEXT_ENCODER_DIR_NAMES for p in parts[:-1]
            )
            is_lora = any(
                p.lower() in LORA_DIR_NAMES for p in parts[:-1]
            ) or "lora" in f.lower()
            if is_controlnet:
                kind = "controlnet"
            elif is_vae:
                kind = "vae"
            elif is_text_encoder:
                kind = "text_encoder"
            elif is_lora:
                kind = "lora"
            else:
                kind = "checkpoint"
            st = os.stat(full)
            items.append(
                {
                    "name": rel,
                    "kind": kind,
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


def _resolve_dir(name: str) -> str:
    """팩 디렉토리 이름을 절대 경로로 바꾼다. 경로 조작 시 400."""
    if not name or name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="잘못된 모델 이름이에요.")
    base = os.path.realpath(models_dir())
    full = os.path.realpath(os.path.join(base, name.replace("/", os.sep)))
    if full != base and not full.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="잘못된 모델 이름이에요.")
    if not os.path.isdir(full):
        raise HTTPException(
            status_code=404, detail="모델 팩을 찾을 수 없어요."
        )
    return full


def is_anima38_pack_dir(path: str) -> bool:
    """절대 경로가 3.8b 팩 디렉토리인지. generate.py에서 사용."""
    from atelier.engine.anima38 import is_anima38_pack_dir as _is_pack
    return _is_pack(path)


# 모듈 로드 시 저장된 활성 모델 복원 (파일/팩이 실제로 있을 때만)
_saved_active = _load_saved_active()
if _saved_active:
    try:
        try:
            _resolve(_saved_active)
        except HTTPException:
            _resolve_dir(_saved_active)
        _active_name = _saved_active
    except HTTPException:
        pass
del _saved_active


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
        (m for m in items
         if m["name"] == req.name and m["kind"] in ("checkpoint", "anima38pack")),
        None,
    )
    if match is None:
        raise HTTPException(
            status_code=404, detail="체크포인트를 찾을 수 없어요."
        )

    try:
        current = generate.get_pipeline()
    except HTTPException:
        current = None
    output_dir = getattr(current, "output_dir", "outputs")

    if match["kind"] == "anima38pack":
        from atelier.engine.anima38 import Anima38Pipeline
        path = _resolve_dir(req.name)
        new_pipe = Anima38Pipeline(pack_dir=path, output_dir=output_dir)
    else:
        path = _resolve(req.name)
        # Anima DiT 단일 파일 (2b 등)인지 확인
        from atelier.engine import anima38 as a38
        if a38.is_anima_dit_file(path):
            new_pipe = a38.Anima2BPipeline(model_path=path,
                                           output_dir=output_dir)
        else:
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
    _save_active(req.name)
    return {"active": _active_name}


def _te_file() -> str:
    """선택된 텍스트 인코더 저장 파일. active_model.json과 같은 폴더."""
    from atelier.api.routes import history as history_module
    return os.path.join(history_module.data_dir(), "text_encoder.json")


def get_selected_text_encoder() -> str | None:
    """사용자가 선택한 텍스트 인코더 (모델 목록의 name). 없으면 None."""
    try:
        with open(_te_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def _save_text_encoder(name: str | None) -> None:
    try:
        with open(_te_file(), "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
    except OSError:
        pass  # 저장 실패해도 동작은 계속


@router.get("/api/models/text-encoder")
def get_text_encoder():
    """선택된 텍스트 인코더 조회."""
    return {"name": get_selected_text_encoder()}


class TextEncoderRequest(BaseModel):
    name: str = ""  # ""이면 선택 해제


@router.post("/api/models/text-encoder")
def set_text_encoder(req: TextEncoderRequest):
    """텍스트 인코더 선택. Anima 3.8b 팩에 TE 파일이 없을 때 대신 쓴다."""
    name = (req.name or "").strip()
    if name:
        items = _scan()
        match = next(
            (m for m in items
             if m["name"] == name and m["kind"] == "text_encoder"),
            None,
        )
        if match is None:
            raise HTTPException(
                status_code=404,
                detail="텍스트 인코더 파일을 찾을 수 없어요.",
            )
        _save_text_encoder(name)
    else:
        _save_text_encoder(None)
    return {"name": get_selected_text_encoder()}


def _vae_file() -> str:
    """선택된 VAE 저장 파일."""
    from atelier.api.routes import history as history_module
    return os.path.join(history_module.data_dir(), "vae.json")


def get_selected_vae() -> str | None:
    """사용자가 선택한 VAE (모델 목록의 name). 없으면 None."""
    try:
        with open(_vae_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def get_selected_vae_path() -> str | None:
    """선택된 VAE의 절대 경로. 파일이 없으면 None."""
    name = get_selected_vae()
    if not name:
        return None
    try:
        return _resolve(name)
    except HTTPException:
        return None


def _save_vae(name: str | None) -> None:
    try:
        with open(_vae_file(), "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
    except OSError:
        pass  # 저장 실패해도 동작은 계속


@router.get("/api/models/vae")
def get_vae():
    """선택된 VAE 조회."""
    return {"name": get_selected_vae()}


class VaeRequest(BaseModel):
    name: str = ""  # ""이면 선택 해제


@router.post("/api/models/vae")
def set_vae(req: VaeRequest):
    """VAE 선택. Anima 3.8b 팩의 VAE 대신 쓴다."""
    name = (req.name or "").strip()
    if name:
        items = _scan()
        match = next(
            (m for m in items
             if m["name"] == name and m["kind"] == "vae"),
            None,
        )
        if match is None:
            raise HTTPException(
                status_code=404,
                detail="VAE 파일을 찾을 수 없어요.",
            )
        _save_vae(name)
    else:
        _save_vae(None)
    return {"name": get_selected_vae()}


@router.delete("/api/models/{name:path}")
def delete_model(name: str):
    """모델 파일 삭제. 사용 중인 활성 모델은 삭제할 수 없다."""
    if name == _active_name:
        raise HTTPException(
            status_code=409,
            detail="사용 중인 모델은 삭제할 수 없어요. "
                   "먼저 다른 모델로 전환하세요.",
        )
    if os.path.splitext(name)[1].lower() not in CHECKPOINT_EXTS:
        raise HTTPException(
            status_code=400, detail="모델 파일만 삭제할 수 있어요.")
    path = _resolve(name)
    os.remove(path)
    return {"deleted": name}
