"""업스케일 API.

- POST /api/upscale: {image: base64 또는 결과 파일명, strength}
  → RealESRGAN(4x) 업스케일 후 output dir에 저장하고 URL 반환
  (generate.py의 결과 서빙 패턴: /api/results/{filename}).

모델 파일은 첫 사용 시 자동 다운로드(18MB)되므로 요청이 수 분 걸릴 수 있다.
"""

from __future__ import annotations

import base64
import binascii
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.api.routes import history as history_module
from atelier.engine import upscale as upscale_engine

router = APIRouter()

# 업로드 크기 상한 (50MB). generate.py와 동일.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UpscaleRequest(BaseModel):
    image: str  # base64 인코딩 이미지 또는 결과 파일명 (output dir 안)
    strength: float = 0.5  # 0.5~1.0, 1=풀 업스케일, 0.5=절반만 적용


def _output_dir() -> str:
    """결과 저장 폴더. generate.py의 get_pipeline() 폴백과 동일 위치."""
    d = os.path.join(history_module.data_dir(), "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def _looks_like_filename(s: str) -> bool:
    """base64가 아니라 결과 파일명인지 판정."""
    return (
        s.endswith((".png", ".jpg", ".jpeg", ".webp"))
        and "/" not in s
        and "\\" not in s
    )


def _resolve_image(raw: str) -> bytes:
    """base64 또는 결과 파일명 → 이미지 바이트."""
    if not raw:
        raise HTTPException(
            status_code=422, detail="업스케일할 이미지가 필요해요.")
    if _looks_like_filename(raw):
        safe = os.path.basename(raw)
        if not safe or safe != raw:
            raise HTTPException(
                status_code=400, detail="잘못된 파일 이름이에요.")
        path = os.path.join(_output_dir(), safe)
        if not os.path.isfile(path):
            raise HTTPException(
                status_code=404, detail="원본 결과 파일을 찾을 수 없어요.")
        with open(path, "rb") as f:
            return f.read()
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422, detail="이미지 데이터가 올바르지 않아요.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지가 너무 커요. "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB 이하로 올려주세요.",
        )
    if not data:
        raise HTTPException(
            status_code=422, detail="업스케일할 이미지가 필요해요.")
    return data


@router.post("/api/upscale")
def upscale(req: UpscaleRequest):
    """이미지를 4x 업스케일한다. 완료 후 /api/results/ URL을 반환."""
    if not (upscale_engine.MIN_STRENGTH
            <= req.strength <= upscale_engine.MAX_STRENGTH):
        raise HTTPException(
            status_code=422,
            detail=f"업스케일 강도는 {upscale_engine.MIN_STRENGTH}~"
                   f"{upscale_engine.MAX_STRENGTH} 사이여야 해요.",
        )
    image_bytes = _resolve_image(req.image)
    try:
        out_bytes = upscale_engine.upscale_image(
            image_bytes, strength=req.strength)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # GPU 없음 → 503 (pipeline.py _default_device 패턴)
        raise HTTPException(status_code=503, detail=str(e))

    out_dir = _output_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uniq = uuid.uuid4().hex[:6]
    name = f"{stamp}-upscale-s{req.strength}-{uniq}.png"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(out_bytes)

    try:
        history_module.append_record(
            history_module.build_record(
                f"upscale-{uniq}", "upscale",
                {"strength": req.strength}, ["/api/results/" + name], None,
            )
        )
    except Exception as e:  # noqa: BLE001 - 기록 실패가 업스케일을 망치면 안 됨
        print(f"[atelier] 작업내역 저장 실패: {e}")

    return {"url": "/api/results/" + name}
