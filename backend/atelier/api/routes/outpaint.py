"""아웃페인팅 API.

- POST /api/outpaint: {image (base64 또는 결과 파일명),
    expand {up, down, left, right} (각 64~512px),
    prompt?, negative_prompt?, denoise_strength (기본 0.6),
    mask_blur (기본 16)}
  → 원본을 큰 캔버스 중앙에 두고 새 영역을 인페인트로 채운 뒤
    /api/results/ URL 반환 (upscale.py 결과 서빙 패턴).

GPU 없으면 503 (get_pipeline fail-fast).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from atelier.api.routes import history as history_module
from atelier.api.routes.generate import get_pipeline
from atelier.api.routes.upscale import _output_dir, _resolve_image
from atelier.engine import outpaint as outpaint_engine

router = APIRouter()


class OutpaintRequest(BaseModel):
    image: str  # base64 인코딩 이미지 또는 결과 파일명 (output dir 안)
    expand: dict = Field(default_factory=dict)  # {up, down, left, right}
    prompt: str = ""
    negative_prompt: str = ""
    denoise_strength: float = 0.6
    mask_blur: int = 16


@router.post("/api/outpaint")
def outpaint(req: OutpaintRequest):
    """이미지 바깥을 확장해 다시 그린다."""
    try:
        expand = outpaint_engine.validate_expand(req.expand)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not (0.0 <= req.denoise_strength <= 1.0):
        raise HTTPException(
            status_code=422, detail="디노이즈 강도는 0~1 사이여야 해요.")
    if not (0 <= req.mask_blur <= 64):
        raise HTTPException(
            status_code=422, detail="마스크 블러는 0~64 사이여야 해요.")
    image_bytes = _resolve_image(req.image)

    # 미설정 시 여기서 503 (fail-fast). GPU 없음도 여기서 걸러진다.
    pipe = get_pipeline()
    try:
        out_bytes = outpaint_engine.outpaint_image(
            image_bytes, expand, pipe,
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            denoise_strength=req.denoise_strength,
            mask_blur=req.mask_blur,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # GPU 없음 → 503 (pipeline.py _default_device 패턴)
        raise HTTPException(status_code=503, detail=str(e))

    out_dir = _output_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uniq = uuid.uuid4().hex[:6]
    name = f"{stamp}-outpaint-{uniq}.png"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(out_bytes)

    try:
        history_module.append_record(
            history_module.build_record(
                f"outpaint-{uniq}", "outpaint",
                {"expand": expand,
                 "denoise_strength": req.denoise_strength},
                ["/api/results/" + name], None,
            )
        )
    except Exception as e:  # noqa: BLE001 - 기록 실패가 작업을 망치면 안 됨
        print(f"[atelier] 작업내역 저장 실패: {e}")

    return {"url": "/api/results/" + name}
