"""ADetailer 스타일 얼굴 보정 API.

- POST /api/adetailer: {image, prompt, negative_prompt,
  denoising_strength, conf, mask_padding}
  → 얼굴 검출 + 인페인트 후 /api/results/ URL 반환, history 기록.
- 얼굴이 검출되지 않으면 원본 그대로의 URL과 faces_detected: 0 반환
  (에러 아님).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.api.routes import history as history_module
from atelier.api.routes.generate import get_pipeline
from atelier.api.routes.upscale import _looks_like_filename, _output_dir, _resolve_image
from atelier.engine import adetailer as ad_engine

router = APIRouter()


class AdetailerRequest(BaseModel):
    image: str  # base64 인코딩 이미지 또는 결과 파일명 (output dir 안)
    prompt: str = ""  # 비어 있으면 원본 생성 프롬프트(프론트가 함께 전달) 또는 폴백
    negative_prompt: str = ""
    denoising_strength: float = ad_engine.DEFAULT_DENOISING  # 0~1
    conf: float = ad_engine.DEFAULT_CONF  # 얼굴 검출 신뢰도 0~1
    mask_padding: int = ad_engine.DEFAULT_PADDING  # 마스크 확장 px


def _original_url(raw: str, image_bytes: bytes) -> str:
    """얼굴 미검출 시 원본 그대로의 URL.

    파일명이면 같은 URL, base64면 output dir에 복사본을 저장하고 URL 반환.
    """
    if _looks_like_filename(raw):
        return "/api/results/" + os.path.basename(raw)
    out_dir = _output_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uniq = uuid.uuid4().hex[:6]
    name = f"{stamp}-adetailer-noface-{uniq}.png"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(image_bytes)
    return "/api/results/" + name


@router.post("/api/adetailer")
def adetailer(req: AdetailerRequest):
    """얼굴을 검출해 해당 영역만 인페인트로 다시 그린다."""
    if not (ad_engine.MIN_DENOISING
            <= req.denoising_strength <= ad_engine.MAX_DENOISING):
        raise HTTPException(
            status_code=422,
            detail=f"디노이즈 강도는 {ad_engine.MIN_DENOISING}~"
                   f"{ad_engine.MAX_DENOISING} 사이여야 해요.",
        )
    if not (ad_engine.MIN_CONF <= req.conf <= ad_engine.MAX_CONF):
        raise HTTPException(
            status_code=422,
            detail=f"검출 신뢰도는 {ad_engine.MIN_CONF}~"
                   f"{ad_engine.MAX_CONF} 사이여야 해요.",
        )
    if req.mask_padding < 0:
        raise HTTPException(
            status_code=422,
            detail="마스크 확장은 0 이상이어야 해요.",
        )

    image_bytes = _resolve_image(req.image)
    pipe = get_pipeline()  # 미설정/GPU 없음 시 503 (fail-fast)
    try:
        out_path, faces = ad_engine.retouch(
            image_bytes,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            denoising_strength=req.denoising_strength,
            conf=req.conf,
            mask_padding=req.mask_padding,
            pipeline=pipe,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # GPU 없음 → 503 (pipeline.py _default_device 패턴)
        raise HTTPException(status_code=503, detail=str(e))

    if faces == 0 or out_path is None:
        # 얼굴 미검출: 원본 그대로 반환 (에러 아님)
        return {
            "url": _original_url(req.image, image_bytes),
            "faces_detected": 0,
        }

    # inpaint가 output dir에 저장한 파일을 그대로 서빙
    url = "/api/results/" + os.path.basename(out_path)
    try:
        history_module.append_record(
            history_module.build_record(
                f"adetailer-{uuid.uuid4().hex[:6]}", "adetailer",
                {
                    "prompt": req.prompt,
                    "denoising_strength": req.denoising_strength,
                    "conf": req.conf,
                    "mask_padding": req.mask_padding,
                    "faces_detected": faces,
                },
                [url], None,
            )
        )
    except Exception as e:  # noqa: BLE001 - 기록 실패가 보정을 망치면 안 됨
        print(f"[atelier] 작업내역 저장 실패: {e}")

    return {"url": url, "faces_detected": faces}
