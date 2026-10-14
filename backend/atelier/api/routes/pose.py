"""포즈 자동 추출 API.

- POST /api/pose-detect: 사진(base64) → OpenPose 스켈레톤 PNG(base64)

엔진(atelier.engine.pose_detect)은 numpy/PIL/onnxruntime이 필요하므로
지연 로딩한다. import 실패 시 500으로 안내하고, 나머지 앱에는 영향 없음.
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from atelier.engine import pose_detect as _pose_engine
    _POSE_IMPORT_ERROR: Exception | None = None
except ImportError as e:  # minimal 환경(numpy 없음)에서도 앱 자체는 뜨게
    _pose_engine = None
    _POSE_IMPORT_ERROR = e

router = APIRouter()


class PoseDetectRequest(BaseModel):
    image: str = ""  # base64 인코딩 사진


def _engine():
    if _pose_engine is None:
        raise HTTPException(
            status_code=500,
            detail="포즈 추출 모듈을 불러오지 못했어요. "
                   "백엔드 가상환경에 numpy·Pillow·onnxruntime이 필요합니다.",
        )
    return _pose_engine


@router.post("/api/pose-detect")
def pose_detect(req: PoseDetectRequest):
    """사진에서 OpenPose 스켈레톤을 추출한다.

    첫 호출 시 포즈 모델(약 350MB)을 내려받으므로 시간이 걸릴 수 있다.
    """
    if not req.image or not req.image.strip():
        raise HTTPException(
            status_code=422,
            detail="사진을 먼저 올려주세요.",
        )
    try:
        raw = base64.b64decode(req.image, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail="사진 데이터가 올바르지 않아요.",
        )
    engine = _engine()
    try:
        png = engine.detect_pose_skeleton(raw)
    except engine.NoPersonError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"skeleton": base64.b64encode(png).decode("ascii")}
