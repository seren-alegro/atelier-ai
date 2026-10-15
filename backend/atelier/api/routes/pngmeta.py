"""PNG 메타데이터 읽기 API.

- POST /api/png-info: 이미지 업로드 → PNG tEXt/iTXt 청크에서 생성 설정 추출.
  Atelier AI가 저장한 `atelier:params` JSON을 우선 읽고, 없으면
  A1111식 `parameters` 텍스트 청크를 파싱한다.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from atelier.api.routes.generate import MAX_UPLOAD_BYTES
from atelier.pngmeta import read_params_from_bytes

router = APIRouter()


@router.post("/api/png-info")
async def png_info(file: UploadFile = File(...)):
    """업로드된 PNG에서 생성 설정을 읽는다.

    PNG가 아니거나 설정 메타가 없으면 {"found": false, "params": null}.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지가 너무 커요. "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB 이하로 올려주세요.",
        )
    params = read_params_from_bytes(data)
    if params is None:
        return {"found": False, "params": None}
    return {"found": True, "params": params}
