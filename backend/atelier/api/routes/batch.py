"""배치 생성 큐 API.

- POST /api/batch: {prompts (1~200개), params (공통 설정), loras?}
  → 202 {job_id}. 프롬프트마다 순차로 txt2img 실행.
- GET /api/batch/{job_id}: 진행 상태 조회
- DELETE /api/batch/{job_id}: 취소 요청

동시에 실행되는 배치는 1개로 제한 (실행 중이면 409).
작업 저장소는 인메모리 dict. 서버 재시작 시 사라진다 (generate.py와 동일).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from atelier.api.routes.generate import get_pipeline
from atelier.engine import batch as batch_engine
from atelier.settings import validate

router = APIRouter()


class BatchRequest(BaseModel):
    prompts: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    loras: dict | None = None  # {이름: {"enabled": bool, "strength": float}}


@router.post("/api/batch", status_code=202)
def create_batch(req: BatchRequest):
    """배치 생성 요청 접수. 검증 후 job_id를 즉시 반환한다."""
    errors = validate(req.params)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    prompts = [
        p.strip() for p in req.prompts
        if isinstance(p, str) and p.strip()
    ]
    if not prompts:
        raise HTTPException(
            status_code=422, detail="프롬프트가 비어 있어요.")
    if len(prompts) > batch_engine.MAX_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail=f"한 번에 {batch_engine.MAX_PROMPTS}개까지만 가능해요.",
        )
    # 미설정 시 여기서 503 (fail-fast). GPU 없음도 여기서 걸러진다.
    pipe = get_pipeline()
    try:
        job = batch_engine.start_job(prompts, req.params, req.loras, pipe)
    except batch_engine.BatchBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/api/batch/{job_id}")
def get_batch(job_id: str):
    """배치 작업 상태 조회."""
    job = batch_engine.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="배치 작업을 찾을 수 없어요.")
    return job.to_dict()


@router.delete("/api/batch/{job_id}")
def cancel_batch(job_id: str):
    """배치 작업 취소 요청. 다음 항목 시작 전에 중단된다."""
    job = batch_engine.cancel_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="배치 작업을 찾을 수 없어요.")
    return job.to_dict()
