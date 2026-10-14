"""생성 작업 API.

- POST /api/generate: 생성 요청 접수 → job_id 즉시 반환(202), 백그라운드 처리
- GET /api/jobs/{job_id}: 작업 상태 조회 (queued/running/done/error)
- GET /api/results/{filename}: 생성된 이미지 파일 서빙

작업 저장소는 인메모리 dict. 서버 재시작 시 사라진다 (MVP).
"""

from __future__ import annotations

import base64
import binascii
import os
import threading
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from atelier.engine.pipeline import Pipeline
from atelier.settings import validate
from atelier.api.routes import history as history_module

router = APIRouter()

MODES = ("txt2img", "img2img", "inpaint")

# job_id -> job dict. 키: job_id, mode, params, status, created_at, results, error
_jobs: dict[str, dict] = {}

_pipeline: Pipeline | None = None


class GenerateRequest(BaseModel):
    mode: Literal["txt2img", "img2img", "inpaint"]
    params: dict = Field(default_factory=dict)
    image: str | None = None  # base64 인코딩 원본 이미지 (img2img/inpaint)
    mask: str | None = None   # base64 인코딩 마스크 (inpaint)


def configure(model_path: str, output_dir: str = "outputs") -> None:
    """파이프라인을 설정한다."""
    set_pipeline(Pipeline(model_path=model_path, output_dir=output_dir))


def set_pipeline(pipe: Pipeline | None) -> None:
    """파이프라인을 직접 주입한다. 테스트용."""
    global _pipeline
    _pipeline = pipe


def get_pipeline() -> Pipeline:
    """설정된 파이프라인 반환. 미설정 시 503."""
    if _pipeline is not None:
        return _pipeline
    model_path = os.environ.get("ATELIER_MODEL_PATH")
    if not model_path:
        raise HTTPException(
            status_code=503,
            detail="모델이 설정되지 않았어요. "
                   "ATELIER_MODEL_PATH 환경 변수에 체크포인트 경로를 설정하세요.",
        )
    configure(model_path)
    assert _pipeline is not None
    return _pipeline


def _save_upload(data_b64: str, job_id: str, name: str) -> str:
    """base64 업로드를 파일로 저장하고 경로 반환."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail=f"{name} 이미지 데이터가 올바르지 않아요.",
        )
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, f"{job_id}-{name}.png")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _run_job_sync(
    job_id: str,
    mode: str,
    params: dict,
    image_path: str | None,
    mask_path: str | None,
) -> None:
    """백그라운드 생성 작업 (별도 스레드). 결과/에러를 job dict에 기록.

    asyncio.to_thread 대신 threading.Thread를 직접 쓴다:
    이 환경에서 to_thread의 future가 간헐적으로 resolve되지 않는
    레이스가 관측됐기 때문. _jobs dict 갱신은 단순 대입이라 GIL로 안전.
    """
    job = _jobs[job_id]
    job["status"] = "running"
    try:
        pipe = get_pipeline()
        if mode == "txt2img":
            paths = pipe.txt2img(params)
        elif mode == "img2img":
            paths = pipe.img2img(params, image_path)
        else:
            paths = pipe.inpaint(params, image_path, mask_path)
        job["status"] = "done"
        job["results"] = [
            "/api/results/" + os.path.basename(p) for p in paths
        ]
        try:
            history_module.append_record(
                history_module.build_record(
                    job_id, mode, params, job["results"]
                )
            )
        except Exception as e:  # noqa: BLE001 - 기록 실패가 생성을 망치면 안 됨
            print(f"[atelier] 작업내역 저장 실패: {e}")
    except Exception as e:  # noqa: BLE001 - 에러 메시지를 그대로 저장
        job["status"] = "error"
        job["error"] = str(e)


@router.post("/api/generate", status_code=202)
def generate(req: GenerateRequest):
    """생성 요청 접수. 검증 후 job_id를 즉시 반환한다."""
    errors = validate(req.params)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    if req.mode == "img2img" and not req.image:
        raise HTTPException(
            status_code=422, detail="img2img에는 원본 이미지가 필요해요.")
    if req.mode == "inpaint" and not (req.image and req.mask):
        raise HTTPException(
            status_code=422,
            detail="인페인트에는 원본 이미지와 마스크가 필요해요.")
    get_pipeline()  # 미설정 시 여기서 503 (fail-fast)

    job_id = uuid.uuid4().hex
    image_path = _save_upload(req.image, job_id, "image") if req.image else None
    mask_path = _save_upload(req.mask, job_id, "mask") if req.mask else None
    _jobs[job_id] = {
        "job_id": job_id,
        "mode": req.mode,
        "params": req.params,
        "status": "queued",
        "created_at": time.time(),
        "results": [],
        "error": None,
    }
    thread = threading.Thread(
        target=_run_job_sync,
        args=(job_id, req.mode, req.params, image_path, mask_path),
        daemon=True,
        name=f"atelier-job-{job_id[:8]}",
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """작업 상태 조회."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="작업을 찾을 수 없어요.")
    return {
        "job_id": job["job_id"],
        "mode": job["mode"],
        "status": job["status"],
        "created_at": job["created_at"],
        "results": job["results"],
        "error": job["error"],
    }


@router.get("/api/results/{filename}")
def get_result(filename: str):
    """생성된 이미지 파일 서빙."""
    pipe = get_pipeline()
    safe = os.path.basename(filename)
    if not safe or safe != filename:
        raise HTTPException(
            status_code=400, detail="잘못된 파일 이름이에요.")
    path = os.path.join(pipe.output_dir, safe)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404, detail="결과 파일을 찾을 수 없어요.")
    return FileResponse(path, media_type="image/png")
