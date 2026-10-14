"""LoRA 학습 API.

- POST /api/train: 학습 요청 접수 → 백그라운드 스레드에서 학습 → job_id 반환
- GET /api/train-jobs/{job_id}: 학습 상태 조회 (queued/running/done/error + 진행률)
- 학습 결과 .safetensors는 모델 폴더의 lora/ 에 저장되어 모델 스캔에 바로 잡힌다.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from atelier.api.routes import models as models_module
from atelier.engine.caption import caption_image
from atelier.engine.training import TrainConfig, train_lora

router = APIRouter()

_train_jobs: dict[str, dict] = {}
_train_lock = threading.Lock()


class TrainRequest(BaseModel):
    model: str = Field(description="베이스 체크포인트 (모델 목록의 상대경로)")
    dataset_dir: str = Field(description="학습 이미지 + .txt 캡션 폴더 (절대경로)")
    trigger_word: str = Field(default="", description="캡션 맨 앞에 삽입할 단어")
    rank: int = Field(default=16, ge=1, le=128)
    lr: float = Field(default=1e-4, gt=0, le=1e-2)
    steps_per_image: int = Field(default=75, ge=1, le=1000,
                                 description="이미지당 반복 스텝 (총 스텝 = 이미지 수 × 이 값)")
    resolution: int = Field(default=1024, ge=256, le=2048)
    train_text_encoder: bool = Field(default=False)
    output_name: str = Field(default="lora",
                              description="저장 파일명 (확장자 제외)")


def _run_train_sync(job_id: str, cfg: TrainConfig, out_name: str) -> None:
    job = _train_jobs[job_id]
    try:
        job["status"] = "running"

        def on_step(step: int, total: int, loss: float) -> None:
            job["step"] = step
            job["total_steps"] = total
            job["loss"] = round(loss, 4)

        cfg.on_step = on_step
        final_path = train_lora(cfg)

        # 최종본을 output_name으로 복사 (중간 체크포인트는 step 파일로 남음)
        import shutil
        dest = os.path.join(cfg.output_dir, f"{out_name}.safetensors")
        if os.path.abspath(final_path) != os.path.abspath(dest):
            shutil.copyfile(final_path, dest)
        job["status"] = "done"
        job["results"] = [os.path.basename(dest)]
    except Exception as e:  # noqa: BLE001 - 에러 메시지를 그대로 저장
        job["status"] = "error"
        job["error"] = str(e)


@router.post("/api/train", status_code=202)
def train(req: TrainRequest):
    """LoRA 학습 요청 접수. 검증 후 job_id를 즉시 반환한다."""
    model_path = models_module._resolve(req.model)

    ds = os.path.abspath(os.path.expanduser(req.dataset_dir))
    if not os.path.isdir(ds):
        raise HTTPException(
            status_code=422, detail="데이터셋 폴더를 찾을 수 없어요.")
    imgs = [f for f in os.listdir(ds)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
    if not imgs:
        raise HTTPException(
            status_code=422, detail="데이터셋 폴더에 이미지가 없어요.")

    out_dir = os.path.join(models_module.models_dir(), "lora")
    os.makedirs(out_dir, exist_ok=True)

    total_steps = len(imgs) * req.steps_per_image
    cfg = TrainConfig(
        model_path=model_path,
        dataset_dir=ds,
        output_dir=out_dir,
        trigger_word=req.trigger_word.strip(),
        rank=req.rank,
        lr=req.lr,
        max_train_steps=total_steps,
        resolution=req.resolution,
        train_text_encoder=req.train_text_encoder,
    )

    job_id = uuid.uuid4().hex
    _train_jobs[job_id] = {
        "job_id": job_id,
        "mode": "train",
        "status": "queued",
        "created_at": time.time(),
        "step": 0,
        "total_steps": total_steps,
        "loss": None,
        "results": [],
        "error": None,
    }
    thread = threading.Thread(
        target=_run_train_sync,
        args=(job_id, cfg, req.output_name),
        daemon=True,
        name=f"atelier-train-{job_id[:8]}",
    )
    thread.start()
    return {"job_id": job_id, "status": "queued",
            "total_steps": total_steps}


@router.get("/api/train-jobs/{job_id}")
def get_train_job(job_id: str):
    """학습 작업 상태 조회."""
    job = _train_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="학습 작업을 찾을 수 없어요.")
    return {
        "job_id": job["job_id"],
        "mode": job["mode"],
        "status": job["status"],
        "created_at": job["created_at"],
        "step": job["step"],
        "total_steps": job["total_steps"],
        "loss": job["loss"],
        "results": job["results"],
        "error": job["error"],
    }


class CaptionRequest(BaseModel):
    image_path: str = Field(description="캡션을 뽑을 이미지 (절대경로)")
    method: str = Field(default="wd14", description="'wd14' | 'blip'")


class CaptionBatchRequest(BaseModel):
    dataset_dir: str = Field(description="이미지 폴더 (절대경로)")
    method: str = Field(default="wd14", description="'wd14' | 'blip'")
    overwrite: bool = Field(default=False,
                            description="기존 .txt가 있어도 다시 쓰기")


@router.post("/api/caption")
def caption(req: CaptionRequest):
    """이미지 한 장에 자동 캡션을 붙인다."""
    if req.method not in ("wd14", "blip"):
        raise HTTPException(status_code=422,
                            detail="method는 'wd14' 또는 'blip'이에요.")
    path = os.path.abspath(os.path.expanduser(req.image_path))
    if not os.path.isfile(path):
        raise HTTPException(status_code=422,
                            detail="이미지 파일을 찾을 수 없어요.")
    try:
        return {"caption": caption_image(path, req.method)}
    except RuntimeError as e:
        # GPU 없음 등
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/api/caption/batch")
def caption_batch(req: CaptionBatchRequest):
    """폴더 안 이미지 전체에 .txt 캡션을 일괄 생성한다."""
    if req.method not in ("wd14", "blip"):
        raise HTTPException(status_code=422,
                            detail="method는 'wd14' 또는 'blip'이에요.")
    ds = os.path.abspath(os.path.expanduser(req.dataset_dir))
    if not os.path.isdir(ds):
        raise HTTPException(status_code=422,
                            detail="데이터셋 폴더를 찾을 수 없어요.")
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    done, skipped, failed = [], [], []
    for name in sorted(os.listdir(ds)):
        if not name.lower().endswith(exts):
            continue
        img_path = os.path.join(ds, name)
        txt_path = os.path.join(ds, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(txt_path) and not req.overwrite:
            skipped.append(name)
            continue
        try:
            cap = caption_image(img_path, req.method)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cap)
            done.append(name)
        except Exception as e:  # noqa: BLE001 - 한 장 실패해도 계속
            failed.append({"file": name, "error": str(e)})
    return {"done": done, "skipped": skipped, "failed": failed}
