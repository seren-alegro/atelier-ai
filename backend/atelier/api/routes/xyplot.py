"""X/Y 플롯 API.

- POST /api/xy-plot: {x_axis: {type, values[]}, y_axis? (선택),
    base_params, loras?, seed_rule?}
  → 202 {job_id}. batch.py의 잡 인프라를 재사용한다.
- GET /api/xy-plot/{job_id}: 진행 상태 조회 (그리드용 축 정보 포함)
- DELETE /api/xy-plot/{job_id}: 취소 요청

축 타입: sampler, cfg_scale, steps, seed.
체크포인트는 모델 리로드 비용 때문에 이번에 제외.
셀 수 상한: x*y ≤ 36 (과도한 GPU 점유 방지).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from atelier.api.routes.generate import get_pipeline
from atelier.engine import batch as batch_engine
from atelier.settings import get_setting, validate

router = APIRouter()

# 축 타입 → settings 키
AXIS_KEY = {
    "sampler": "sampler",
    "cfg_scale": "cfg_scale",
    "steps": "steps",
    "seed": "seed",
}

# 축 라벨 표시용 이름
AXIS_LABEL = {
    "sampler": "샘플러",
    "cfg_scale": "CFG",
    "steps": "스텝",
    "seed": "시드",
}

MAX_CELLS = 36


class AxisSpec(BaseModel):
    type: str
    values: list = Field(default_factory=list)


class XyPlotRequest(BaseModel):
    x_axis: AxisSpec
    y_axis: AxisSpec | None = None
    base_params: dict = Field(default_factory=dict)
    loras: dict | None = None  # {이름: {"enabled": bool, "strength": float}}
    seed_rule: str = "fixed"  # fixed (기본, 비교용) | increment


def _coerce_value(axis_type: str, v):
    """축 값을 settings 키에 맞는 타입으로 정리. 불가하면 422."""
    key = AXIS_KEY[axis_type]
    if axis_type == "sampler":
        if not isinstance(v, str):
            raise HTTPException(
                status_code=422,
                detail=f"'{AXIS_LABEL[axis_type]}' 값은 문자열이어야 해요: {v}",
            )
        coerced = v
    else:
        # bool은 int로 오인되므로 먼저 거름
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise HTTPException(
                status_code=422,
                detail=f"'{AXIS_LABEL[axis_type]}' 값은 숫자여야 해요: {v}",
            )
        coerced = v
    # settings.validate로 범위·옵션 검증 (모르는 키 422 패턴과 동일)
    errors = validate({key: coerced})
    if errors:
        raise HTTPException(
            status_code=422, detail=list(errors.values())[0])
    return coerced


def _validate_axis(spec: AxisSpec, name: str) -> list:
    if spec.type not in AXIS_KEY:
        raise HTTPException(
            status_code=422,
            detail=f"{name}축 타입은 {', '.join(AXIS_KEY)} 중 하나여야 해요.",
        )
    if not spec.values:
        raise HTTPException(
            status_code=422,
            detail=f"{name}축 값을 하나 이상 입력해주세요.",
        )
    return [_coerce_value(spec.type, v) for v in spec.values]


@router.post("/api/xy-plot", status_code=202)
def create_xy_plot(req: XyPlotRequest):
    """X/Y 플롯 요청 접수. 검증 후 job_id를 즉시 반환한다."""
    errors = validate(req.base_params)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    if req.seed_rule not in ("fixed", "increment"):
        raise HTTPException(
            status_code=422,
            detail="seed_rule은 fixed 또는 increment여야 해요.",
        )
    x_values = _validate_axis(req.x_axis, "X")
    y_values = _validate_axis(req.y_axis, "Y") if req.y_axis else []
    if len(x_values) * max(1, len(y_values)) > MAX_CELLS:
        raise HTTPException(
            status_code=422,
            detail=f"한 번에 {MAX_CELLS}칸까지만 가능해요 "
                   f"(지금 {len(x_values)}×{max(1, len(y_values))}).",
        )

    x_key = AXIS_KEY[req.x_axis.type]
    y_key = AXIS_KEY[req.y_axis.type] if req.y_axis else None

    prompts: list[str] = []
    item_params: list[dict] = []
    item_labels: list[str] = []
    y_labels = [str(v) for v in y_values] if y_values else []
    x_labels = [str(v) for v in x_values]
    for yi, yv in enumerate(y_values if y_values else [None]):
        for xi, xv in enumerate(x_values):
            params: dict = {x_key: xv}
            label_parts = [f"{AXIS_LABEL[req.x_axis.type]} {xv}"]
            if y_key is not None:
                params[y_key] = yv
                label_parts.append(f"{AXIS_LABEL[req.y_axis.type]} {yv}")
            prompts.append(req.base_params.get("prompt", ""))
            item_params.append(params)
            item_labels.append(" / ".join(label_parts))

    # 미설정 시 여기서 503 (fail-fast). GPU 없음도 여기서 걸러진다.
    pipe = get_pipeline()
    try:
        job = batch_engine.start_job(
            prompts, req.base_params, req.loras, pipe,
            item_params=item_params, item_labels=item_labels,
            seed_rule=req.seed_rule,
            meta={
                "kind": "xy-plot",
                "x_type": req.x_axis.type,
                "y_type": req.y_axis.type if req.y_axis else None,
                "x_labels": x_labels,
                "y_labels": y_labels,
            },
        )
    except batch_engine.BatchBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job.job_id, "status": job.status}


@router.get("/api/xy-plot/{job_id}")
def get_xy_plot(job_id: str):
    """X/Y 플롯 작업 상태 조회."""
    job = batch_engine.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="X/Y 플롯 작업을 찾을 수 없어요.")
    return job.to_dict()


@router.delete("/api/xy-plot/{job_id}")
def cancel_xy_plot(job_id: str):
    """X/Y 플롯 작업 취소 요청. 다음 칸 시작 전에 중단된다."""
    job = batch_engine.cancel_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="X/Y 플롯 작업을 찾을 수 없어요.")
    return job.to_dict()
