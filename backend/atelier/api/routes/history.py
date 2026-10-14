"""작업내역 API.

- GET /api/history: 생성 기록 목록 (최신순)
- DELETE /api/history/{record_id}: 기록 삭제 (결과 파일은 유지)
- POST /api/history/{record_id}/favorite: 즐겨찾기 토글

기록은 JSONL 파일에 append. 경로: ATELIER_DATA_DIR 환경 변수 우선,
없으면 ~/.atelier-ai/history.jsonl.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException

router = APIRouter()


def data_dir() -> str:
    """데이터 디렉토리. 없으면 만들어 둔다."""
    d = os.environ.get("ATELIER_DATA_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai"
    )
    os.makedirs(d, exist_ok=True)
    return d


def history_path() -> str:
    return os.path.join(data_dir(), "history.jsonl")


def read_records() -> list[dict]:
    """JSONL을 읽어 레코드 목록 반환 (오래된 순). 깨진 줄은 건너뛴다."""
    path = history_path()
    records: list[dict] = []
    if not os.path.isfile(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("id"):
                records.append(rec)
    return records


def append_record(record: dict) -> None:
    """생성 완료 기록을 JSONL에 추가한다."""
    record = dict(record)
    record.setdefault("favorite", False)
    with open(history_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_all(records: list[dict]) -> None:
    with open(history_path(), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_record(
    job_id: str, mode: str, params: dict, results: list[str]
) -> dict:
    """job 완료 시 저장할 레코드를 만든다. 당시 활성 모델 이름도 함께 담는다."""
    try:
        # 순환 import를 피하려고 함수 안에서 가져온다.
        from atelier.api.routes import models as models_module

        model = models_module.get_active_name()
    except Exception:  # noqa: BLE001 - 모델 이름은 부가 정보라 실패해도 무시
        model = None
    return {
        "id": job_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "params": params,
        "results": results,
        "favorite": False,
    }


@router.get("/api/history")
def list_history(limit: int = 100):
    """생성 기록 목록 (최신순)."""
    records = read_records()
    records.reverse()
    return {"history": records[: max(limit, 0)]}


@router.delete("/api/history/{record_id}")
def delete_history(record_id: str):
    """기록 삭제. 결과 이미지 파일은 남겨둔다."""
    records = read_records()
    kept = [r for r in records if r["id"] != record_id]
    if len(kept) == len(records):
        raise HTTPException(
            status_code=404, detail="기록을 찾을 수 없어요."
        )
    _write_all(kept)
    return {"deleted": record_id}


@router.post("/api/history/{record_id}/favorite")
def toggle_favorite(record_id: str):
    """즐겨찾기 토글. 바뀐 상태를 반환한다."""
    records = read_records()
    new_value: bool | None = None
    for r in records:
        if r["id"] == record_id:
            r["favorite"] = not r.get("favorite", False)
            new_value = r["favorite"]
    if new_value is None:
        raise HTTPException(
            status_code=404, detail="기록을 찾을 수 없어요."
        )
    _write_all(records)
    return {"id": record_id, "favorite": new_value}
