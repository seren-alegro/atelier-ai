"""태그 한국어 라벨 API.

- GET /api/tag-labels: 전체 라벨 dict ({tag: [카테고리, 설명]}).
  첫 호출 시 원본 CSV를 내려받아 파싱하므로 수십 초 걸릴 수 있다.
  프론트는 한 번 받아 메모리에 캐시한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atelier.engine import tag_labels as tl

router = APIRouter()


@router.get("/api/tag-labels")
def get_tag_labels():
    try:
        labels = tl.ensure_labels()
    except Exception as e:  # noqa: BLE001 - 다운로드 실패 등
        raise HTTPException(
            status_code=503,
            detail=f"태그 라벨을 불러오지 못했어요: {e}")
    return {
        "source": tl.SOURCE_NAME,
        "source_url": tl.SOURCE_URL,
        "count": len(labels),
        "labels": labels,
    }
