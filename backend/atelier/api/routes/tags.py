"""태그 한국어 라벨 API.

- GET /api/tag-labels: 전체 라벨 dict ({tag: [카테고리, 설명]}).
  첫 호출 시 원본 CSV를 내려받아 파싱하므로 수십 초 걸릴 수 있다.
  프론트는 한 번 받아 메모리에 캐시한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atelier.engine import tag_labels as tl
from atelier.engine import tag_search as ts

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


@router.get("/api/tag-search")
def search_tags(q: str = "", limit: int = 50):
    """단보루 태그 검색. q 없거나 비면 422, limit은 1~100."""
    if not (q or "").strip():
        raise HTTPException(status_code=422, detail="검색어를 입력해주세요.")
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422, detail="limit은 1~100 사이여야 해요.")
    try:
        results = ts.search(q, limit=limit)
        total = ts.total_count()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - 파싱 실패 등
        raise HTTPException(
            status_code=503, detail=f"태그 검색 중 오류가 발생했어요: {e}")
    return {"count": total, "results": results}
