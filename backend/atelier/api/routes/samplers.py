"""샘플러 별칭 API.

- GET /api/sampler-aliases: 별칭→내부 키 매핑 + 내부 키 목록.
  프론트가 PNG 메타데이터의 샘플러 표시 이름("Euler a" 등)을
  내부 키("euler_ancestral")로 정규화할 때 쓴다.
"""

from __future__ import annotations

from fastapi import APIRouter

from atelier.engine.samplers import SAMPLER_ALIASES, SAMPLER_MAP

router = APIRouter()


@router.get("/api/sampler-aliases")
def sampler_aliases():
    """샘플러 별칭 매핑 반환."""
    return {"aliases": SAMPLER_ALIASES, "keys": sorted(SAMPLER_MAP)}


@router.get("/api/samplers")
def list_samplers():
    """샘플러별 스케줄러 정보 반환. note는 한국어 원문 그대로."""
    return {
        key: {
            "scheduler": spec.scheduler,
            "approximate": spec.approximate,
            "note": spec.note,
        }
        for key, spec in SAMPLER_MAP.items()
    }
