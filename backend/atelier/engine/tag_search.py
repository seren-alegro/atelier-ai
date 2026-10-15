"""단보루 태그 검색 (프롬프트 도우미용).

backend/atelier/data/danbooru-tags-kr.csv를 lazy-load한다
(127,742행 · 24MB — 첫 검색 시 한 번만 읽음).
사용자 지정으로 카테고리 1(작가)·3(저작권/시리즈)·4(캐릭터)는 로드 시 제외한다.

카테고리: 0=일반, 1=작가, 3=저작권/시리즈, 4=캐릭터, 5=메타.
"""

from __future__ import annotations

import csv
import sys
import threading
from pathlib import Path

CSV_NAME = "danbooru-tags-kr.csv"

# 제외 카테고리: 1=작가, 3=저작권/시리즈, 4=캐릭터 (사용자 지정)
EXCLUDED_CATEGORIES = {1, 3, 4}

CATEGORY_LABELS = {
    0: "일반",
    5: "메타",
}

_lock = threading.Lock()
_entries: list[dict] | None = None  # {tag, category, count, desc_ko}


def _resolve_csv() -> Path:
    """CSV 경로 탐색: 패키지 내장 → frozen → 사용자 데이터 dir 순."""
    bundled = Path(__file__).resolve().parent / "data" / CSV_NAME
    if bundled.is_file():
        return bundled
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        frozen = Path(meipass) / "atelier" / "data" / CSV_NAME
        if frozen.is_file():
            return frozen
    from atelier.api.routes.history import data_dir  # 순환 import 방지용 지연
    user_copy = Path(data_dir()) / CSV_NAME
    if user_copy.is_file():
        return user_copy
    raise FileNotFoundError(f"태그 CSV를 찾지 못했어요: {CSV_NAME}")


def _strip_bracket(desc: str) -> str:
    """desc_ko 앞의 [분류] 부분을 떼고 설명+키워드만 남긴다."""
    desc = desc.strip()
    if desc.startswith("[") and "]" in desc:
        desc = desc[desc.index("]") + 1:].strip()
    return desc


def _load() -> list[dict]:
    global _entries
    if _entries is not None:
        return _entries
    with _lock:
        if _entries is not None:
            return _entries
        path = _resolve_csv()
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                try:
                    category = int(row[1])
                    count = int(row[2])
                except ValueError:
                    continue
                if category in EXCLUDED_CATEGORIES:
                    continue
                rows.append({
                    "tag": row[0],
                    "category": category,
                    "count": count,
                    "desc_ko": _strip_bracket(row[3]),
                })
        _entries = rows
        return _entries


def total_count() -> int:
    """검색 대상 전체 태그 수 (제외 카테고리 반영)."""
    return len(_load())


def search(query: str, limit: int = 50) -> list[dict]:
    """태그명(소문자, `_`→공백 매칭 포함)과 한국어 설명에 부분일치 검색.

    사용횟수(count) 내림차순 정렬. 빈 쿼리는 [] 반환.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    limit = max(1, min(100, int(limit)))
    hits: list[dict] = []
    for e in _load():
        tag_l = e["tag"].lower()
        if q in tag_l or q in tag_l.replace("_", " ") \
                or q in e["desc_ko"].lower():
            hits.append(e)
    hits.sort(key=lambda e: e["count"], reverse=True)
    return hits[:limit]
