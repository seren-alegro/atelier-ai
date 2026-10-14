"""단보루 태그 한국어 라벨.

출처: localsmile님의 "단부루 위키 한국어 번역 및 검색"
(https://localsmile.github.io/danbooru_KR_wiki_tag_search/)
공개 CSV(danbooru_tags_classified.csv)를 첫 사용 시 내려받아
사용횟수 1000 이상 태그만 간추린 JSON으로 캐시한다.
재배포하지 않고 런타임에 내려받는 방식이다.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import urllib.request

CSV_URL = ("https://localsmile.github.io/danbooru_KR_wiki_tag_search/"
           "danbooru_tags_classified.csv")
SOURCE_NAME = "단부루 위키 한국어 번역 (localsmile)"
SOURCE_URL = "https://localsmile.github.io/danbooru_KR_wiki_tag_search/"
MIN_COUNT = 1000

_lock = threading.Lock()
_cache: dict | None = None


def labels_path() -> str:
    from atelier.api.routes.history import data_dir
    return os.path.join(data_dir(), "tag_labels.json")


def _download_csv() -> str:
    tmp = labels_path() + ".csv.part"
    final_csv = labels_path() + ".csv"
    if os.path.isfile(final_csv):
        return final_csv
    try:
        with urllib.request.urlopen(CSV_URL, timeout=300) as r, \
                open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, final_csv)
    return final_csv


def _parse(csv_path: str) -> dict:
    """CSV → {tag: [카테고리, 설명]}. 설명 앞의 [분류]는 카테고리로 분리."""
    data: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                count = int(row.get("post_count") or 0)
            except ValueError:
                continue
            if count < MIN_COUNT:
                continue
            desc = (row.get("description") or "").strip()
            cat = ""
            if desc.startswith("[") and "]" in desc:
                cat = desc[1:desc.index("]")].strip()
                desc = desc[desc.index("]") + 1:].strip()
            data[row["name"]] = [cat, desc]
    return data


def ensure_labels() -> dict:
    """라벨 dict를 반환. 없으면 다운로드+파싱 후 JSON 캐시."""
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        path = labels_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                _cache = json.load(f)
            return _cache
        print("[atelier] 태그 한국어 라벨을 내려받습니다 (첫 실행 시 약 16MB)...")
        csv_path = _download_csv()
        data = _parse(csv_path)
        # 원본 CSV는 캐시 JSON만 남기고 정리
        try:
            os.remove(csv_path)
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        _cache = data
        return _cache


def lookup(tag: str) -> list[str] | None:
    """태그 하나 조회 → [카테고리, 설명] 또는 None."""
    return ensure_labels().get(tag)
