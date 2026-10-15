"""GET /api/tag-search + engine/tag_search 테스트. 작은 mock CSV 사용."""

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.engine import tag_search as ts

MOCK_CSV = (
    '1girl,0,8502520,"[인물 > 인원수] 여성 캐릭터가 한 명 등장함. 키워드: 여자 1명"\n'
    'long_hair,0,5000000,"[인물 > 머리] 긴 머리. 키워드: 긴머리카락"\n'
    'hatsune_miku,4,3000000,"[캐릭터] 하츠네 미쿠"\n'
    'some_artist,1,2000000,"[작가] 모 작가의 화풍"\n'
    'some_series,3,1000000,"[저작권] 모 시리즈"\n'
    'highres,5,8268926,"[메타 > 해상도] 고해상도 이미지"\n'
    'blue_eyes,0,4000000,"[인물 > 눈] 파란 눈. 키워드: 파란눈"\n'
)


@pytest.fixture()
def mock_csv(tmp_path, monkeypatch):
    p = tmp_path / "danbooru-tags-kr.csv"
    p.write_text(MOCK_CSV, encoding="utf-8")
    monkeypatch.setattr(ts, "_resolve_csv", lambda: p)
    monkeypatch.setattr(ts, "_entries", None)
    yield p
    ts._entries = None


@pytest.fixture()
def client(mock_csv):
    return TestClient(app_module.create_app())


def test_excluded_categories(mock_csv):
    entries = ts._load()
    tags = {e["tag"] for e in entries}
    assert "some_artist" not in tags  # 카테고리 1 제외
    assert "some_series" not in tags  # 카테고리 3 제외
    assert "hatsune_miku" not in tags  # 카테고리 4 제외
    assert {"1girl", "long_hair", "highres", "blue_eyes"} <= tags
    assert ts.total_count() == 4


def test_bracket_stripped(mock_csv):
    entries = {e["tag"]: e for e in ts._load()}
    assert entries["1girl"]["desc_ko"].startswith("여성 캐릭터")
    assert not entries["1girl"]["desc_ko"].startswith("[")


def test_search_english(mock_csv):
    res = ts.search("hair")
    assert [e["tag"] for e in res] == ["long_hair"]


def test_search_underscore_as_space(mock_csv):
    # "blue eyes"처럼 공백으로 검색해도 blue_eyes가 걸린다
    res = ts.search("blue eyes")
    assert any(e["tag"] == "blue_eyes" for e in res)


def test_search_korean_desc(mock_csv):
    res = ts.search("파란눈")
    assert any(e["tag"] == "blue_eyes" for e in res)


def test_search_sorted_by_count(mock_csv):
    res = ts.search("키워드")
    counts = [e["count"] for e in res]
    assert counts == sorted(counts, reverse=True)
    assert res[0]["tag"] == "1girl"


def test_search_limit(mock_csv):
    res = ts.search("e", limit=2)
    assert len(res) == 2


def test_search_empty_query(mock_csv):
    assert ts.search("") == []
    assert ts.search("   ") == []


def test_endpoint_ok(client):
    r = client.get("/api/tag-search", params={"q": "hair", "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    assert body["results"][0]["tag"] == "long_hair"
    assert "desc_ko" in body["results"][0]


def test_endpoint_empty_q_422(client):
    assert client.get("/api/tag-search").status_code == 422
    assert client.get("/api/tag-search", params={"q": "  "}).status_code == 422


def test_endpoint_limit_range_422(client):
    assert client.get("/api/tag-search", params={"q": "a", "limit": 0}).status_code == 422
    assert client.get("/api/tag-search", params={"q": "a", "limit": 101}).status_code == 422


def test_endpoint_missing_csv_503(monkeypatch):
    monkeypatch.setattr(ts, "_resolve_csv", lambda: (_ for _ in ()).throw(
        FileNotFoundError("태그 CSV를 찾지 못했어요")))
    monkeypatch.setattr(ts, "_entries", None)
    try:
        r = TestClient(app_module.create_app()).get(
            "/api/tag-search", params={"q": "hair"})
        assert r.status_code == 503
    finally:
        ts._entries = None
