"""GET /api/tag-labels 테스트. 실제 다운로드는 모킹."""

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import tags as tags_module
from atelier.engine import tag_labels as tl


@pytest.fixture()
def client(monkeypatch):
    fake = {
        "1girl": ["인물 > 인원수", "여성 캐릭터 한 명"],
        "solo": ["인물 > 인원수", "한 명만 등장"],
    }
    monkeypatch.setattr(tl, "ensure_labels", lambda: fake)
    return TestClient(app_module.create_app())


def test_tag_labels_endpoint(client):
    r = client.get("/api/tag-labels")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["labels"]["1girl"][0] == "인물 > 인원수"
    assert "source_url" in body


def test_tag_labels_lookup():
    # lookup은 ensure_labels를 그대로 쓰므로 _cache 주입으로 테스트
    tl._cache = {"test_tag": ["분류", "설명"]}
    try:
        assert tl.lookup("test_tag") == ["분류", "설명"]
        assert tl.lookup("nope") is None
    finally:
        tl._cache = None


def test_parse_filters_by_count(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "name,category,post_count,description\n"
        'a_tag,0,5000,"[인물] 설명 A"\n'
        'b_tag,0,50,"[인물] 설명 B"\n',
        encoding="utf-8",
    )
    data = tl._parse(str(csv_path))
    assert "a_tag" in data
    assert "b_tag" not in data  # 1000 미만 제외
    assert data["a_tag"] == ["인물", "설명 A"]
