"""프리셋 API 테스트: 즐겨찾기 CRUD, 네거티브 프리셋 내장/사용자.

ATELIER_DATA_DIR를 tmp로 격리한다.
"""

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import presets


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    yield TestClient(app_module.app)


# -- 프롬프트 즐겨찾기 ------------------------------------------------------


def test_favorites_empty_initially(client):
    assert client.get("/api/prompt-favorites").json() == []


def test_favorites_crud(client):
    r = client.post(
        "/api/prompt-favorites",
        json={"name": "은발 소녀", "prompt": "1girl, silver hair, solo"},
    )
    assert r.status_code == 201
    item = r.json()
    assert item["name"] == "은발 소녀"
    assert item["prompt"] == "1girl, silver hair, solo"
    assert item["id"]
    assert item["created_at"]

    items = client.get("/api/prompt-favorites").json()
    assert len(items) == 1
    assert items[0]["id"] == item["id"]

    r = client.delete(f"/api/prompt-favorites/{item['id']}")
    assert r.status_code == 200
    assert client.get("/api/prompt-favorites").json() == []


def test_favorites_empty_name_or_prompt_422(client):
    assert client.post(
        "/api/prompt-favorites", json={"name": "  ", "prompt": "1girl"}
    ).status_code == 422
    assert client.post(
        "/api/prompt-favorites", json={"name": "x", "prompt": ""}
    ).status_code == 422


def test_favorites_delete_missing_404(client):
    assert client.delete("/api/prompt-favorites/nonexistent").status_code == 404


# -- 네거티브 프리셋 --------------------------------------------------------


def test_negative_presets_builtin_three(client):
    items = client.get("/api/negative-presets").json()
    builtins = [i for i in items if i.get("builtin")]
    assert len(builtins) == 3
    names = {i["name"] for i in builtins}
    assert names == {"기본 정리", "손 보정", "글자 제거"}
    basic = next(i for i in builtins if i["name"] == "기본 정리")
    assert "bad hands" in basic["text"]


def test_negative_preset_user_crud(client):
    r = client.post(
        "/api/negative-presets",
        json={"name": "내 프리셋", "text": "lowres, blurry"},
    )
    assert r.status_code == 201
    item = r.json()
    assert item["builtin"] is False

    items = client.get("/api/negative-presets").json()
    assert len(items) == 4  # 내장 3 + 사용자 1

    r = client.delete(f"/api/negative-presets/{item['id']}")
    assert r.status_code == 200
    items = client.get("/api/negative-presets").json()
    assert len(items) == 3


def test_negative_preset_builtin_delete_400(client):
    r = client.delete("/api/negative-presets/builtin-basic")
    assert r.status_code == 400


def test_negative_preset_delete_missing_404(client):
    assert client.delete("/api/negative-presets/nonexistent").status_code == 404


def test_negative_preset_empty_422(client):
    assert client.post(
        "/api/negative-presets", json={"name": "x", "text": "  "}
    ).status_code == 422
