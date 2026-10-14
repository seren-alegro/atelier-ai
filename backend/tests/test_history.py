"""작업내역 API 테스트: 완료 기록 append, 목록, 즐겨찾기 토글, 삭제.

실제 생성은 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.api.routes import history as hist
from atelier.settings import defaults

# test_generate.py와 같은 가짜 파이프라인을 재사용한다.
from test_generate import FakePipeline, wait_done  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    gen.set_pipeline(FakePipeline(str(out)))
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # uploads/ 격리
    gen._jobs.clear()
    yield TestClient(app_module.app)
    gen.set_pipeline(None)
    gen._jobs.clear()


def wait_history(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    """작업내역에 job 기록이 나타날 때까지 대기 (백그라운드 append 레이스 대응)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get("/api/history")
        assert r.status_code == 200
        for rec in r.json()["history"]:
            if rec["id"] == job_id:
                return rec
        time.sleep(0.05)
    raise AssertionError("작업내역에 기록이 저장되지 않았어요.")


def make_job(client: TestClient, params: dict | None = None) -> dict:
    r = client.post(
        "/api/generate",
        json={"mode": "txt2img", "params": params or defaults()},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_done(client, job_id)
    return wait_history(client, job_id)


# -- 기록 append --------------------------------------------------------

def test_history_append_on_done(client):
    params = defaults()
    params["prompt"] = "1girl, solo, test history"
    rec = make_job(client, params)

    assert rec["mode"] == "txt2img"
    assert rec["params"]["prompt"] == "1girl, solo, test history"
    assert rec["results"] == ["/api/results/fake-t.png"]
    assert rec["favorite"] is False
    assert rec["timestamp"]  # ISO 문자열

    # JSONL 파일에도 한 줄로 저장됨
    path = hist.history_path()
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 1


def test_history_newest_first_and_limit(client):
    ids = [make_job(client)["id"] for _ in range(3)]

    r = client.get("/api/history")
    got = [rec["id"] for rec in r.json()["history"]]
    assert got == ids[::-1]  # 최신순

    r = client.get("/api/history?limit=2")
    assert len(r.json()["history"]) == 2
    assert r.json()["history"][0]["id"] == ids[-1]


# -- 즐겨찾기 ------------------------------------------------------------

def test_history_favorite_toggle(client):
    rec = make_job(client)
    rid = rec["id"]

    r = client.post(f"/api/history/{rid}/favorite")
    assert r.status_code == 200
    assert r.json()["favorite"] is True

    r = client.post(f"/api/history/{rid}/favorite")
    assert r.json()["favorite"] is False

    # 목록 조회에도 반영됨
    r = client.get("/api/history")
    assert r.json()["history"][0]["favorite"] is False


def test_history_favorite_not_found(client):
    r = client.post("/api/history/없는기록/favorite")
    assert r.status_code == 404


# -- 삭제 ---------------------------------------------------------------

def test_history_delete(client):
    rec = make_job(client)
    rid = rec["id"]

    # 결과 파일 경로 확보
    result_file = rec["results"][0].rsplit("/", 1)[-1]

    r = client.delete(f"/api/history/{rid}")
    assert r.status_code == 200
    assert r.json()["deleted"] == rid

    r = client.get("/api/history")
    assert all(x["id"] != rid for x in r.json()["history"])

    # 결과 이미지 파일은 남아야 함
    r = client.get(f"/api/results/{result_file}")
    assert r.status_code == 200

    # 다시 삭제하면 404
    r = client.delete(f"/api/history/{rid}")
    assert r.status_code == 404


def test_history_delete_not_found(client):
    r = client.delete("/api/history/없는기록")
    assert r.status_code == 404


# -- 모델 이름 기록 -------------------------------------------------------

def test_history_records_active_model(client, monkeypatch):
    import atelier.api.routes.models as models_module

    monkeypatch.setattr(models_module, "_active_name", "anima.safetensors")
    rec = make_job(client)
    assert rec["model"] == "anima.safetensors"
