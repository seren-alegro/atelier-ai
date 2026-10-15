"""POST /api/batch + 배치 큐 엔진 테스트.

실제 생성은 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
"""

import os
import struct
import time
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.engine import batch as batch_engine


def make_png_bytes() -> bytes:
    """stdlib만으로 만드는 1x1 PNG."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\x00\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class FakePipeline:
    """호출된 프롬프트·시드를 기록하는 가짜 파이프라인."""

    def __init__(self, output_dir: str, delay: float = 0.0, fail_on: str = ""):
        self.output_dir = output_dir
        self.delay = delay
        self.fail_on = fail_on
        self.calls: list[dict] = []  # 순서대로 기록
        self._n = 0

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"prompt": params.get("prompt"),
                           "seed": params.get("seed"), "loras": loras})
        if self.fail_on and self.fail_on in (params.get("prompt") or ""):
            raise RuntimeError("가짜 생성 실패")
        os.makedirs(self.output_dir, exist_ok=True)
        self._n += 1
        path = os.path.join(self.output_dir, f"fake-batch-{self._n}.png")
        with open(path, "wb") as f:
            f.write(make_png_bytes())
        return [path]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    pipe = FakePipeline(str(out))
    gen.set_pipeline(pipe)
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    batch_engine._jobs.clear()
    yield TestClient(app_module.app), pipe
    gen.set_pipeline(None)
    batch_engine._jobs.clear()


def wait_terminal(client: TestClient, job_id: str,
                  timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/batch/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "cancelled", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("배치 작업이 시간 안에 끝나지 않았어요.")


def post_batch(client: TestClient, prompts, params=None, loras=None):
    body = {"prompts": prompts, "params": params or {"steps": 20}}
    if loras is not None:
        body["loras"] = loras
    return client.post("/api/batch", json=body)


# -- 정상 흐름 ------------------------------------------------------------

def test_batch_sequential_order(client):
    c, pipe = client
    prompts = ["고양이", "강아지", "토끼"]
    r = post_batch(c, prompts)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    body = wait_terminal(c, job_id)
    assert body["status"] == "done"
    assert body["total"] == 3 and body["done"] == 3
    # 순서대로 실행됐는지 확인
    assert [call["prompt"] for call in pipe.calls] == prompts
    for item in body["items"]:
        assert item["status"] == "done"
        assert item["url"].startswith("/api/results/")
        assert item["error"] is None


def test_blank_lines_ignored(client):
    c, _ = client
    r = post_batch(c, ["  고양이  ", "", "   ", "강아지"])
    assert r.status_code == 202
    body = wait_terminal(c, r.json()["job_id"])
    assert body["total"] == 2
    assert [i["prompt"] for i in body["items"]] == ["고양이", "강아지"]


def test_fixed_seed_increments(client):
    c, pipe = client
    r = post_batch(c, ["a", "b", "c"], params={"steps": 20, "seed": 100})
    assert r.status_code == 202
    wait_terminal(c, r.json()["job_id"])
    assert [call["seed"] for call in pipe.calls] == [100, 101, 102]


def test_random_seed_not_incremented(client):
    c, pipe = client
    r = post_batch(c, ["a", "b"], params={"steps": 20, "seed": -1})
    assert r.status_code == 202
    wait_terminal(c, r.json()["job_id"])
    # 랜덤 시드는 그대로 전달 (파이프라인이 매번 새로 뽑음)
    assert [call["seed"] for call in pipe.calls] == [-1, -1]


def test_one_failure_does_not_stop_others(client):
    c, pipe = client
    pipe.fail_on = "boom"
    r = post_batch(c, ["ok1", "boom", "ok2"])
    assert r.status_code == 202
    body = wait_terminal(c, r.json()["job_id"])
    assert body["status"] == "done"
    statuses = [i["status"] for i in body["items"]]
    assert statuses == ["done", "error", "done"]
    assert "가짜 생성 실패" in body["items"][1]["error"]


def test_loras_passed_through(client):
    c, pipe = client
    loras = {"my-lora": {"enabled": True, "strength": 0.8}}
    r = post_batch(c, ["a"], loras=loras)
    assert r.status_code == 202
    wait_terminal(c, r.json()["job_id"])
    assert pipe.calls[0]["loras"] == loras


# -- 취소 -----------------------------------------------------------------

def test_cancel_batch(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    pipe = FakePipeline(str(out), delay=0.4)
    gen.set_pipeline(pipe)
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    batch_engine._jobs.clear()
    c = TestClient(app_module.app)
    try:
        r = post_batch(c, ["a", "b", "c", "d", "e"])
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        time.sleep(0.15)  # 첫 항목 실행 중에 취소
        dr = c.delete(f"/api/batch/{job_id}")
        assert dr.status_code == 200
        body = wait_terminal(c, job_id)
        assert body["status"] == "cancelled"
        # 취소 시점 이후 항목은 cancelled, 전체가 done이 아님
        assert body["done"] == body["total"]
        assert any(i["status"] == "cancelled" for i in body["items"])
        assert not all(i["status"] == "done" for i in body["items"])
    finally:
        gen.set_pipeline(None)
        batch_engine._jobs.clear()


def test_cancel_finished_job_ok(client):
    c, _ = client
    r = post_batch(c, ["a"])
    job_id = r.json()["job_id"]
    wait_terminal(c, job_id)
    dr = c.delete(f"/api/batch/{job_id}")
    assert dr.status_code == 200
    assert dr.json()["status"] == "done"  # 끝난 작업은 그대로 반환


# -- 검증·에러 --------------------------------------------------------------

def test_empty_prompts_422(client):
    c, _ = client
    r = post_batch(c, ["", "   "])
    assert r.status_code == 422


def test_too_many_prompts_422(client):
    c, _ = client
    r = post_batch(c, ["x"] * 201)
    assert r.status_code == 422


def test_invalid_params_422(client):
    c, _ = client
    r = post_batch(c, ["a"], params={"없는설정": 1})
    assert r.status_code == 422


def test_unknown_job_404(client):
    c, _ = client
    assert c.get("/api/batch/nope").status_code == 404
    assert c.delete("/api/batch/nope").status_code == 404


def test_no_pipeline_503(monkeypatch, tmp_path):
    gen.set_pipeline(None)
    monkeypatch.delenv("ATELIER_MODEL_PATH", raising=False)
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    batch_engine._jobs.clear()
    try:
        c = TestClient(app_module.app)
        r = c.post("/api/batch", json={"prompts": ["a"], "params": {}})
        assert r.status_code == 503
    finally:
        gen.set_pipeline(None)
        batch_engine._jobs.clear()


def test_busy_returns_409(client, monkeypatch):
    c, _ = client
    # 실행 중인 배치가 있는 것처럼 락을 선점
    assert batch_engine._batch_run_lock.acquire(blocking=False)
    try:
        r = post_batch(c, ["a"])
        assert r.status_code == 409
    finally:
        batch_engine._batch_run_lock.release()
