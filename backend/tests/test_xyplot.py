"""POST /api/xy-plot + X/Y 플롯 잡 테스트.

실제 생성은 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
batch.py의 잡 인프라를 재사용하므로 락/취소 기본 동작은 test_batch.py 참조.
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
    """파라미터 오버라이드를 기록하는 가짜 파이프라인."""

    def __init__(self, output_dir: str, delay: float = 0.0):
        self.output_dir = output_dir
        self.delay = delay
        self.calls: list[dict] = []
        self._n = 0

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(dict(params))
        os.makedirs(self.output_dir, exist_ok=True)
        self._n += 1
        path = os.path.join(self.output_dir, f"fake-xy-{self._n}.png")
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
    # 테스트용 파이프 접근
    client_holder = {}
    yield TestClient(app_module.app), pipe, client_holder
    gen.set_pipeline(None)
    batch_engine._jobs.clear()


def start(client, body) -> dict:
    r = client.post("/api/xy-plot", json=body)
    assert r.status_code == 202, r.text
    return r.json()


def wait_done(client, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/xy-plot/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "error", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("X/Y 플롯이 시간 안에 끝나지 않았어요.")


def base_params(**kw):
    p = {"prompt": "1girl, solo", "seed": 42, "steps": 25}
    p.update(kw)
    return p


# -- 검증 ---------------------------------------------------------------

def test_xyplot_bad_axis_type(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "model", "values": ["a"]},
        "base_params": base_params()})
    assert r.status_code == 422


def test_xyplot_empty_values(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "cfg_scale", "values": []},
        "base_params": base_params()})
    assert r.status_code == 422


def test_xyplot_too_many_cells(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "seed", "values": list(range(10))},
        "y_axis": {"type": "seed", "values": list(range(10))},
        "base_params": base_params()})
    assert r.status_code == 422  # 100 > 36


def test_xyplot_bad_sampler_value(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "sampler", "values": ["not_a_sampler"]},
        "base_params": base_params()})
    assert r.status_code == 422


def test_xyplot_bad_cfg_value(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "cfg_scale", "values": [99]},
        "base_params": base_params()})
    assert r.status_code == 422


def test_xyplot_bad_seed_rule(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "seed", "values": [1]},
        "base_params": base_params(), "seed_rule": "weird"})
    assert r.status_code == 422


def test_xyplot_bad_base_params(client):
    c, _, _ = client
    r = c.post("/api/xy-plot", json={
        "x_axis": {"type": "seed", "values": [1]},
        "base_params": {"no_such_key": 1}})
    assert r.status_code == 422


def test_xyplot_not_found(client):
    c, _, _ = client
    assert c.get("/api/xy-plot/nope").status_code == 404
    assert c.delete("/api/xy-plot/nope").status_code == 404


# -- 실행 ---------------------------------------------------------------

def test_xyplot_two_axes_flow(client):
    c, pipe, _ = client
    body = start(c, {
        "x_axis": {"type": "cfg_scale", "values": [3, 5]},
        "y_axis": {"type": "steps", "values": [20, 30]},
        "base_params": base_params(),
    })
    job = wait_done(c, body["job_id"])
    assert job["status"] == "done"
    assert job["total"] == 4
    assert job["meta"]["x_labels"] == ["3", "5"]
    assert job["meta"]["y_labels"] == ["20", "30"]
    assert job["meta"]["x_type"] == "cfg_scale"
    # 라벨이 붙어 있다
    assert job["items"][0]["label"] == "CFG 3 / 스텝 20"
    # 축 오버라이드가 셀마다 적용됐다 (y 바깥, x 안쪽 순)
    cfgs = [call["cfg_scale"] for call in pipe.calls]
    steps = [call["steps"] for call in pipe.calls]
    assert cfgs == [3, 5, 3, 5]
    assert steps == [20, 20, 30, 30]
    # 전부 URL이 있다
    assert all(i["status"] == "done" and i["url"] for i in job["items"])


def test_xyplot_fixed_seed_default(client):
    c, pipe, _ = client
    body = start(c, {
        "x_axis": {"type": "cfg_scale", "values": [3, 5, 7]},
        "base_params": base_params(seed=123),
    })
    job = wait_done(c, body["job_id"])
    assert job["status"] == "done"
    # 비교용: 모든 셀에 같은 시드
    assert [call["seed"] for call in pipe.calls] == [123, 123, 123]


def test_xyplot_seed_axis_wins(client):
    c, pipe, _ = client
    body = start(c, {
        "x_axis": {"type": "seed", "values": [11, 22]},
        "base_params": base_params(seed=99),
    })
    job = wait_done(c, body["job_id"])
    assert job["status"] == "done"
    assert [call["seed"] for call in pipe.calls] == [11, 22]


def test_xyplot_increment_seed_rule(client):
    c, pipe, _ = client
    body = start(c, {
        "x_axis": {"type": "cfg_scale", "values": [3, 5]},
        "base_params": base_params(seed=7),
        "seed_rule": "increment",
    })
    job = wait_done(c, body["job_id"])
    assert job["status"] == "done"
    assert [call["seed"] for call in pipe.calls] == [7, 8]


def test_xyplot_single_axis(client):
    c, pipe, _ = client
    body = start(c, {
        "x_axis": {"type": "sampler", "values": ["ddim", "euler_ancestral"]},
        "base_params": base_params(),
    })
    job = wait_done(c, body["job_id"])
    assert job["status"] == "done"
    assert job["total"] == 2
    assert job["meta"]["y_labels"] == []
    assert [call["sampler"] for call in pipe.calls] == [
        "ddim", "euler_ancestral"]


def test_xyplot_cancel(client):
    c, pipe_slow, _ = client
    # 느린 파이프로 교체
    out = pipe_slow.output_dir
    slow = FakePipeline(out, delay=0.3)
    gen.set_pipeline(slow)
    body = start(c, {
        "x_axis": {"type": "seed", "values": list(range(6))},
        "base_params": base_params(),
    })
    r = c.delete(f"/api/xy-plot/{body['job_id']}")
    assert r.status_code == 200
    job = wait_done(c, body["job_id"])
    assert job["status"] == "cancelled"
