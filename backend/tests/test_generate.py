"""POST /api/generate + 작업 큐 + 결과 조회 테스트.

실제 생성은 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
"""

import base64
import os
import struct
import time
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.settings import defaults


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
    """실제 diffusers 없이 동작하는 가짜 파이프라인."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        assert "denoise_strength" not in params or isinstance(
            params["denoise_strength"], (int, float))
        return self._write("t")

    def img2img(self, params: dict, image: str,
                loras: dict | None = None) -> list[str]:
        assert image and os.path.isfile(image)
        return self._write("i")

    def inpaint(self, params: dict, image: str, mask: str,
                loras: dict | None = None) -> list[str]:
        assert image and os.path.isfile(image)
        assert mask and os.path.isfile(mask)
        return self._write("p")

    def _write(self, tag: str) -> list[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"fake-{tag}.png")
        with open(path, "wb") as f:
            f.write(make_png_bytes())
        return [path]


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


def wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("작업이 시간 안에 끝나지 않았어요.")


def b64_png() -> str:
    return base64.b64encode(make_png_bytes()).decode("ascii")


# -- 검증 ---------------------------------------------------------------

def test_generate_validation_error(client):
    r = client.post("/api/generate",
                    json={"mode": "txt2img", "params": {"steps": 10}})
    assert r.status_code == 422
    assert "steps" in r.json()["detail"]


def test_generate_unknown_param(client):
    r = client.post("/api/generate",
                    json={"mode": "txt2img", "params": {"없는설정": 1}})
    assert r.status_code == 422


def test_generate_unknown_mode(client):
    r = client.post("/api/generate",
                    json={"mode": "그리기", "params": {}})
    assert r.status_code == 422


def test_generate_img2img_requires_image(client):
    r = client.post("/api/generate",
                    json={"mode": "img2img", "params": {}})
    assert r.status_code == 422
    assert "원본 이미지" in r.json()["detail"]


def test_generate_inpaint_requires_mask(client):
    r = client.post("/api/generate",
                    json={"mode": "inpaint", "params": {},
                          "image": b64_png()})
    assert r.status_code == 422
    assert "마스크" in r.json()["detail"]


def test_generate_bad_base64(client):
    r = client.post("/api/generate",
                    json={"mode": "img2img", "params": {},
                          "image": "이건-base64가-아님!!!"})
    assert r.status_code == 422


# -- 작업 흐름 -----------------------------------------------------------

def test_generate_txt2img_flow(client):
    r = client.post("/api/generate",
                    json={"mode": "txt2img", "params": defaults()})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = wait_done(client, job_id)
    assert job["status"] == "done"
    assert job["results"] == ["/api/results/fake-t.png"]
    assert job["error"] is None

    # 결과 이미지 서빙
    r = client.get("/api/results/fake-t.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_generate_img2img_flow(client):
    r = client.post("/api/generate",
                    json={"mode": "img2img", "params": defaults(),
                          "image": b64_png()})
    assert r.status_code == 202
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["results"] == ["/api/results/fake-i.png"]


def test_generate_inpaint_flow(client):
    r = client.post("/api/generate",
                    json={"mode": "inpaint", "params": defaults(),
                          "image": b64_png(), "mask": b64_png()})
    assert r.status_code == 202
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["results"] == ["/api/results/fake-p.png"]


def test_job_not_found(client):
    r = client.get("/api/jobs/없는잡")
    assert r.status_code == 404


def test_result_not_found(client):
    r = client.get("/api/results/없는파일.png")
    assert r.status_code == 404


def test_result_traversal_blocked(client):
    r = client.get("/api/results/..%2Fsecret.png")
    assert r.status_code in (400, 404)


def test_no_pipeline_503(monkeypatch, tmp_path):
    gen.set_pipeline(None)
    monkeypatch.delenv("ATELIER_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    c = TestClient(app_module.app)
    try:
        r = c.post("/api/generate",
                   json={"mode": "txt2img", "params": {}})
        assert r.status_code == 503
        assert "ATELIER_MODEL_PATH" in r.json()["detail"]
    finally:
        gen.set_pipeline(None)


def test_upload_too_large_rejected(client, monkeypatch):
    """50MB 초과 업로드는 413."""
    monkeypatch.setattr(gen, "MAX_UPLOAD_BYTES", 10)
    big = base64.b64encode(b"x" * 11).decode()
    r = client.post("/api/generate",
                    json={"mode": "img2img", "params": {}, "image": big})
    assert r.status_code == 413


def test_upload_cleaned_after_job(client):
    """작업 완료 후 업로드 원본 파일이 삭제된다."""
    import glob
    img = base64.b64encode(make_png_bytes()).decode()
    r = client.post("/api/generate",
                    json={"mode": "img2img", "params": {}, "image": img})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # 백그라운드 스레드 완료 대기
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    leftovers = glob.glob(os.path.join("uploads", f"{job_id}-*.png"))
    assert leftovers == []


def test_old_jobs_pruned(client):
    """1시간 지난 완료 job은 새 요청 시 정리된다."""
    old_id = "old-job-123"
    gen._jobs[old_id] = {
        "job_id": old_id, "mode": "txt2img", "params": {},
        "status": "done", "created_at": time.time() - 7200,
        "results": [], "error": None,
    }
    r = client.post("/api/generate",
                    json={"mode": "txt2img", "params": {}})
    assert r.status_code == 202
    assert old_id not in gen._jobs


def test_ensure_loaded_calls_load_once():
    """_ensure_loaded: load()가 있으면 호출, 멱등이라 여러 번 안전."""
    calls = []

    class Loadable:
        def load(self):
            calls.append(1)
            return self

    gen._ensure_loaded(Loadable())
    gen._ensure_loaded(Loadable())
    assert len(calls) == 2  # 호출은 되지만 실제 Pipeline.load()는 내부에서 한 번만 수행


def test_ensure_loaded_without_load_method():
    """load()가 없는 파이프라인(FakePipeline)도 통과."""
    gen._ensure_loaded(object())  # 예외 없어야 함
