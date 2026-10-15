"""POST /api/outpaint + 아웃페인팅 엔진 테스트.

실제 인페인트는 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
PIL은 필요 (Pillow는 백엔드 의존성에 포함).
"""

import base64
import os
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.engine import outpaint as outpaint_engine


def make_png_bytes(w: int = 64, h: int = 64) -> bytes:
    """stdlib만으로 만드는 w×h PNG."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class FakePipeline:
    """inpaint 호출 파라미터를 기록하는 가짜 파이프라인."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.calls: list[dict] = []
        self._n = 0

    def inpaint(self, params: dict, image: str, mask: str,
                loras: dict | None = None) -> list[str]:
        from PIL import Image
        assert image and os.path.isfile(image)
        assert mask and os.path.isfile(mask)
        # 마스크가 원본 영역(검정)과 새 영역(흰색)을 모두 포함하는지
        m = Image.open(mask).convert("L")
        px = list(m.getdata())
        assert min(px) < 128 and max(px) > 128
        self.calls.append({
            "params": dict(params),
            "image_size": Image.open(image).size,
            "mask_size": m.size,
        })
        os.makedirs(self.output_dir, exist_ok=True)
        self._n += 1
        path = os.path.join(self.output_dir, f"fake-outpaint-{self._n}.png")
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
    yield TestClient(app_module.app), pipe
    gen.set_pipeline(None)


def b64_png() -> str:
    return base64.b64encode(make_png_bytes()).decode("ascii")


def expand_all(v: int = 64) -> dict:
    return {"up": v, "down": v, "left": v, "right": v}


# -- 엔진 단위 --------------------------------------------------------

def test_validate_expand_ok():
    assert outpaint_engine.validate_expand(expand_all(64)) == expand_all(64)
    assert outpaint_engine.validate_expand(
        {"up": 512, "down": 64, "left": 100, "right": 200}
    ) == {"up": 512, "down": 64, "left": 100, "right": 200}


def test_validate_expand_too_small():
    with pytest.raises(ValueError):
        outpaint_engine.validate_expand(
            {"up": 32, "down": 64, "left": 64, "right": 64})


def test_validate_expand_too_big():
    with pytest.raises(ValueError):
        outpaint_engine.validate_expand(
            {"up": 64, "down": 64, "left": 513, "right": 64})


def test_validate_expand_not_int():
    with pytest.raises(ValueError):
        outpaint_engine.validate_expand(
            {"up": "64", "down": 64, "left": 64, "right": 64})


def test_outpaint_image_canvas_size(tmp_path):
    pipe = FakePipeline(str(tmp_path / "outputs"))
    out = outpaint_engine.outpaint_image(
        make_png_bytes(64, 64), expand_all(64), pipe,
        prompt="night sky", denoise_strength=0.5, mask_blur=4)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    call = pipe.calls[0]
    # 64 + 64 + 64 = 192
    assert call["image_size"] == (192, 192)
    assert call["mask_size"] == (192, 192)
    assert call["params"]["width"] == 192
    assert call["params"]["height"] == 192
    assert call["params"]["denoise_strength"] == 0.5
    assert call["params"]["prompt"] == "night sky"


def test_outpaint_image_prompt_fallback(tmp_path):
    pipe = FakePipeline(str(tmp_path / "outputs"))
    outpaint_engine.outpaint_image(
        make_png_bytes(), expand_all(64), pipe, prompt="  ")
    assert pipe.calls[0]["params"]["prompt"]


def test_outpaint_image_bad_denoise(tmp_path):
    pipe = FakePipeline(str(tmp_path / "outputs"))
    with pytest.raises(ValueError):
        outpaint_engine.outpaint_image(
            make_png_bytes(), expand_all(64), pipe, denoise_strength=1.5)


# -- API --------------------------------------------------------------

def test_outpaint_flow(client):
    c, pipe = client
    r = c.post("/api/outpaint", json={
        "image": b64_png(),
        "expand": {"up": 64, "down": 64, "left": 128, "right": 128},
        "prompt": "starry night",
        "denoise_strength": 0.6,
        "mask_blur": 8,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"].startswith("/api/results/")
    # 64 + 128 + 128 = 320 가로, 64 + 64 + 64 = 192 세로
    assert pipe.calls[0]["image_size"] == (320, 192)


def test_outpaint_bad_expand(client):
    c, _ = client
    r = c.post("/api/outpaint", json={
        "image": b64_png(), "expand": {"up": 10, "down": 64,
                                       "left": 64, "right": 64}})
    assert r.status_code == 422


def test_outpaint_bad_denoise(client):
    c, _ = client
    r = c.post("/api/outpaint", json={
        "image": b64_png(), "expand": expand_all(),
        "denoise_strength": 2.0})
    assert r.status_code == 422


def test_outpaint_bad_blur(client):
    c, _ = client
    r = c.post("/api/outpaint", json={
        "image": b64_png(), "expand": expand_all(), "mask_blur": 100})
    assert r.status_code == 422


def test_outpaint_empty_image(client):
    c, _ = client
    r = c.post("/api/outpaint", json={"image": "", "expand": expand_all()})
    assert r.status_code == 422


def test_outpaint_missing_file(client):
    c, _ = client
    r = c.post("/api/outpaint", json={
        "image": "no-such-file.png", "expand": expand_all()})
    assert r.status_code == 404


def test_outpaint_no_pipeline(monkeypatch):
    """파이프라인 미설정 시 503 (fail-fast)."""
    gen.set_pipeline(None)
    monkeypatch.delenv("ATELIER_MODEL_PATH", raising=False)
    from atelier.api.routes import models as models_module
    monkeypatch.setattr(models_module, "get_saved_active_path",
                        lambda: None)
    c = TestClient(app_module.app)
    r = c.post("/api/outpaint", json={
        "image": b64_png(), "expand": expand_all()})
    assert r.status_code == 503
