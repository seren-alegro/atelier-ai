"""POST /api/upscale + 업스케일 엔진 테스트.

모델 로드/추론은 모킹한다. torch 없이 실행 가능.
"""

import base64
import io
import os
import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atelier.api import app as app_module
from atelier.api.routes import upscale as up
from atelier.engine import upscale as up_engine


def make_png_bytes(w: int = 8, h: int = 8, color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# -- 엔진 모킹 --------------------------------------------------------------

@pytest.fixture()
def stub_upscale(monkeypatch):
    """모델·device·타일 추론을 전부 스텁으로 바꾼다.

    스텁 추론은 항상 흰색(1.0) 4x 이미지를 반환 → 블렌드 로직을 검증할 수 있다.
    """
    def fake_get_upscaler(device):
        return ("fake-model", device)

    def fake_run_tiled(model, img, device, tile, overlap):
        assert model == ("fake-model", device)
        h, w = img.shape[:2]
        return np.ones((h * 4, w * 4, 3), dtype=np.float32)

    monkeypatch.setattr(up_engine, "get_upscaler", fake_get_upscaler)
    monkeypatch.setattr(up_engine, "_run_tiled", fake_run_tiled)
    monkeypatch.setattr(up_engine, "_default_device", lambda: "cpu")
    up_engine.reset_upscaler()
    yield
    up_engine.reset_upscaler()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    yield TestClient(app_module.app)


# -- 엔진: 블렌드·검증 ---------------------------------------------------------

def test_blend_logic(stub_upscale):
    """esrgan(흰색) × strength + lanczos(원본 4x) × (1-strength)."""
    out = up_engine.upscale_image(make_png_bytes(color=(0, 0, 0)), strength=0.5)
    arr = np.asarray(Image.open(io.BytesIO(out)))
    # 흰(255) * 0.5 + 검정(0) * 0.5 → 127~128
    assert arr.shape == (32, 32, 3)
    assert 126 <= arr.mean() <= 129


def test_blend_full_strength(stub_upscale):
    out = up_engine.upscale_image(make_png_bytes(color=(0, 0, 0)), strength=1.0)
    arr = np.asarray(Image.open(io.BytesIO(out)))
    assert arr.min() >= 254  # esrgan(흰색) 그대로


def test_strength_out_of_range_rejected(stub_upscale):
    with pytest.raises(ValueError, match="강도"):
        up_engine.upscale_image(make_png_bytes(), strength=0.4)
    with pytest.raises(ValueError, match="강도"):
        up_engine.upscale_image(make_png_bytes(), strength=1.2)


def test_strength_boundaries_ok(stub_upscale):
    up_engine.upscale_image(make_png_bytes(), strength=0.5)
    up_engine.upscale_image(make_png_bytes(), strength=1.0)


def test_no_gpu_raises_runtime_error(monkeypatch):
    """GPU가 없으면 RuntimeError → API에서 503으로 전달."""
    monkeypatch.setattr(
        up_engine, "_default_device",
        lambda: (_ for _ in ()).throw(
            RuntimeError("GPU를 찾을 수 없습니다.")),
    )
    with pytest.raises(RuntimeError, match="GPU"):
        up_engine.upscale_image(make_png_bytes(), device=None)


def test_rrdbnet_arch_params():
    """RRDBNet 아키텍처: num_feat=64, num_block=6, num_grow_ch=32.

    모델 가중치 없이 구조만 확인하려면 torch가 필요하므로
    torch가 없으면 스킵한다.
    """
    torch = pytest.importorskip("torch")
    model = up_engine._build_rrdbnet()
    assert model.conv_first.out_channels == 64
    assert len(model.body) == 6  # num_block
    assert model.body[0].conv1.out_channels == 32  # num_grow_ch
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 128, 128)  # 4x


# -- 엔드포인트 ---------------------------------------------------------------

def test_upscale_base64_flow(client, stub_upscale):
    r = client.post("/api/upscale",
                    json={"image": b64(make_png_bytes()), "strength": 0.7})
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    assert url.startswith("/api/results/")
    path = os.path.join(
        os.environ["ATELIER_DATA_DIR"], "outputs", url.rsplit("/", 1)[1])
    assert os.path.isfile(path)
    img = Image.open(path)
    assert img.size == (32, 32)  # 8x8 → 4x


def test_upscale_result_filename_flow(client, stub_upscale):
    """결과 파일명으로 업스케일 (갤러리에서 바로 호출)."""
    out_dir = os.path.join(os.environ["ATELIER_DATA_DIR"], "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "20261015-txt2img.png"), "wb") as f:
        f.write(make_png_bytes())
    r = client.post("/api/upscale",
                    json={"image": "20261015-txt2img.png", "strength": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["url"].startswith("/api/results/")


def test_upscale_bad_strength(client):
    r = client.post("/api/upscale",
                    json={"image": b64(make_png_bytes()), "strength": 2.0})
    assert r.status_code == 422
    assert "강도" in r.json()["detail"]


def test_upscale_bad_base64(client):
    r = client.post("/api/upscale",
                    json={"image": "!!!깨진데이터!!!", "strength": 0.5})
    assert r.status_code == 422


def test_upscale_missing_file(client):
    r = client.post("/api/upscale",
                    json={"image": "없는파일.png", "strength": 0.5})
    assert r.status_code == 404


def test_upscale_no_gpu_503(client, monkeypatch, stub_upscale):
    def boom(image_bytes, strength=0.5, tile=512, tile_overlap=16,
             device=None):
        raise RuntimeError("GPU를 찾을 수 없습니다.")
    monkeypatch.setattr(up_engine, "upscale_image", boom)
    r = client.post("/api/upscale",
                    json={"image": b64(make_png_bytes()), "strength": 0.5})
    assert r.status_code == 503
    assert "GPU" in r.json()["detail"]


def test_upscale_requires_image(client):
    r = client.post("/api/upscale", json={"image": "", "strength": 0.5})
    assert r.status_code == 422
