"""포즈 자동 추출 테스트.

실제 ONNX 모델·다운로드는 모킹. numpy/PIL이 필요한 테스트는
importorskip으로 건너뛰어, minimal venv에서도 전체 수집이 깨지지 않게 한다.
torch/diffusers/onnxruntime 없이 실행 가능.
"""

import base64
import struct
import types
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import pose as pose_route


class FakeNoPersonError(ValueError):
    """minimal venv용 가짜 NoPersonError."""


def _fake_engine(detect_fn):
    return types.SimpleNamespace(
        detect_pose_skeleton=detect_fn,
        NoPersonError=FakeNoPersonError,
    )


def make_png_bytes() -> bytes:
    """stdlib만으로 만드는 1x1 PNG."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\x00\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture()
def client():
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# /api/pose-detect 엔드포인트 (검출기 모킹, numpy 불필요)
# ---------------------------------------------------------------------------

def test_pose_detect_missing_image(client):
    r = client.post("/api/pose-detect", json={})
    assert r.status_code == 422
    assert "사진" in r.json()["detail"]


def test_pose_detect_bad_base64(client):
    r = client.post("/api/pose-detect", json={"image": "!!!not-base64!!!"})
    assert r.status_code == 422


def test_pose_detect_ok(client, monkeypatch):
    fake_png = make_png_bytes()

    def fake_detect(raw: bytes) -> bytes:
        assert raw == make_png_bytes()
        return fake_png

    monkeypatch.setattr(
        pose_route, "_pose_engine", _fake_engine(fake_detect))
    r = client.post("/api/pose-detect", json={"image": _b64(make_png_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["skeleton"]) == fake_png


def test_pose_detect_no_person(client, monkeypatch):
    def fake_detect(raw: bytes) -> bytes:
        raise FakeNoPersonError("사진에서 사람을 찾지 못했어요.")

    monkeypatch.setattr(
        pose_route, "_pose_engine", _fake_engine(fake_detect))
    r = client.post("/api/pose-detect", json={"image": _b64(make_png_bytes())})
    assert r.status_code == 422
    assert "사람을 찾지 못했어요" in r.json()["detail"]


def test_pose_detect_invalid_image_bytes(client, monkeypatch):
    def fake_detect(raw: bytes) -> bytes:
        raise ValueError("이미지 파일이 올바르지 않아요.")

    monkeypatch.setattr(
        pose_route, "_pose_engine", _fake_engine(fake_detect))
    r = client.post(
        "/api/pose-detect", json={"image": _b64(b"this is not an image")}
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 아래는 numpy/PIL 필요 → 없으면 스킵
# ---------------------------------------------------------------------------

def _np():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL.Image")
    from atelier.engine import pose_detect as pd
    return np, pd


def test_decode_simcc_peak_location():
    np, pd = _np()
    K, Dx, Dy = 133, 576, 768
    sx = np.full((K, Dx), -10.0, dtype=np.float32)
    sy = np.full((K, Dy), -10.0, dtype=np.float32)
    sx[0, 100] = 10.0
    sy[0, 200] = 10.0
    out = pd.decode_simcc(sx, sy)
    assert out.shape == (K, 3)
    # split_ratio=2.0 → x=50.0, y=100.0
    assert out[0, 0] == 50.0
    assert out[0, 1] == 100.0
    # score는 x/y raw logit 최대값 중 작은 쪽
    assert out[0, 2] == 10.0
    # 피크가 없는 키포인트는 score가 -10 → 무효(-1)
    assert out[1, 0] == -1.0
    assert out[1, 1] == -1.0


def test_decode_yolox_person_box():
    np, pd = _np()
    pred = np.zeros((8400, 85), dtype=np.float64)
    # stride 16 레벨, grid (10, 20)에 raw 값 주입
    idx = 6400 + 20 * (640 // 16) + 10
    pred[idx, 0] = 0.0
    pred[idx, 1] = 0.0
    pred[idx, 2] = np.log(100.0 / 16)   # w = exp(tw)*16 = 100
    pred[idx, 3] = np.log(200.0 / 16)   # h = 200
    pred[idx, 4] = 0.99                 # obj (이미 sigmoid된 확률)
    pred[idx, 5] = 0.99                 # class 0 = person
    out = np.zeros((1, 8400, 85))
    out[0] = pred
    # cx=(10.5)*16=168, cy=(20.5)*16=328
    boxes = pd.decode_yolox(out, ratio=1.0, img_w=640, img_h=640)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert (x1, y1, x2, y2) == pytest.approx((118.0, 228.0, 218.0, 428.0))


def test_decode_yolox_no_person():
    np, pd = _np()
    out = np.zeros((1, 8400, 85))
    assert pd.decode_yolox(out, ratio=1.0, img_w=640, img_h=640) == []


def test_wholebody_to_coco18_mapping():
    np, pd = _np()
    k = np.zeros((133, 3))
    for i in range(133):
        k[i] = [i * 10.0, i * 10.0 + 1.0, 0.9]
    c = pd.wholebody_to_coco18(k)
    assert c.shape == (18, 3)
    assert c[0, 0] == 0.0    # 0 코 = wholebody 0
    assert c[2, 0] == 60.0   # 2 오른쪽 어깨 = wholebody 6
    assert c[5, 0] == 50.0   # 5 왼쪽 어깨 = wholebody 5
    assert c[1, 0] == 55.0   # 1 목 = 5,6번 중간
    assert c[8, 0] == 115.0  # 8 골반중앙 = 11,12번 중간
    assert c[11, 0] == 160.0  # 11 오른쪽 발목 = wholebody 16


def test_draw_skeleton_black_bg_white_limbs():
    np, pd = _np()
    pts = np.zeros((18, 3))
    pts[1] = [50.0, 10.0, 1.0]   # 목
    pts[8] = [50.0, 90.0, 1.0]   # 골반중앙
    img = pd.draw_skeleton((100, 100), [pts])
    assert img.size == (100, 100)
    px = np.asarray(img)
    assert (px[0, 0] == [0, 0, 0]).all()        # 배경은 검정
    assert (px[50, 50] == [255, 255, 255]).all()  # 선 위는 흰색


def test_draw_skeleton_no_persons():
    np, pd = _np()
    img = pd.draw_skeleton((64, 64), [])
    assert (np.asarray(img) == 0).all()
