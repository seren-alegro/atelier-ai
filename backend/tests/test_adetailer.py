"""POST /api/adetailer + 얼굴 보정 엔진 테스트.

YOLO·inpaint는 모킹한다. ultralytics/torch 없이 실행 가능.
"""

import base64
import io
import os

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from atelier.api import app as app_module
from atelier.api.routes import adetailer as ad_route
from atelier.engine import adetailer as ad_engine


def make_png_bytes(w: int = 64, h: int = 64, color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# -- YOLO 모킹 ---------------------------------------------------------------

class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """predict()가 고정 bbox를 돌려주는 가짜 검출기."""
    def __init__(self, boxes=(((10, 10, 30, 30, 0.9),))):
        self._boxes = boxes
        self.calls = []

    def predict(self, img, conf=0.3, device=None, verbose=False):
        self.calls.append(
            {"conf": conf, "device": device, "shape": np.asarray(img).shape})
        xyxy = [b[:4] for b in self._boxes]
        cf = [b[4] for b in self._boxes]
        return [_FakeResult(_FakeBoxes(xyxy, cf))]


@pytest.fixture()
def stub_detector(monkeypatch, boxes=(((10, 10, 30, 30, 0.9),))):
    """YOLO·device를 스텁으로 바꾼다. boxes로 검출 결과를 조절."""
    fake = FakeYOLO(boxes)
    monkeypatch.setattr(ad_engine, "get_detector", lambda device: fake)
    monkeypatch.setattr(ad_engine, "_default_device", lambda: "cpu")
    ad_engine.reset_detector()
    yield fake
    ad_engine.reset_detector()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    # 실제 파이프라인 로드를 막는다 (모델 없음 → 503 대신 더미 주입)
    monkeypatch.setattr(ad_route, "get_pipeline", lambda: object())
    yield TestClient(app_module.app)


class FakePipeline:
    """inpaint 호출을 기록하는 가짜 파이프라인."""
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.calls = []

    def inpaint(self, params, image_path, mask_path, loras=None):
        self.calls.append({
            "params": dict(params),
            "image_path": image_path,
            "mask_path": mask_path,
            "loras": loras,
            "mask": np.asarray(Image.open(mask_path)),
            "image": np.asarray(Image.open(image_path)),
        })
        os.makedirs(self.out_dir, exist_ok=True)
        out = os.path.join(self.out_dir, "fake-inpaint.png")
        Image.fromarray(self.calls[-1]["image"]).save(out)
        return [out]


# -- 엔진: 얼굴 검출 -------------------------------------------------------------

def test_detect_faces_parses_boxes(stub_detector):
    boxes = ad_engine.detect_faces(make_png_bytes(), conf=0.3)
    assert len(boxes) == 1
    x1, y1, x2, y2, c = boxes[0]
    assert (x1, y1, x2, y2) == (10.0, 10.0, 30.0, 30.0)
    assert c == pytest.approx(0.9)  # float32 정밀도
    # conf/device가 predict에 그대로 전달되는지
    assert stub_detector.calls[0]["conf"] == 0.3
    assert stub_detector.calls[0]["device"] == "cpu"


def test_detect_faces_sorted_by_conf(monkeypatch):
    fake = FakeYOLO(boxes=(
        (0, 0, 10, 10, 0.4),
        (20, 20, 40, 40, 0.95),
    ))
    monkeypatch.setattr(ad_engine, "get_detector", lambda device: fake)
    monkeypatch.setattr(ad_engine, "_default_device", lambda: "cpu")
    boxes = ad_engine.detect_faces(make_png_bytes())
    assert [b[4] for b in boxes] == pytest.approx([0.95, 0.4])


def test_detect_faces_conf_out_of_range(stub_detector):
    with pytest.raises(ValueError, match="신뢰도"):
        ad_engine.detect_faces(make_png_bytes(), conf=1.5)


def test_detect_faces_bad_image(stub_detector):
    with pytest.raises(ValueError, match="이미지"):
        ad_engine.detect_faces(b"not-an-image")


# -- 엔진: 마스크 -----------------------------------------------------------------

def test_build_face_mask_covers_padded_box():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    mask = ad_engine.build_face_mask(
        img, [(10, 10, 30, 30, 0.9)], padding=8, blur=0)
    m = np.asarray(mask)
    assert m.shape == (64, 64)
    # padding 8 → (2,2)-(38,38) 영역이 흰색
    assert m[20, 20] == 255
    assert m[2, 2] == 255
    assert m[39, 39] == 0
    assert m[0, 0] == 0


def test_build_face_mask_feathering():
    """blur>0이면 경계가 feathering되어 중간값이 생긴다."""
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    sharp = np.asarray(ad_engine.build_face_mask(
        img, [(10, 10, 30, 30, 0.9)], padding=8, blur=0))
    soft = np.asarray(ad_engine.build_face_mask(
        img, [(10, 10, 54, 54, 0.9)], padding=4, blur=8))
    assert set(np.unique(sharp)) <= {0, 255}
    assert len(np.unique(soft)) > 2  # 중간값 존재
    assert soft[32, 32] == 255  # 중심은 그대로


def test_build_face_mask_multiple_boxes_merged():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    mask = ad_engine.build_face_mask(
        img, [(5, 5, 15, 15, 0.9), (40, 40, 55, 55, 0.8)],
        padding=4, blur=0)
    m = np.asarray(mask)
    assert m[10, 10] == 255 and m[47, 47] == 255
    assert m[30, 30] == 0  # 두 얼굴 사이


def test_build_face_mask_bad_padding():
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    with pytest.raises(ValueError, match="확장"):
        ad_engine.build_face_mask(img, [(1, 1, 5, 5, 0.9)], padding=-1)


# -- 엔진: retouch -----------------------------------------------------------------

def test_retouch_calls_inpaint_once(stub_detector, tmp_path):
    pipe = FakePipeline(str(tmp_path))
    out, faces = ad_engine.retouch(
        make_png_bytes(), "1girl, smile", "lowres",
        denoising_strength=0.4, mask_padding=8, mask_blur=0,
        pipeline=pipe)
    assert faces == 1
    assert out.endswith("fake-inpaint.png")
    assert len(pipe.calls) == 1
    call = pipe.calls[0]
    assert call["params"]["prompt"] == "1girl, smile"
    assert call["params"]["negative_prompt"] == "lowres"
    assert call["params"]["denoise_strength"] == 0.4
    # 원본 크기 유지 (기본값으로 화면비가 깨지면 안 됨)
    assert call["params"]["height"] == 64
    assert call["params"]["width"] == 64
    # 마스크가 얼굴 영역을 덮는가
    assert call["mask"][20, 20] == 255
    assert call["mask"][0, 0] == 0


def test_retouch_empty_prompt_uses_fallback(stub_detector, tmp_path):
    pipe = FakePipeline(str(tmp_path))
    ad_engine.retouch(make_png_bytes(), "   ", pipeline=pipe)
    assert pipe.calls[0]["params"]["prompt"] == ad_engine.FALLBACK_PROMPT


def test_retouch_no_faces(stub_detector, tmp_path):
    stub_detector._boxes = ()
    pipe = FakePipeline(str(tmp_path))
    out, faces = ad_engine.retouch(make_png_bytes(), "1girl", pipeline=pipe)
    assert (out, faces) == (None, 0)
    assert pipe.calls == []  # inpaint 호출 없음


def test_retouch_bad_denoising(stub_detector, tmp_path):
    pipe = FakePipeline(str(tmp_path))
    with pytest.raises(ValueError, match="디노이즈"):
        ad_engine.retouch(make_png_bytes(), "x",
                          denoising_strength=1.5, pipeline=pipe)
    with pytest.raises(ValueError, match="디노이즈"):
        ad_engine.retouch(make_png_bytes(), "x",
                          denoising_strength=-0.1, pipeline=pipe)


def test_retouch_no_gpu(monkeypatch):
    monkeypatch.setattr(
        ad_engine, "_default_device",
        lambda: (_ for _ in ()).throw(
            RuntimeError("GPU를 찾을 수 없습니다.")),
    )
    with pytest.raises(RuntimeError, match="GPU"):
        ad_engine.retouch(make_png_bytes(), "x", pipeline=object())


def test_retouch_needs_pipeline(stub_detector):
    with pytest.raises(RuntimeError, match="파이프라인"):
        ad_engine.retouch(make_png_bytes(), "x", pipeline=None)


# -- 엔드포인트 ---------------------------------------------------------------------

def test_adetailer_flow(client, tmp_path, monkeypatch):
    """얼굴 2개 검출 → inpaint 결과 URL + faces_detected."""
    out_dir = os.path.join(os.environ["ATELIER_DATA_DIR"], "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fake_out = os.path.join(out_dir, "20261015-inpaint-s1-0-abc.png")
    with open(fake_out, "wb") as f:
        f.write(make_png_bytes())
    monkeypatch.setattr(
        ad_engine, "retouch", lambda *a, **k: (fake_out, 2))
    r = client.post("/api/adetailer", json={
        "image": b64(make_png_bytes()),
        "prompt": "1girl",
        "denoising_strength": 0.4,
        "conf": 0.5,
        "mask_padding": 32,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["faces_detected"] == 2
    assert data["url"] == "/api/results/20261015-inpaint-s1-0-abc.png"


def test_adetailer_no_faces_filename(client, monkeypatch):
    """얼굴 없음 + 파일명 입력 → 원본 URL 그대로, faces_detected 0."""
    out_dir = os.path.join(os.environ["ATELIER_DATA_DIR"], "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "20261015-txt2img.png"), "wb") as f:
        f.write(make_png_bytes())
    monkeypatch.setattr(
        ad_engine, "retouch", lambda *a, **k: (None, 0))
    r = client.post("/api/adetailer", json={
        "image": "20261015-txt2img.png", "prompt": "1girl"})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "url": "/api/results/20261015-txt2img.png",
        "faces_detected": 0,
    }


def test_adetailer_no_faces_base64_saves_copy(client, monkeypatch):
    """얼굴 없음 + base64 입력 → 복사본 URL 반환."""
    monkeypatch.setattr(
        ad_engine, "retouch", lambda *a, **k: (None, 0))
    r = client.post("/api/adetailer", json={
        "image": b64(make_png_bytes()), "prompt": "1girl"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["faces_detected"] == 0
    assert data["url"].startswith("/api/results/")
    name = data["url"].rsplit("/", 1)[1]
    assert "adetailer-noface" in name
    assert os.path.isfile(os.path.join(
        os.environ["ATELIER_DATA_DIR"], "outputs", name))


def test_adetailer_bad_denoising(client):
    r = client.post("/api/adetailer", json={
        "image": b64(make_png_bytes()), "denoising_strength": 1.5})
    assert r.status_code == 422
    assert "디노이즈" in r.json()["detail"]


def test_adetailer_bad_conf(client):
    r = client.post("/api/adetailer",
                    json={"image": b64(make_png_bytes()), "conf": -0.1})
    assert r.status_code == 422


def test_adetailer_negative_padding(client):
    r = client.post("/api/adetailer",
                    json={"image": b64(make_png_bytes()), "mask_padding": -5})
    assert r.status_code == 422


def test_adetailer_bad_base64(client):
    r = client.post("/api/adetailer", json={"image": "!!!깨진데이터!!!"})
    assert r.status_code == 422


def test_adetailer_missing_file(client):
    r = client.post("/api/adetailer", json={"image": "없는파일.png"})
    assert r.status_code == 404


def test_adetailer_no_gpu_503(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("GPU를 찾을 수 없습니다.")
    monkeypatch.setattr(ad_engine, "retouch", boom)
    r = client.post("/api/adetailer",
                    json={"image": b64(make_png_bytes())})
    assert r.status_code == 503
    assert "GPU" in r.json()["detail"]


def test_adetailer_no_model_503(client, monkeypatch):
    """파이프라인 미설정 → get_pipeline이 503."""
    def no_pipe():
        raise HTTPException(status_code=503, detail="모델이 설정되지 않았어요.")
    monkeypatch.setattr(ad_route, "get_pipeline", no_pipe)
    r = client.post("/api/adetailer",
                    json={"image": b64(make_png_bytes())})
    assert r.status_code == 503
