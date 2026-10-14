"""POST /api/train + /api/train-jobs + /api/caption 테스트.

실제 학습은 모킹. torch/diffusers 없이 실행 가능.
"""

import os
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import train as train_module


def make_png_bytes() -> bytes:
    """stdlib만으로 만드는 1x1 PNG."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\x00\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 모델 폴더를 임시 디렉토리로 격리
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models_dir))
    # 체크포인트 가짜 파일
    ckpt = models_dir / "base.safetensors"
    ckpt.write_bytes(b"fake")
    return TestClient(app_module.create_app())


def test_train_validates_dataset_dir(client):
    r = client.post("/api/train", json={
        "model": "base.safetensors",
        "dataset_dir": "/nonexistent-dir-xyz",
    })
    assert r.status_code == 422
    assert "데이터셋" in r.json()["detail"]


def test_train_rejects_missing_model(client):
    r = client.post("/api/train", json={
        "model": "nope.safetensors",
        "dataset_dir": "/tmp",
    })
    assert r.status_code in (400, 404)


def test_train_rejects_empty_dataset(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = client.post("/api/train", json={
        "model": "base.safetensors",
        "dataset_dir": str(empty),
    })
    assert r.status_code == 422


def test_train_accepts_and_reports_progress(client, tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.png").write_bytes(make_png_bytes())
    (ds / "b.png").write_bytes(make_png_bytes())

    # 실제 학습 대신 on_step 콜백만 호출하는 가짜
    calls = []
    fake_out = tmp_path / "lora.safetensors"
    fake_out.write_bytes(b"fake-lora")

    def fake_train(cfg):
        if cfg.on_step:
            for s in (1, 2, 3):
                cfg.on_step(s, 10, 0.5)
                calls.append(s)
        return str(fake_out)

    monkeypatch.setattr(train_module, "train_lora", fake_train)

    r = client.post("/api/train", json={
        "model": "base.safetensors",
        "dataset_dir": str(ds),
        "steps_per_image": 5,
        "output_name": "test-lora",
    })
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["total_steps"] == 2 * 5

    # 백그라운드 스레드가 끝날 때까지 폴링
    job_id = body["job_id"]
    final = None
    for _ in range(100):
        jr = client.get(f"/api/train-jobs/{job_id}")
        assert jr.status_code == 200
        final = jr.json()
        if final["status"] in ("done", "error"):
            break
    assert final["status"] == "done", final
    assert final["step"] == 3
    assert final["total_steps"] == 10
    assert final["results"] == ["test-lora.safetensors"]
    assert calls == [1, 2, 3]


def test_train_job_404(client):
    r = client.get("/api/train-jobs/does-not-exist")
    assert r.status_code == 404


def test_caption_validates_method(client):
    r = client.post("/api/caption", json={
        "image_path": "/tmp/x.png", "method": "nope"})
    assert r.status_code == 422


def test_caption_missing_file(client):
    r = client.post("/api/caption", json={
        "image_path": "/tmp/no-such-file.png", "method": "wd14"})
    assert r.status_code == 422


def test_caption_batch_missing_dir(client):
    r = client.post("/api/caption/batch", json={
        "dataset_dir": "/nonexistent-dir-xyz", "method": "wd14"})
    assert r.status_code == 422


def test_caption_batch_writes_txt(client, tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.png").write_bytes(make_png_bytes())
    (ds / "b.txt").write_text("existing")  # 이미지는 없지만 .txt만 있음

    monkeypatch.setattr(train_module, "caption_image",
                        lambda path, method="wd14": "1girl, solo")

    r = client.post("/api/caption/batch", json={
        "dataset_dir": str(ds), "method": "wd14"})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] == ["a.png"]
    assert (ds / "a.txt").read_text() == "1girl, solo"

    # 두 번째 실행: overwrite 없이 건너뜀
    r2 = client.post("/api/caption/batch", json={
        "dataset_dir": str(ds), "method": "wd14"})
    assert r2.json()["skipped"] == ["a.png"]
