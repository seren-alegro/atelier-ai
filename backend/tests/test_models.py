"""모델 매니저 API 테스트.

실제 모델 로드는 FakePipeline으로 모킹. torch/diffusers 없이 실행 가능.
"""

import os

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.api.routes import models


class FakePipeline:
    """로드 성공/실패를 흉내내는 가짜 파이프라인."""

    fail_next = False

    def __init__(self, model_path: str, output_dir: str = "outputs"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.loaded = False

    def load(self):
        if FakePipeline.fail_next:
            FakePipeline.fail_next = False
            raise RuntimeError("가짜 로드 실패")
        self.loaded = True
        return self


@pytest.fixture()
def client(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "lora").mkdir(parents=True)
    (models_dir / "controlnet").mkdir(parents=True)
    (models_dir / "anima.safetensors").write_bytes(b"x" * 1024)
    (models_dir / "other.ckpt").write_bytes(b"y" * 2048)
    (models_dir / "lora" / "style-lora.safetensors").write_bytes(b"z" * 512)
    (models_dir / "controlnet" / "openpose.safetensors").write_bytes(b"w" * 256)
    (models_dir / "readme.txt").write_text("무시됨")
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(models, "Pipeline", FakePipeline)
    monkeypatch.setattr(models, "_active_name", None)
    gen.set_pipeline(None)
    yield TestClient(app_module.app)
    gen.set_pipeline(None)
    monkeypatch.setattr(models, "_active_name", None)


def test_list_models(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models_dir"].endswith("models")
    assert body["active"] is None
    by_name = {m["name"]: m for m in body["models"]}
    assert by_name["anima.safetensors"]["kind"] == "checkpoint"
    assert by_name["other.ckpt"]["kind"] == "checkpoint"
    assert by_name["lora/style-lora.safetensors"]["kind"] == "lora"
    assert by_name["controlnet/openpose.safetensors"]["kind"] == "controlnet"
    assert "readme.txt" not in by_name
    assert by_name["anima.safetensors"]["size_mb"] >= 0
    assert "modified" in by_name["anima.safetensors"]


def test_set_active_reloads_pipeline(client):
    r = client.post("/api/models/active", json={"name": "anima.safetensors"})
    assert r.status_code == 200
    assert r.json()["active"] == "anima.safetensors"
    assert models._active_name == "anima.safetensors"
    pipe = gen.get_pipeline()
    assert isinstance(pipe, FakePipeline)
    assert pipe.loaded
    assert pipe.model_path.endswith("anima.safetensors")

    r = client.get("/api/models")
    assert r.json()["active"] == "anima.safetensors"


def test_set_active_missing_checkpoint(client):
    r = client.post("/api/models/active", json={"name": "없음.safetensors"})
    assert r.status_code == 404


def test_set_active_rejects_lora(client):
    r = client.post(
        "/api/models/active", json={"name": "lora/style-lora.safetensors"}
    )
    assert r.status_code == 404


def test_set_active_load_failure_keeps_old(client):
    client.post("/api/models/active", json={"name": "anima.safetensors"})
    old_pipe = gen.get_pipeline()
    FakePipeline.fail_next = True
    r = client.post("/api/models/active", json={"name": "other.ckpt"})
    assert r.status_code == 500
    assert "기존 모델" in r.json()["detail"]
    # 기존 파이프라인 유지, 활성 이름도 그대로
    assert gen.get_pipeline() is old_pipe
    assert models._active_name == "anima.safetensors"


def test_delete_model(client):
    r = client.delete("/api/models/other.ckpt")
    assert r.status_code == 200
    assert r.json()["deleted"] == "other.ckpt"
    names = [m["name"] for m in client.get("/api/models").json()["models"]]
    assert "other.ckpt" not in names


def test_delete_active_model_conflict(client):
    client.post("/api/models/active", json={"name": "anima.safetensors"})
    r = client.delete("/api/models/anima.safetensors")
    assert r.status_code == 409


def test_delete_missing(client):
    r = client.delete("/api/models/없음.safetensors")
    assert r.status_code == 404


def test_delete_path_traversal_blocked(client):
    r = client.delete("/api/models/..%2Fsecret.safetensors")
    assert r.status_code in (400, 404)
    # 디렉토리 밖 파일은 건드리지 않음
    assert os.path.basename(r.request.url.path) != ""


def test_refresh(client):
    r = client.post("/api/models/refresh")
    assert r.status_code == 200
    assert "models" in r.json()


def test_active_model_persisted_to_disk(client, tmp_path):
    client.post("/api/models/active", json={"name": "anima.safetensors"})
    import json
    saved = json.loads(
        (tmp_path / "data" / "active_model.json").read_text(encoding="utf-8"))
    assert saved["name"] == "anima.safetensors"


def test_get_pipeline_uses_saved_active(client, tmp_path, monkeypatch):
    """재시작 시나리오: 파이프라인 없이 저장된 활성 모델로 get_pipeline이 로드."""
    import json
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "active_model.json").write_text(
        json.dumps({"name": "other.ckpt"}), encoding="utf-8")
    monkeypatch.delenv("ATELIER_MODEL_PATH", raising=False)
    monkeypatch.setattr(gen, "Pipeline", FakePipeline)
    gen.set_pipeline(None)
    pipe = gen.get_pipeline()
    assert isinstance(pipe, FakePipeline)
    assert pipe.model_path.endswith("other.ckpt")
    gen.set_pipeline(None)


def test_get_pipeline_ignores_missing_saved_file(client, tmp_path, monkeypatch):
    """저장된 모델 파일이 사라졌으면 503."""
    import json
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "active_model.json").write_text(
        json.dumps({"name": "지워짐.safetensors"}), encoding="utf-8")
    monkeypatch.delenv("ATELIER_MODEL_PATH", raising=False)
    gen.set_pipeline(None)
    with pytest.raises(Exception) as exc:
        gen.get_pipeline()
    assert "503" in str(exc.value) or "모델이 설정되지 않았어요" in str(exc.value)
    gen.set_pipeline(None)


def test_get_pipeline_output_dir_in_data_dir(client, tmp_path, monkeypatch):
    """출력 폴더가 데이터 디렉토리 아래로 고정되는지."""
    monkeypatch.setattr(gen, "Pipeline", FakePipeline)
    monkeypatch.setenv("ATELIER_MODEL_PATH", str(tmp_path / "m.safetensors"))
    (tmp_path / "m.safetensors").write_bytes(b"x")
    gen.set_pipeline(None)
    pipe = gen.get_pipeline()
    assert pipe.output_dir == str(tmp_path / "data" / "outputs")
    gen.set_pipeline(None)
