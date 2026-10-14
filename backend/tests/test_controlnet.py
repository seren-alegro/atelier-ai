"""포즈 지정(ControlNet) 테스트.

실제 생성·로딩은 모킹. torch/diffusers 없이 실행 가능.
"""

import base64
import os
import struct
import sys
import time
import types
import zlib

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.engine import controlnet as cn
from atelier.settings import defaults, get_setting, validate


def make_png_bytes() -> bytes:
    """stdlib만으로 만드는 1x1 PNG."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\x00\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def b64_png() -> str:
    return base64.b64encode(make_png_bytes()).decode("ascii")


class FakePipeline:
    """pose 모드를 흉내내는 가짜 파이프라인."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.last_pose_args: dict | None = None

    def pose(self, params: dict, control_image: str) -> list[str]:
        assert control_image and os.path.isfile(control_image)
        self.last_pose_args = {"params": params, "control_image": control_image}
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "fake-pose.png")
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


# -- 설정 ---------------------------------------------------------------

def test_controlnet_settings_registered():
    model = get_setting("controlnet_model")
    assert model is not None and model.kind == "text"
    strength = get_setting("controlnet_strength")
    assert strength is not None
    assert strength.recommended == 0.8
    assert strength.min == 0.0 and strength.max == 1.0


def test_controlnet_strength_range():
    assert "controlnet_strength" in validate({"controlnet_strength": 1.5})
    assert "controlnet_strength" in validate({"controlnet_strength": -0.1})
    assert "controlnet_strength" not in validate({"controlnet_strength": 0.8})
    assert "controlnet_strength" not in validate({"controlnet_strength": 0.0})
    assert "controlnet_strength" not in validate({"controlnet_strength": 1.0})


def test_controlnet_defaults_present():
    d = defaults()
    assert d["controlnet_model"] == ""
    assert d["controlnet_strength"] == 0.8


# -- generate 라우트 -----------------------------------------------------

def test_pose_requires_posemap(client):
    r = client.post("/api/generate",
                    json={"mode": "pose", "params": {}})
    assert r.status_code == 422
    assert "포즈맵" in r.json()["detail"]


def test_pose_bad_strength(client):
    r = client.post("/api/generate",
                    json={"mode": "pose", "params": {"controlnet_strength": 2.0},
                          "image": b64_png()})
    assert r.status_code == 422
    assert "controlnet_strength" in r.json()["detail"]


def test_pose_runs(client):
    r = client.post("/api/generate",
                    json={"mode": "pose",
                          "params": {"prompt": "1girl, solo",
                                     "controlnet_strength": 0.8},
                          "image": b64_png()})
    assert r.status_code == 202
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert len(job["results"]) == 1


# -- controlnet ref 해석 --------------------------------------------------

def test_resolve_empty_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path))
    assert cn.resolve_controlnet_ref("") == cn.DEFAULT_CONTROLNET
    assert cn.resolve_controlnet_ref(None) == cn.DEFAULT_CONTROLNET


def test_resolve_hf_id_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path))
    ref = "someone/controlnet-depth-sdxl-1.0"
    assert cn.resolve_controlnet_ref(ref) == ref


def test_resolve_local_file_by_name(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "controlnet").mkdir(parents=True)
    f = models / "controlnet" / "pose.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    assert cn.resolve_controlnet_ref("pose.safetensors") == str(f)


def test_resolve_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path))
    f = tmp_path / "elsewhere.safetensors"
    f.write_bytes(b"x")
    assert cn.resolve_controlnet_ref(str(f)) == str(f)


def test_controlnet_models_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path / "models"))
    d = cn.controlnet_models_dir()
    assert d.endswith(os.path.join("models", "controlnet"))
    assert os.path.isdir(d)


# -- load_controlnet 분기 (diffusers 모킹) ---------------------------------

def _install_fake_diffusers(monkeypatch, calls: dict):
    fake = types.ModuleType("diffusers")

    class FakeControlNetModel:
        @classmethod
        def from_pretrained(cls, ref, torch_dtype=None):
            calls["via"] = "from_pretrained"
            calls["ref"] = ref
            return cls()

        @classmethod
        def from_single_file(cls, ref, torch_dtype=None):
            calls["via"] = "from_single_file"
            calls["ref"] = ref
            return cls()

    fake.ControlNetModel = FakeControlNetModel
    monkeypatch.setitem(sys.modules, "diffusers", fake)


def test_load_controlnet_hf_uses_pretrained(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path))
    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    cn.load_controlnet("someone/controlnet-openpose-sdxl-1.0")
    assert calls["via"] == "from_pretrained"
    assert calls["ref"] == "someone/controlnet-openpose-sdxl-1.0"


def test_load_controlnet_local_file_uses_single_file(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "controlnet").mkdir(parents=True)
    f = models / "controlnet" / "pose.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    cn.load_controlnet("pose.safetensors")
    assert calls["via"] == "from_single_file"
    assert calls["ref"] == str(f)


def test_load_controlnet_empty_uses_default_pretrained(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path))
    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    cn.load_controlnet("")
    assert calls["via"] == "from_pretrained"
    assert calls["ref"] == cn.DEFAULT_CONTROLNET

def test_resolve_models_relative_path(tmp_path, monkeypatch):
    """프론트가 보내는 models dir 기준 상대경로("controlnet/pose.safetensors")."""
    models = tmp_path / "models"
    (models / "controlnet").mkdir(parents=True)
    f = models / "controlnet" / "pose.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    assert cn.resolve_controlnet_ref("controlnet/pose.safetensors") == str(f)
