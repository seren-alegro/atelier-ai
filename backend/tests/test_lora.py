"""LoRA 적용 테스트. torch/diffusers 없이 실행 가능."""

import pytest

from atelier.engine.pipeline import Pipeline


class FakeDiffusersPipe:
    """load_lora_weights/set_adapters 호출을 기록하는 가짜 파이프라인."""

    def __init__(self):
        self.loaded = []  # [(path, adapter_name)]
        self.adapters_set = None  # (names, weights)
        self.disabled = False

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded.append((path, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.adapters_set = (list(names), adapter_weights)

    def disable_lora(self):
        self.disabled = True


@pytest.fixture()
def lora_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "lora").mkdir(parents=True)
    f = models / "lora" / "style.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    return f


def _pipe():
    return Pipeline(model_path="dummy", output_dir="dummy")


def test_apply_enabled_lora(lora_env):
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, {"lora/style.safetensors":
                           {"enabled": True, "strength": 0.7}})
    assert dpipe.loaded == [(str(lora_env), "atelier_lora_0")]
    assert dpipe.adapters_set == (["atelier_lora_0"], [0.7])
    assert not dpipe.disabled


def test_disabled_lora_skipped(lora_env):
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, {"lora/style.safetensors":
                           {"enabled": False, "strength": 0.7}})
    assert dpipe.loaded == []
    assert dpipe.adapters_set is None
    assert dpipe.disabled


def test_no_loras_disables(lora_env):
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, None)
    assert dpipe.disabled
    p2, dpipe2 = _pipe(), FakeDiffusersPipe()
    p2._apply_loras(dpipe2, {})
    assert dpipe2.disabled


def test_traversal_path_skipped(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, {"../secret.safetensors":
                           {"enabled": True, "strength": 0.8}})
    assert dpipe.loaded == []
    assert dpipe.disabled


def test_missing_file_skipped(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models))
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, {"lora/none.safetensors":
                           {"enabled": True, "strength": 0.8}})
    assert dpipe.loaded == []


def test_lora_load_cached(lora_env):
    """같은 파일은 두 번째부터 다시 로드하지 않는다."""
    p, dpipe = _pipe(), FakeDiffusersPipe()
    loras = {"lora/style.safetensors": {"enabled": True, "strength": 0.5}}
    p._apply_loras(dpipe, loras)
    p._apply_loras(dpipe, loras)
    assert len(dpipe.loaded) == 1
    assert dpipe.adapters_set == (["atelier_lora_0"], [0.5])


def test_multiple_loras_weights(lora_env, tmp_path):
    second = lora_env.parent / "char.safetensors"
    second.write_bytes(b"x")
    p, dpipe = _pipe(), FakeDiffusersPipe()
    p._apply_loras(dpipe, {
        "lora/style.safetensors": {"enabled": True, "strength": 0.7},
        "lora/char.safetensors": {"enabled": True, "strength": 0.3},
    })
    assert len(dpipe.loaded) == 2
    names, weights = dpipe.adapters_set
    assert weights == [0.7, 0.3]
