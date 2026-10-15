"""Anima 3.8b 커스텀 파이프라인 테스트.

실제 가중치 파일이 없으므로 모킹 기반. torch는 CPU tiny 텐서로 사용.
덤프(anima38_dump.json)의 실제 키 이름은 CONFIRMED 키 테스트에 사용.
"""

import os
import subprocess
import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import generate as gen
from atelier.api.routes import models
from atelier.engine import anima38 as a38


# -- 모듈 import 규약 ---------------------------------------------------------

def test_import_without_torch():
    """모듈 import 시 torch/transformers를 끌어오지 않는다 (pipeline.py 패턴)."""
    code = (
        "import sys;"
        "import atelier.engine.anima38;"
        "assert 'torch' not in sys.modules, 'torch가 import됨';"
        "assert 'transformers' not in sys.modules, 'transformers가 import됨';"
        "print('ok')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


# -- tiny forward -------------------------------------------------------------

def _tiny_cfg():
    return a38.AnimaDiTConfig(
        num_layers=2, hidden_size=64, mlp_dim=128,
        qkv_rank=8, qkv_up_dim=192, num_heads=6, head_dim=32,
        cond_dim=32, patch_size=2, in_channels=4,
        connector_dim=32, resampler_dim=48, resampler_mlp_dim=72,
        te35_hidden=80, te06_hidden=32,
    )


def test_dit_forward_tiny():
    torch = pytest.importorskip("torch")
    dit = a38.build_anima_dit(_tiny_cfg())
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([10])
    cond = torch.randn(1, 5, 32)
    h, (b, gh, gw) = dit(x, t, cond)
    assert h.shape == (1, 16, 64)  # 8x8, patch 2 → 16 토큰
    assert (b, gh, gw) == (1, 4, 4)
    # cond 없이도 동작
    h2, _ = dit(x, t, None)
    assert h2.shape == (1, 16, 64)


def test_connector_forward_tiny():
    torch = pytest.importorskip("torch")
    conn = a38.build_connector_v2(_tiny_cfg())
    out = conn(torch.randn(2, 10, 80))
    assert out.shape == (2, 32, 32)  # (B, 쿼리수, connector_dim)


# -- 덤프 실측 키 매핑 ----------------------------------------------------------

# 덤프 sample_keys에서 가져온 실제 키 이름 (tiny shape로 축소)
REAL_CONNECTOR_KEYS = {
    "net.anima_v2_connector.quality_anchor.layer_mix_logits": (6, 4),
    "net.anima_v2_connector.quality_anchor.query_norms.0.weight": (64,),
    "net.anima_v2_connector.quality_anchor.query_norms.5.weight": (64,),
    "net.anima_v2_connector.quality_anchor.semantic_attentions.0.k_norm.weight": (64,),
    "net.anima_v2_connector.quality_anchor.semantic_attentions.0.k_proj.weight": (32, 80),
    "net.anima_v2_connector.quality_anchor.semantic_attentions.0.o_proj.weight": (32, 32),
}


def test_confirmed_key_mapping(tmp_path):
    """덤프의 실제 키 이름이 prefix 제거 후 DiT 모듈에 매핑된다."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    p = tmp_path / "conn.safetensors"
    save_file(
        {k: torch.randn(*shape) for k, shape in REAL_CONNECTOR_KEYS.items()},
        str(p),
    )
    # 커넥터는 DiT의 서브모듈이므로 전체 DiT에 로드한다
    dit = a38.build_anima_dit(_tiny_cfg())
    state = a38.load_safetensors_mapped(str(p), prefix_strip="net.")
    rep = a38.apply_state_dict(dit, state)
    assert rep["unexpected"] == []
    for k in REAL_CONNECTOR_KEYS:
        short = k[len("net."):]
        assert short not in rep["missing"], f"매핑 실패: {short}"


def test_remap_adapter_keys():
    """어댑터 파일 최상위 키 → 모듈 키 remap."""
    state = {
        "layer_mix_logits": 1,
        "query_norms.0.weight": 2,
        "source_norms.3.weight": 3,
        "semantic_attentions.0.k_proj.weight": 4,
    }
    out = a38.remap_adapter_keys(state)
    assert out["quality_anchor.layer_mix_logits"] == 1
    assert out["quality_anchor.query_norms.0.weight"] == 2
    assert out["quality_anchor.source_norms.3.weight"] == 3
    assert out["quality_anchor.semantic_attentions.0.k_proj.weight"] == 4


def test_dit_block_key_roundtrip(tmp_path):
    """DiT 블록 내부 키(구현체 정의)는 저장→로드 라운드트립이 된다."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    dit = a38.build_anima_dit(_tiny_cfg())
    sd = dit.state_dict()
    subset = {k: v for k, v in sd.items() if k.startswith("blocks.0.")}
    assert "blocks.0.attn.q_down.weight" in subset  # factorized QKV (덤프 shape 실측)
    assert "blocks.0.attn.q_up.weight" in subset
    assert "blocks.0.mlp_fc1.weight" in subset
    p = tmp_path / "dit.safetensors"
    save_file(subset, str(p))
    dit2 = a38.build_anima_dit(_tiny_cfg())
    state = a38.load_safetensors_mapped(str(p))
    rep = a38.apply_state_dict(dit2, state)
    assert rep["unexpected"] == []
    for k in subset:
        assert k not in rep["missing"]


# -- FP8 폴백 -------------------------------------------------------------------

def test_fp8_dequantize(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    t = torch.randn(8, 16).to(torch.float8_e4m3fn)
    p = tmp_path / "fp8.safetensors"
    save_file({"w": t, "b": torch.randn(8)}, str(p))
    out = a38.dequantize_fp8_state_dict(str(p), "bfloat16")
    assert out["w"].dtype == torch.bfloat16
    assert torch.isfinite(out["w"]).all()
    assert out["b"].dtype == torch.float32  # FP8 아니면 그대로


def test_qwen35_load_fallback(tmp_path):
    """transformers 직접 로드가 실패하면 FP8 dequantize 폴백으로 간다.

    의존성 주입(_model_cls/_config_cls)으로 테스트 — transformers의
    lazy 모듈에는 monkeypatch가 안정적으로 동작하지 않아 이 방식을 쓴다.
    """
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("FP8 미지원 시뮬레이션")

        def __init__(self, config):
            self.config = config
            self.loaded_keys = []

        def load_state_dict(self, state, strict=False):
            self.loaded_keys = sorted(state.keys())
            return ([], [])

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return object()

    # prefix 없는 덤프 실측 키 형태
    save_file(
        {"embed_tokens.weight": torch.randn(32, 16).to(torch.float8_e4m3fn),
         "layers.0.input_layernorm.weight": torch.randn(16)},
        str(tmp_path / "te.safetensors"),
    )

    model, info = a38.load_qwen35_te(
        str(tmp_path / "te.safetensors"),
        _model_cls=FakeModel, _config_cls=FakeConfig,
    )
    assert info["fp8_fallback"] is True
    # prefix 보정 확인
    assert "model.embed_tokens.weight" in model.loaded_keys
    assert "model.layers.0.input_layernorm.weight" in model.loaded_keys
    assert model.loaded_keys[0].startswith("model.")


# -- 토크나이저 분리 -------------------------------------------------------------

def test_select_tokenizer_dirs():
    tok35, tok06 = a38.select_tokenizer_dirs(
        ["qwen35_tokenizer", "qwen3_06b_tokenizer"])
    assert tok35 == "qwen35_tokenizer"
    assert tok06 == "qwen3_06b_tokenizer"
    # 순서가 바뀌어도 35가 먼저
    tok35, tok06 = a38.select_tokenizer_dirs(
        ["qwen3_06b_tokenizer", "qwen35_tokenizer"])
    assert tok35 == "qwen35_tokenizer"


def test_select_tokenizer_dirs_missing():
    with pytest.raises(RuntimeError, match="토크나이저 디렉토리가 2개 필요해요"):
        a38.select_tokenizer_dirs(["qwen35_tokenizer"])
    with pytest.raises(RuntimeError, match="토크나이저 디렉토리가 2개 필요해요"):
        a38.select_tokenizer_dirs([])


# -- 팩 감지 ---------------------------------------------------------------------

def _make_pack(base: str, name: str = "anima38pack1", partial: bool = False):
    pack = os.path.join(base, name)
    os.makedirs(pack, exist_ok=True)
    if not partial:
        for f in ("Anima-3.8B-v1.1.safetensors", "qwen35_4b.safetensors",
                  "qwen_3_06b_base.safetensors",
                  "Anima-3.8B-expanded_adapter.safetensors"):
            with open(os.path.join(pack, f), "wb") as fh:
                fh.write(b"x" * 64)
        os.makedirs(os.path.join(pack, "qwen35_tokenizer"), exist_ok=True)
        os.makedirs(os.path.join(pack, "qwen3_06b_tokenizer"), exist_ok=True)
    else:
        # DiT만 있는 불완전 팩
        with open(os.path.join(pack, "Anima-3.8B-test.safetensors"),
                  "wb") as fh:
            fh.write(b"x" * 64)
    return pack


def test_scan_finds_pack(tmp_path, monkeypatch):
    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir, exist_ok=True)
    pack = _make_pack(models_dir)
    monkeypatch.setenv("ATELIER_MODELS_DIR", models_dir)

    items = models._scan()
    packs = [m for m in items if m["kind"] == "anima38pack"]
    assert len(packs) == 1
    assert packs[0]["name"] == "anima38pack1"
    st = packs[0]["pack_status"]
    assert st["qwen35_4b"] is True
    assert st["qwen3_06b"] is True
    assert st["adapter"] is not None
    assert len(st["tokenizer_dirs"]) == 2
    # 팩 안의 파일은 checkpoint로 중복 표시되지 않음
    assert not [m for m in items
                if m["kind"] == "checkpoint" and "anima38pack1" in m["name"]]
    # 절대 경로 판정
    assert models.is_anima38_pack_dir(pack) is True


def test_scan_partial_pack_still_listed(tmp_path, monkeypatch):
    """파일이 부족해도 목록·상태 표시는 된다."""
    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir, exist_ok=True)
    _make_pack(models_dir, name="partial-3.8", partial=True)
    monkeypatch.setenv("ATELIER_MODELS_DIR", models_dir)

    items = models._scan()
    packs = [m for m in items if m["kind"] == "anima38pack"]
    assert len(packs) == 1
    st = packs[0]["pack_status"]
    assert st["qwen35_4b"] is False
    assert st["tokenizer_dirs"] == []


# -- 파이프라인 동작 ---------------------------------------------------------------

class FakePipelineOK:
    def __init__(self, *a, **k):
        self.output_dir = "outputs"

    def load(self):
        return self


@pytest.fixture()
def client(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    _make_pack(str(models_dir), name="partial-3.8", partial=True)
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(models, "_active_name", None)
    # GPU 없음 환경을 가정: device 확인을 cpu로 고정해 팩 파일 검증까지 진행
    import atelier.engine.pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "_default_device", lambda: "cpu")
    gen.set_pipeline(None)
    yield TestClient(app_module.app)
    gen.set_pipeline(None)


def test_set_active_pack_missing_files_500_korean(client):
    """팩 파일이 부족하면 500 + 한국어 에러 (로드 실패 시 한국어 에러)."""
    r = client.post("/api/models/active", json={"name": "partial-3.8"})
    assert r.status_code == 500
    assert "모델 로드에 실패했어요" in r.json()["detail"]
    assert "qwen35_4b" in r.json()["detail"]


def test_pack_load_no_gpu_raises_runtimeerror(tmp_path, monkeypatch):
    """GPU가 없으면 load()에서 RuntimeError (→ API 503)."""
    import atelier.engine.pipeline as pipe_mod

    monkeypatch.setattr(
        pipe_mod, "_default_device",
        lambda: (_ for _ in ()).throw(
            RuntimeError("GPU를 찾을 수 없습니다. 테스트")))
    pack = _make_pack(str(tmp_path), name="p")
    pipe = a38.Anima38Pipeline(pack_dir=pack, output_dir=str(tmp_path / "o"))
    with pytest.raises(RuntimeError, match="GPU를 찾을 수 없습니다"):
        pipe.load()


def test_ensure_loaded_converts_to_503():
    """RuntimeError → 503 변환 (generate._ensure_loaded)."""

    class BadPipe:
        def load(self):
            raise RuntimeError("GPU를 찾을 수 없습니다. 테스트")

    with pytest.raises(HTTPException) as ei:
        gen._ensure_loaded(BadPipe())
    assert ei.value.status_code == 503


def test_unsupported_modes_raise():
    pipe = a38.Anima38Pipeline(pack_dir="/nonexistent")
    with pytest.raises(RuntimeError, match="아직 지원하지 않아요"):
        pipe.img2img({})
    with pytest.raises(RuntimeError, match="아직 지원하지 않아요"):
        pipe.inpaint({})
    with pytest.raises(RuntimeError, match="아직 지원하지 않아요"):
        pipe.pose({})
