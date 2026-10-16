"""samplers.py 테스트: 매핑표 무결성.

주의: torch/diffusers 없이도 돌아야 한다.
"""

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.engine.samplers import (
    SAMPLER_ALIASES,
    SAMPLER_MAP,
    normalize_sampler_name,
    resolve_sampler,
)
from atelier.settings import get_setting


def _all_configured_samplers():
    return list(get_setting("sampler").options)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    yield TestClient(app_module.app)


def test_every_configured_sampler_is_mapped():
    missing = [n for n in _all_configured_samplers() if n not in SAMPLER_MAP]
    assert missing == [], f"매핑 누락: {missing}"


def test_expected_sampler_names():
    assert set(SAMPLER_MAP) == {
        "er_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde",
        "euler_ancestral", "dpmpp_2m_karras", "ddim",
        "restart", "sa_solver", "deis", "lcm", "tcd",
        # A1111 전체 (추가)
        "euler", "lms", "heun", "dpm2", "dpm2_ancestral",
        "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_karras",
        "dpmpp_2m", "dpmpp_sde", "dpmpp_sde_karras",
        "dpmpp_2m_sde_karras", "dpmpp_2m_sde_heun",
        "dpmpp_2m_sde_heun_karras",
        "dpmpp_3m_sde", "dpmpp_3m_sde_karras",
        "dpm_fast", "dpm_adaptive", "plms", "unipc",
    }


def test_spec_fields_wellformed():
    for name, spec in SAMPLER_MAP.items():
        assert isinstance(spec.scheduler, str) and spec.scheduler, name
        assert spec.scheduler.endswith("Scheduler"), name
        assert isinstance(spec.kwargs, dict), name
        assert isinstance(spec.approximate, bool), name
        assert isinstance(spec.note, str) and spec.note, name


def test_approximate_mappings_disclose():
    # 근사 매핑은 note에 반드시 '근사 매핑, 실측 필요' 문구를 포함
    approx = {
        "er_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "restart",
        "dpmpp_sde", "dpmpp_sde_karras",
        "dpmpp_2m_sde_karras", "dpmpp_2m_sde_heun",
        "dpmpp_2m_sde_heun_karras",
        "dpmpp_3m_sde", "dpmpp_3m_sde_karras",
        "dpm_fast", "dpm_adaptive",
    }
    for name in approx:
        spec = SAMPLER_MAP[name]
        assert spec.approximate is True, name
        assert "근사 매핑, 실측 필요" in spec.note, name
    # 나머지는 정확 매핑
    for name, spec in SAMPLER_MAP.items():
        if name not in approx:
            assert spec.approximate is False, name


def test_exact_mappings():
    assert SAMPLER_MAP["euler_ancestral"].scheduler == "EulerAncestralDiscreteScheduler"
    assert SAMPLER_MAP["euler"].scheduler == "EulerDiscreteScheduler"
    assert SAMPLER_MAP["ddim"].scheduler == "DDIMScheduler"
    assert SAMPLER_MAP["sa_solver"].scheduler == "SASolverScheduler"
    assert SAMPLER_MAP["deis"].scheduler == "DEISMultistepScheduler"
    assert SAMPLER_MAP["lcm"].scheduler == "LCMScheduler"
    assert SAMPLER_MAP["tcd"].scheduler == "TCDScheduler"
    assert SAMPLER_MAP["dpm2"].scheduler == "KDPM2DiscreteScheduler"
    assert SAMPLER_MAP["dpm2_ancestral"].scheduler == "KDPM2AncestralDiscreteScheduler"
    assert SAMPLER_MAP["dpmpp_2s_ancestral"].scheduler == "DPMSolverSinglestepScheduler"
    assert SAMPLER_MAP["lms"].scheduler == "LMSDiscreteScheduler"
    assert SAMPLER_MAP["heun"].scheduler == "HeunDiscreteScheduler"
    assert SAMPLER_MAP["plms"].scheduler == "PNDMScheduler"
    assert SAMPLER_MAP["unipc"].scheduler == "UniPCMultistepScheduler"
    assert SAMPLER_MAP["dpmpp_2m"].kwargs == {
        "algorithm_type": "dpmsolver++", "solver_order": 2}
    assert SAMPLER_MAP["dpmpp_3m_sde"].kwargs["solver_order"] == 3


def test_karras_mapping_kwargs():
    for name in ("dpmpp_2m_karras", "dpmpp_2s_ancestral_karras",
                 "dpmpp_sde_karras", "dpmpp_2m_sde_karras",
                 "dpmpp_2m_sde_heun_karras", "dpmpp_3m_sde_karras"):
        spec = SAMPLER_MAP[name]
        assert spec.kwargs.get("use_karras_sigmas") is True, name


def test_sde_mapping_uses_sde_algorithm():
    for name in ("dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_sde",
                 "dpmpp_sde_karras", "dpmpp_2m_sde_karras",
                 "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_karras",
                 "dpmpp_3m_sde", "dpmpp_3m_sde_karras"):
        assert SAMPLER_MAP[name].kwargs.get("algorithm_type") == "sde-dpmsolver++", name


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_sampler("nope")


def test_resolve_returns_spec():
    spec = resolve_sampler("er_sde")
    assert spec.scheduler == "EulerAncestralDiscreteScheduler"


# -- 별칭 ---------------------------------------------------------------


def test_all_alias_targets_valid():
    bad = {a: t for a, t in SAMPLER_ALIASES.items() if t not in SAMPLER_MAP}
    assert bad == {}, f"잘못된 별칭 대상: {bad}"


def test_a1111_display_names_covered():
    for display in ("Euler a", "Euler", "DPM++ 2M Karras", "DPM++ 2S a",
                    "DPM++ 2M SDE", "DPM++ 3M SDE Karras", "DPM2 a",
                    "DPM fast", "DPM adaptive", "DDIM", "PLMS", "UniPC",
                    "LMS", "Heun", "DPM2"):
        assert display in SAMPLER_ALIASES, display


def test_comfyui_names_covered():
    for name in ("euler", "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral",
                 "dpmpp_2s_ancestral_cfgpp", "dpmpp_2m_sde_gpu",
                 "dpmpp_3m_sde_gpu", "uni_pc", "uni_pc_bh2", "euler_cfg_pp",
                 "ddim_cfg_pp"):
        assert name in SAMPLER_ALIASES, name


def test_normalize_a1111_display_names():
    assert normalize_sampler_name("Euler a") == "euler_ancestral"
    assert normalize_sampler_name("DPM++ 2M Karras") == "dpmpp_2m_karras"
    assert normalize_sampler_name("DPM++ 2M SDE Heun Karras") == "dpmpp_2m_sde_heun_karras"
    assert normalize_sampler_name("DPM fast") == "dpm_fast"


def test_normalize_case_and_space_insensitive():
    assert normalize_sampler_name("EULER A") == "euler_ancestral"
    assert normalize_sampler_name("  dpm++ 2m KARRAS ") == "dpmpp_2m_karras"
    assert normalize_sampler_name("UniPC") == "unipc"
    assert normalize_sampler_name("uni_pc_bh2") == "unipc"


def test_normalize_direct_keys():
    assert normalize_sampler_name("euler_ancestral") == "euler_ancestral"
    assert normalize_sampler_name("DDIM") == "ddim"
    assert normalize_sampler_name("er_sde") == "er_sde"


def test_normalize_unknown_returns_none():
    assert normalize_sampler_name("nope") is None
    assert normalize_sampler_name("") is None
    assert normalize_sampler_name(None) is None
    assert normalize_sampler_name("   ") is None


def test_sampler_aliases_endpoint(client):
    r = client.get("/api/sampler-aliases")
    assert r.status_code == 200
    data = r.json()
    assert data["aliases"]["Euler a"] == "euler_ancestral"
    assert data["aliases"]["uni_pc_bh2"] == "unipc"
    assert set(data["keys"]) == set(SAMPLER_MAP)
    assert set(data["keys"]) == set(get_setting("sampler").options)


def test_samplers_endpoint(client):
    r = client.get("/api/samplers")
    assert r.status_code == 200
    data = r.json()
    assert set(data) == set(SAMPLER_MAP)
    assert set(data) == set(get_setting("sampler").options)
    entry = data["dpmpp_2m_karras"]
    assert entry["scheduler"] == "DPMSolverMultistepScheduler"
    assert entry["approximate"] is False
    assert isinstance(entry["note"], str) and entry["note"]
    approx_entry = data["dpmpp_2m_sde_heun"]
    assert approx_entry["approximate"] is True
    assert "근사 매핑, 실측 필요" in approx_entry["note"]
