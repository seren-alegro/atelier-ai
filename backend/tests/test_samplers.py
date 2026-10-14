"""samplers.py 테스트: 매핑표 무결성.

주의: torch/diffusers 없이도 돌아야 한다.
"""

import pytest

from atelier.engine.samplers import SAMPLER_MAP, resolve_sampler
from atelier.settings import get_setting


def _all_configured_samplers():
    return list(get_setting("sampler").options)


def test_every_configured_sampler_is_mapped():
    missing = [n for n in _all_configured_samplers() if n not in SAMPLER_MAP]
    assert missing == [], f"매핑 누락: {missing}"


def test_expected_sampler_names():
    assert set(SAMPLER_MAP) == {
        "er_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde",
        "euler_ancestral", "dpmpp_2m_karras", "ddim",
        "restart", "sa_solver", "deis", "lcm", "tcd",
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
    for name in ("er_sde", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "restart"):
        spec = SAMPLER_MAP[name]
        assert spec.approximate is True, name
        assert "근사 매핑, 실측 필요" in spec.note, name


def test_exact_mappings():
    assert SAMPLER_MAP["euler_ancestral"].scheduler == "EulerAncestralDiscreteScheduler"
    assert SAMPLER_MAP["ddim"].scheduler == "DDIMScheduler"
    assert SAMPLER_MAP["sa_solver"].scheduler == "SASolverScheduler"
    assert SAMPLER_MAP["deis"].scheduler == "DEISMultistepScheduler"
    assert SAMPLER_MAP["lcm"].scheduler == "LCMScheduler"
    assert SAMPLER_MAP["tcd"].scheduler == "TCDScheduler"
    for name in ("euler_ancestral", "ddim", "sa_solver", "deis", "lcm", "tcd"):
        assert SAMPLER_MAP[name].approximate is False, name


def test_karras_mapping_kwargs():
    spec = SAMPLER_MAP["dpmpp_2m_karras"]
    assert spec.kwargs.get("use_karras_sigmas") is True
    assert spec.kwargs.get("algorithm_type") == "dpmsolver++"


def test_sde_mapping_uses_sde_algorithm():
    for name in ("dpmpp_2m_sde", "dpmpp_2m_sde_gpu"):
        assert SAMPLER_MAP[name].kwargs.get("algorithm_type") == "sde-dpmsolver++"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_sampler("nope")


def test_resolve_returns_spec():
    spec = resolve_sampler("er_sde")
    assert spec.scheduler == "EulerAncestralDiscreteScheduler"
