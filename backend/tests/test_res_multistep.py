"""res_multistep 커스텀 스케줄러 테스트 (클린룸 구현).

phi 함수 정확성 + 스케줄러 스모크 테스트.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from atelier.engine.res_multistep import (
    ResMultistepScheduler,
    phi1,
    phi2,
)
from atelier.engine.samplers import build_scheduler, resolve_sampler


def _phi_reference(j: int, h: float) -> float:
    """phi_j(-h) 직접 정의 (테스트용 오라클)."""
    z = -h
    s = sum(z ** k / math.factorial(k) for k in range(j))
    return (math.exp(z) - s) / (z ** j)


def test_phi1_matches_definition():
    for h in (0.01, 0.1, 0.5, 1.0, 2.0):
        assert phi1(h) == pytest.approx(_phi_reference(1, h), rel=1e-9)


def test_phi2_matches_definition():
    for h in (0.01, 0.1, 0.5, 1.0, 2.0):
        assert phi2(h) == pytest.approx(_phi_reference(2, h), rel=1e-9)


def test_phi_limits():
    assert phi1(0.0) == pytest.approx(1.0)
    assert phi2(0.0) == pytest.approx(0.5)


def test_res_multistep_registered():
    spec = resolve_sampler("res_multistep")
    assert spec.scheduler == "ResMultistepScheduler"
    assert spec.approximate is False


def test_build_scheduler_returns_custom():
    sched = build_scheduler("res_multistep")
    assert isinstance(sched, ResMultistepScheduler)


def test_scheduler_smoke():
    """전체 스텝을 돌려 NaN 없이 latent가 나오는지."""
    sched = ResMultistepScheduler()
    sched.set_timesteps(10)
    assert len(sched.timesteps) == 10
    assert len(sched.sigmas) == 11

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 8, 8, generator=g)
    for t in sched.timesteps:
        noise_pred = torch.randn_like(x)
        out = sched.step(noise_pred, t, x)
        x = out.prev_sample
    assert x.shape == (1, 4, 8, 8)
    assert torch.isfinite(x).all()


def test_first_step_is_deterministic():
    """같은 시드로 같은 결과가 나와야 한다."""
    def run():
        sched = ResMultistepScheduler()
        sched.set_timesteps(4)
        g = torch.Generator().manual_seed(42)
        x = torch.randn(1, 4, 4, 4, generator=g)
        outs = []
        for t in sched.timesteps:
            gn = torch.Generator().manual_seed(7)
            noise_pred = torch.randn(1, 4, 4, 4, generator=gn)
            x = sched.step(noise_pred, t, x).prev_sample
            outs.append(x.clone())
        return outs

    a, b = run(), run()
    for xa, xb in zip(a, b):
        assert torch.equal(xa, xb)


def test_monotone_denoising_trend():
    """denoised가 0이면 latent 크기가 단조 감소해야 한다 (산ity)."""
    sched = ResMultistepScheduler()
    sched.set_timesteps(8)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 8, 8, generator=g)
    norms = []
    for t in sched.timesteps:
        sigma = sched._sigma_for_timestep(t)
        # denoised = 0 → model_output = x / sigma (epsilon)
        eps = x / max(sigma, 1e-12)
        x = sched.step(eps, t, x).prev_sample
        norms.append(float(x.norm()))
    # 엄격한 단조성은 아니지만 전체적으로 감소 추세
    assert norms[-1] < norms[0]
