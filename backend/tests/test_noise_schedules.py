"""노이즈 스케줄 (ComfyUI식 스케줄러 선택) 테스트."""

import math

import pytest

torch = pytest.importorskip("torch")

from atelier.engine.samplers import (
    NOISE_SCHEDULES,
    _NoiseScheduleWrapper,
    apply_scheduler,
    make_noise_sigmas,
)


def _native(steps=20, sigma_max=14.6):
    # 네이티브 스케줄 흉내 (감소 곡선)
    return torch.linspace(sigma_max, 0.01, steps + 1)


@pytest.mark.parametrize("schedule", [s for s in NOISE_SCHEDULES
                                      if s != "auto"])
def test_schedule_shape_and_monotone(schedule):
    steps = 20
    sigs = make_noise_sigmas(schedule, _native(steps), steps)
    if schedule == "beta":
        # beta는 ComfyUI처럼 중복 인덱스를 제거해서 짧아질 수 있음
        assert 2 <= len(sigs) <= steps + 1
    else:
        assert len(sigs) == steps + 1
    assert sigs[-1] == pytest.approx(0.0)
    assert sigs[0] > 0
    for a, b in zip(sigs, sigs[1:]):
        assert a >= b, f"{schedule}: not monotone ({a} < {b})"


def test_schedule_first_close_to_sigma_max():
    steps = 20
    for schedule in ("normal", "karras", "simple", "sgm_uniform",
                     "exponential", "ddim_uniform", "beta",
                     "linear_quadratic", "bong_tangent"):
        sigs = make_noise_sigmas(schedule, _native(steps), steps)
        assert sigs[0] == pytest.approx(14.6, rel=0.1), schedule


def test_beta_ppf_properties():
    from atelier.engine.samplers import _beta_ppf, _betai
    # 대칭성: I_0.5(0.6, 0.6) = 0.5
    assert _betai(0.6, 0.6, 0.5) == pytest.approx(0.5, abs=1e-6)
    assert _beta_ppf(0.5, 0.6, 0.6) == pytest.approx(0.5, abs=1e-6)
    # 역함수 관계
    for p in (0.1, 0.3, 0.7, 0.9):
        assert _betai(0.6, 0.6, _beta_ppf(p, 0.6, 0.6)) == pytest.approx(
            p, abs=1e-6)
    # 단조성
    vals = [_beta_ppf(p / 10, 0.6, 0.6) for p in range(1, 10)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_karras_exponential_endpoints():
    from atelier.engine.samplers import make_noise_sigmas
    steps = 20
    for schedule in ("karras", "exponential"):
        sigs = make_noise_sigmas(schedule, _native(steps), steps)
        assert sigs[0] == pytest.approx(14.6, rel=1e-6), schedule
        assert sigs[-2] > 0, schedule  # 마지막 0 앞은 양수


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        make_noise_sigmas("bogus", _native(10), 10)


class _FakeScheduler:
    """sigmas 파라미터 미지원 구식 스케줄러 흉내 (폴백 경로 테스트)."""

    def __init__(self):
        self.num_train_timesteps = 1000
        self.sigmas = None
        self.timesteps = None
        self.calls = []

    def set_timesteps(self, num_inference_steps: int):
        self.calls.append(num_inference_steps)
        self.sigmas = torch.linspace(14.6, 0.0, num_inference_steps + 1)
        self.timesteps = (self.sigmas[:-1] * 1000).long()


class _ModernScheduler(_FakeScheduler):
    """set_timesteps(sigmas=...) 지원 스케줄러 흉내."""

    def set_timesteps(self, num_inference_steps=None, sigmas=None):
        if sigmas is not None:
            assert num_inference_steps is None
            self.calls.append(f"sigmas:{len(sigmas)}")
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
            self.timesteps = (self.sigmas[:-1] * 1000).long()
        else:
            super().set_timesteps(num_inference_steps)


def test_wrapper_auto_passthrough():
    inner = _FakeScheduler()
    w = _NoiseScheduleWrapper(inner, "auto")
    w.set_timesteps(10)
    assert inner.calls == [10]
    assert len(inner.sigmas) == 11


def test_wrapper_applies_custom_schedule_modern():
    inner = _ModernScheduler()
    w = _NoiseScheduleWrapper(inner, "simple")
    w.set_timesteps(10)
    # 네이티브 1회 + 커스텀 sigmas 1회
    assert inner.calls == [10, "sigmas:11"]
    assert len(inner.sigmas) == 11
    assert float(inner.sigmas[-1]) == pytest.approx(0.0)


def test_wrapper_applies_custom_schedule_fallback():
    inner = _FakeScheduler()
    w = _NoiseScheduleWrapper(inner, "linear_quadratic")
    w.set_timesteps(10)
    assert inner.calls == [10]  # 구식은 네이티브 1회만
    assert len(inner.sigmas) == 11
    assert float(inner.sigmas[-1]) == pytest.approx(0.0)
    assert len(inner.timesteps) == 10


def test_wrapper_delegates_attributes():
    inner = _FakeScheduler()
    inner.custom_attr = "hello"
    w = _NoiseScheduleWrapper(inner, "simple")
    assert w.custom_attr == "hello"


def test_apply_scheduler_wraps_only_when_needed():
    class Pipe:
        def __init__(self):
            from diffusers import EulerDiscreteScheduler
            self.scheduler = EulerDiscreteScheduler()

    diffusers = pytest.importorskip("diffusers")
    p1 = Pipe()
    apply_scheduler(p1, "euler", "auto")
    assert not isinstance(p1.scheduler, _NoiseScheduleWrapper)

    p2 = Pipe()
    apply_scheduler(p2, "euler", "simple")
    assert isinstance(p2.scheduler, _NoiseScheduleWrapper)
    p2.scheduler.set_timesteps(10)
    assert len(p2.scheduler.sigmas) == 11
    assert len(p2.scheduler.timesteps) == 10
