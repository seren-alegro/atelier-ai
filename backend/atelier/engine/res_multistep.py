"""RES multistep (res_2m 계열) 커스텀 스케줄러 — 클린룸 구현.

ComfyUI/RES4LYF의 res_2m 아이디어를 diffusers 스케줄러 인터페이스
(set_timesteps / timesteps / step)에 맞게 재구현한 것이다.
exponential integrator + phi 가중 multistep은 공개된 수치해석 기법이며,
이 파일의 코드는 원본을 복사하지 않고 수식에서 직접 작성했다.

핵심 수식 (2차 exponential multistep):
    h      = -log(σ_next / σ)      # 현재 스텝 시간 간격
    h_prev = -log(σ / σ_prev)      # 이전 스텝 시간 간격
    c2     = -h_prev / h
    b2     = φ₂(-h) / c2
    b1     = φ₁(-h) - b2
    ε₁     = D(x, σ) - x           # 현재 denoised (x0 예측)
    ε₂     = D_prev - x            # 이전 스텝 denoised
    x_next = e^(-h)·x + h·(b₁·ε₁ + b₂·ε₂)

    φ₁(-h) = (1 - e^(-h)) / h
    φ₂(-h) = (e^(-h) - 1 + h) / h²

첫 스텝(이전 denoised가 없음)은 1차 exponential euler로 폴백한다:
    x_next = e^(-h)·x + h·φ₁(-h)·ε₁

이 모듈은 torch 없이 import 가능해야 한다 (torch는 메서드 안에서 import).
"""

from __future__ import annotations

import math


def phi1(h: float) -> float:
    """φ₁(-h) = (1 - e^(-h)) / h. h→0에서는 테일러 전개."""
    if abs(h) < 1e-6:
        return 1.0 - h / 2.0
    return (1.0 - math.exp(-h)) / h


def phi2(h: float) -> float:
    """φ₂(-h) = (e^(-h) - 1 + h) / h². h→0에서는 테일러 전개."""
    if abs(h) < 1e-6:
        return 0.5 - h / 6.0
    return (math.exp(-h) - 1.0 + h) / (h * h)


class ResMultistepStepOutput:
    """diffusers 스케줄러 출력과 같은 형태 (prev_sample)."""

    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


class ResMultistepScheduler:
    """RES 2차 multistep 스케줄러 (diffusers 호환 인터페이스).

    - set_timesteps(num_inference_steps): 시그마 스케줄 생성
    - timesteps: 샘플링 루프용 타임스텝 (sigma * 1000)
    - step(model_output, timestep, sample): epsilon 예측 → 다음 latent
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 sigma_max: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_max = sigma_max
        self.sigmas = None
        self.timesteps = None
        self._step_index = 0
        self._prev_denoised = None

    def set_timesteps(self, num_inference_steps: int):
        import torch

        # simple 스케줄: sigma_max → 0 선형
        self.sigmas = torch.linspace(
            self.sigma_max, 0.0, num_inference_steps + 1)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).long()
        self._step_index = 0
        self._prev_denoised = None
        return self

    def _sigma_for_timestep(self, timestep) -> float:
        """timestep 값으로 현재 sigma를 찾는다 (부동소수점 오차 허용)."""
        import torch

        t = float(timestep) if not hasattr(timestep, "item") else float(
            timestep.item() if timestep.numel() == 1 else timestep.flatten()[0])
        target = t / self.num_train_timesteps
        idx = int(torch.argmin((self.sigmas - target).abs()).item())
        # timesteps는 sigmas[:-1]에 대응하므로 마지막 sigma(0)는 제외
        idx = min(idx, len(self.sigmas) - 2)
        self._step_index = idx
        return float(self.sigmas[idx].item())

    def step(self, model_output, timestep, sample):
        """한 스텝 진행. model_output은 epsilon 예측."""
        import torch

        sigma = self._sigma_for_timestep(timestep)
        sigma_next = float(self.sigmas[self._step_index + 1].item())

        # epsilon → x0 (denoised)
        denoised = sample - sigma * model_output

        h = -math.log(max(sigma_next, 1e-12) / max(sigma, 1e-12))
        eps_1 = denoised - sample

        if self._prev_denoised is None or self._step_index == 0:
            # 첫 스텝: 1차 exponential euler 폴백
            x_next = (math.exp(-h) * sample
                      + h * phi1(h) * eps_1)
        else:
            sigma_prev = float(self.sigmas[self._step_index - 1].item())
            h_prev = -math.log(max(sigma, 1e-12) / max(sigma_prev, 1e-12))
            c2 = -h_prev / h if h != 0 else 0.0
            if c2 == 0:
                x_next = (math.exp(-h) * sample
                          + h * phi1(h) * eps_1)
            else:
                b2 = phi2(h) / c2
                b1 = phi1(h) - b2
                eps_2 = self._prev_denoised - sample
                x_next = (math.exp(-h) * sample
                          + h * (b1 * eps_1 + b2 * eps_2))

        self._prev_denoised = denoised.detach().clone() \
            if isinstance(denoised, torch.Tensor) else denoised
        self._step_index += 1
        return ResMultistepStepOutput(prev_sample=x_next)
