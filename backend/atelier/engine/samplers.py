"""샘플러 이름 → diffusers 스케줄러 매핑.

settings.py의 `sampler` / `experimental_sampler` 설정값을
diffusers 스케줄러로 변환한다.

원칙:
- diffusers에 직접 대응되는 것은 정확 매핑 (approximate=False)
- diffusers에 없는 것(Kohya/ComfyUI 계열 SDE 등)은 가장 가까운 스케줄러로
  매핑하고 approximate=True + note에 "근사 매핑, 실측 필요"를 명시한다.
- 매핑 근거가 불확실한 것은 지어내지 않고 note에 불확실함을 적는다.

이 모듈은 diffusers/torch 없이 import 가능해야 한다
(스케줄러 클래스는 문자열로 보관, 실제 로딩은 build_scheduler에서 lazy import).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplerSpec:
    scheduler: str                  # diffusers 스케줄러 클래스명
    kwargs: dict = field(default_factory=dict)  # 스케줄러 생성 인자
    approximate: bool = False       # 근사 매핑 여부
    note: str = ""                  # 설명 / 주의사항


SAMPLER_MAP: dict[str, SamplerSpec] = {
    # --- 기본 샘플러 (settings.py `sampler`) ---
    "euler_ancestral": SamplerSpec(
        scheduler="EulerAncestralDiscreteScheduler",
        note="정확 대응. diffusers의 확률적 오일러 샘플러.",
    ),
    "dpmpp_2m_karras": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "dpmsolver++", "solver_order": 2,
                "use_karras_sigmas": True},
        note="정확 대응. DPM-Solver++ 2M + Karras 시그마.",
    ),
    "dpmpp_2m_sde": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. diffusers의 sde-dpmsolver++는 동일 알고리즘 "
             "계열이나 Kohya/ComfyUI 구현체와 노이즈 처리 등 세부 구현이 다를 수 있음.",
    ),
    "dpmpp_2m_sde_gpu": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. dpmpp_2m_sde와 동일 알고리즘이며 "
             "_gpu 접미사는 ComfyUI의 노이즈 생성 위치 차이일 뿐이므로 동일 매핑.",
    ),
    "er_sde": SamplerSpec(
        scheduler="EulerAncestralDiscreteScheduler",
        approximate=True,
        note="근사 매핑, 실측 필요. er_sde의 정확한 알고리즘 사양 미확인 "
             "(Kohya/ComfyUI 계열 SDE 샘플러로 추정). diffusers의 대표 확률적 "
             "스케줄러로 매핑했으므로 Anima 실측 후 조정 필요.",
    ),
    "ddim": SamplerSpec(
        scheduler="DDIMScheduler",
        note="정확 대응.",
    ),
    # --- 실험실 샘플러 (settings.py `experimental_sampler`) ---
    "restart": SamplerSpec(
        scheduler="HeunDiscreteScheduler",
        approximate=True,
        note="근사 매핑, 실측 필요. Restart 샘플링(Karras et al.)의 반복 "
             "노이즈 주입 사이클은 diffusers 미지원. 2차 Heun으로 근사.",
    ),
    "sa_solver": SamplerSpec(
        scheduler="SASolverScheduler",
        note="diffusers 네이티브 지원 (Stochastic Adams Solver).",
    ),
    "deis": SamplerSpec(
        scheduler="DEISMultistepScheduler",
        note="diffusers 네이티브 지원.",
    ),
    "lcm": SamplerSpec(
        scheduler="LCMScheduler",
        note="diffusers 네이티브 지원. LCM 증류 모델/LoRA와 함께 써야 제 성능.",
    ),
    "tcd": SamplerSpec(
        scheduler="TCDScheduler",
        note="diffusers 네이티브 지원. TCD 증류 모델/LoRA와 함께 써야 제 성능.",
    ),
}


def resolve_sampler(name: str) -> SamplerSpec:
    """샘플러 이름 → SamplerSpec. 모르면 ValueError."""
    try:
        return SAMPLER_MAP[name]
    except KeyError:
        raise ValueError(
            f"알 수 없는 샘플러입니다: {name!r}. "
            f"지원 목록: {sorted(SAMPLER_MAP)}"
        ) from None


def build_scheduler(name: str, base_config=None):
    """diffusers 스케줄러 인스턴스를 생성한다.

    base_config가 주어지면(보통 파이프라인의 기존 스케줄러 config)
    from_config로 베타 스케줄 등을 계승한다.
    diffusers는 여기서 lazy import하므로, 이 모듈 자체는
    torch/diffusers 없이 import 가능하다.
    """
    import diffusers  # lazy import: 테스트 환경엔 없어도 됨

    spec = resolve_sampler(name)
    cls = getattr(diffusers, spec.scheduler)
    if base_config is not None:
        return cls.from_config(base_config, **spec.kwargs)
    return cls(**spec.kwargs)


def apply_scheduler(pipe, name: str):
    """파이프라인의 스케줄러를 교체하고 pipe를 반환한다."""
    pipe.scheduler = build_scheduler(name, base_config=pipe.scheduler.config)
    return pipe
