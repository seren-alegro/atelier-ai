"""샘플러 이름 → diffusers 스케줄러 매핑.

settings.py의 `sampler` 설정값(일반 25종 + 실험실 5종)을
diffusers 스케줄러로 변환한다.

원칙:
- diffusers에 직접 대응되는 것은 정확 매핑 (approximate=False)
- diffusers에 없는 것(Kohya/ComfyUI 계열 SDE 등)은 가장 가까운 스케줄러로
  매핑하고 approximate=True + note에 "근사 매핑, 실측 필요"를 명시한다.
- 매핑 근거가 불확실한 것은 지어내지 않고 note에 불확실함을 적는다.

SAMPLER_ALIASES는 A1111 표시 이름("Euler a" 등)과 ComfyUI 이름("dpm_2" 등)을
내부 키로 연결한다. normalize_sampler_name()으로 대소문자·공백을 무시하고
내부 키를 찾는다 (PNG 메타데이터의 샘플러 이름 정규화용).

이 모듈은 diffusers/torch 없이 import 가능해야 한다
(스케줄러 클래스는 문자열로 보관, 실제 로딩은 build_scheduler에서 lazy import).
"""

from __future__ import annotations

import re
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
    # --- 실험실 샘플러 (settings.py `sampler`의 실험실 5종) ---
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
    # --- A1111 전체 샘플러 (추가) ---
    "euler": SamplerSpec(
        scheduler="EulerDiscreteScheduler",
        note="정확 대응. A1111 'Euler'.",
    ),
    "lms": SamplerSpec(
        scheduler="LMSDiscreteScheduler",
        note="정확 대응. A1111 'LMS'.",
    ),
    "heun": SamplerSpec(
        scheduler="HeunDiscreteScheduler",
        note="정확 대응. A1111 'Heun'.",
    ),
    "dpm2": SamplerSpec(
        scheduler="KDPM2DiscreteScheduler",
        note="정확 대응. A1111 'DPM2' (Karras DPM 2차).",
    ),
    "dpm2_ancestral": SamplerSpec(
        scheduler="KDPM2AncestralDiscreteScheduler",
        note="정확 대응. A1111 'DPM2 a'.",
    ),
    "dpmpp_2s_ancestral": SamplerSpec(
        scheduler="DPMSolverSinglestepScheduler",
        note="정확 대응. A1111 'DPM++ 2S a' (singlestep ancestral).",
    ),
    "dpmpp_2s_ancestral_karras": SamplerSpec(
        scheduler="DPMSolverSinglestepScheduler",
        kwargs={"use_karras_sigmas": True},
        note="정확 대응. A1111 'DPM++ 2S a Karras'.",
    ),
    "dpmpp_2m": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "dpmsolver++", "solver_order": 2},
        note="정확 대응. A1111 'DPM++ 2M'.",
    ),
    "dpmpp_sde": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ SDE'. diffusers의 sde-dpmsolver++는 "
             "동일 알고리즘 계열이나 Kohya/ComfyUI 구현체와 세부 구현이 다를 수 있음.",
    ),
    "dpmpp_sde_karras": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2,
                "use_karras_sigmas": True},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ SDE Karras'. diffusers의 "
             "sde-dpmsolver++는 동일 알고리즘 계열이나 구현체와 세부 구현이 "
             "다를 수 있음.",
    ),
    "dpmpp_2m_sde_karras": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2,
                "use_karras_sigmas": True},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ 2M SDE Karras'. diffusers의 "
             "sde-dpmsolver++는 동일 알고리즘 계열이나 구현체와 세부 구현이 "
             "다를 수 있음.",
    ),
    "dpmpp_2m_sde_heun": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ 2M SDE Heun'의 Heun 보정 단계는 "
             "diffusers에 직접 대응이 없어 sde-dpmsolver++로 근사.",
    ),
    "dpmpp_2m_sde_heun_karras": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 2,
                "use_karras_sigmas": True},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ 2M SDE Heun Karras'의 Heun 보정 "
             "단계는 diffusers에 직접 대응이 없어 sde-dpmsolver++로 근사.",
    ),
    "dpmpp_3m_sde": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ 3M SDE'. diffusers의 sde-dpmsolver++는 "
             "동일 알고리즘 계열이나 Kohya/ComfyUI 구현체와 세부 구현이 다를 수 있음.",
    ),
    "dpmpp_3m_sde_karras": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "sde-dpmsolver++", "solver_order": 3,
                "use_karras_sigmas": True},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM++ 3M SDE Karras'. diffusers의 "
             "sde-dpmsolver++는 동일 알고리즘 계열이나 구현체와 세부 구현이 "
             "다를 수 있음.",
    ),
    "dpm_fast": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM fast'(k-diffusion sample_dpm_fast)는 "
             "diffusers에 직접 대응이 없어 DPM-Solver++ 2M으로 근사.",
    ),
    "dpm_adaptive": SamplerSpec(
        scheduler="DPMSolverMultistepScheduler",
        kwargs={"algorithm_type": "dpmsolver++", "solver_order": 2},
        approximate=True,
        note="근사 매핑, 실측 필요. A1111 'DPM adaptive'의 적응형 스텝 조절은 "
             "diffusers에 대응이 없어 고정 스텝 DPM-Solver++ 2M으로 근사.",
    ),
    "plms": SamplerSpec(
        scheduler="PNDMScheduler",
        note="정확 대응. A1111 'PLMS' (PNDM pseudo-linear multistep).",
    ),
    "unipc": SamplerSpec(
        scheduler="UniPCMultistepScheduler",
        note="정확 대응. A1111 'UniPC'.",
    ),
    # --- Rectified-flow 2b용 (Comfyui-anima-sampler 지원 목적) ---
    "flow_euler": SamplerSpec(
        scheduler="FlowMatchEulerDiscreteScheduler",
        kwargs={"shift": 3.0},
        note="정확 대응. Anima-Base-v1.0-Diffusers 공식 기본 스케줄 "
             "(flow_diffusers_linear_shift, shift 3.0). rectified-flow 2b용.",
    ),
    "res_multistep": SamplerSpec(
        scheduler="FlowMatchEulerDiscreteScheduler",
        kwargs={"shift": 3.0},
        approximate=True,
        note="근사 매핑, 실측 필요. RES4LYF res_multistep(RF multistep)의 "
             "diffusers 직접 대응이 없어 같은 RF 계열의 FlowMatchEuler로 근사. "
             "차수 차이로 결과가 다를 수 있음.",
    ),
}


#: A1111 표시 이름 / ComfyUI 이름 → 내부 키
SAMPLER_ALIASES: dict[str, str] = {
    # --- A1111 표시 이름 ---
    "Euler a": "euler_ancestral",
    "Euler": "euler",
    "LMS": "lms",
    "Heun": "heun",
    "DPM2": "dpm2",
    "DPM2 a": "dpm2_ancestral",
    "DPM++ 2S a": "dpmpp_2s_ancestral",
    "DPM++ 2S a Karras": "dpmpp_2s_ancestral_karras",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ 2M Karras": "dpmpp_2m_karras",
    "DPM++ SDE": "dpmpp_sde",
    "DPM++ SDE Karras": "dpmpp_sde_karras",
    "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ 2M SDE Karras": "dpmpp_2m_sde_karras",
    "DPM++ 2M SDE Heun": "dpmpp_2m_sde_heun",
    "DPM++ 2M SDE Heun Karras": "dpmpp_2m_sde_heun_karras",
    "DPM++ 3M SDE": "dpmpp_3m_sde",
    "DPM++ 3M SDE Karras": "dpmpp_3m_sde_karras",
    "DPM fast": "dpm_fast",
    "DPM adaptive": "dpm_adaptive",
    "DDIM": "ddim",
    "PLMS": "plms",
    "UniPC": "unipc",
    "LCM": "lcm",
    "Restart": "restart",
    # --- ComfyUI 이름 ---
    "euler": "euler",
    "euler_ancestral": "euler_ancestral",
    "euler_cfg_pp": "euler",
    "dpm_2": "dpm2",
    "dpm_2_ancestral": "dpm2_ancestral",
    "dpmpp_2s_ancestral": "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfgpp": "dpmpp_2s_ancestral",
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp_2m_cfgpp": "dpmpp_2m",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu": "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde": "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu": "dpmpp_3m_sde",
    "ddim": "ddim",
    "ddim_cfg_pp": "ddim",
    "uni_pc": "unipc",
    "uni_pc_bh2": "unipc",
    "lcm": "lcm",
    "tcd": "tcd",
    "sa_solver": "sa_solver",
}


def _normalize_alias_key(name: str) -> str:
    """별칭 비교용 정규화: 소문자 + 공백 제거."""
    return re.sub(r"\s+", "", name.strip().lower())


#: 정규화된 별칭/키 → 내부 키 (모듈 로드 시 1회 구축)
_NORMALIZED_SAMPLER_LOOKUP: dict[str, str] = {}
for _key in SAMPLER_MAP:
    _NORMALIZED_SAMPLER_LOOKUP.setdefault(
        _normalize_alias_key(_key), _key)
for _alias, _target in SAMPLER_ALIASES.items():
    _NORMALIZED_SAMPLER_LOOKUP.setdefault(
        _normalize_alias_key(_alias), _target)
del _key, _alias, _target


def normalize_sampler_name(name) -> str | None:
    """표시 이름/별칭/내부 키 → 내부 키. 대소문자·공백 무시.

    찾지 못하면 None (예외를 던지지 않는다).
    """
    if not isinstance(name, str) or not name.strip():
        return None
    return _NORMALIZED_SAMPLER_LOOKUP.get(_normalize_alias_key(name))


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
