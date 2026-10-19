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

import importlib
import re
from dataclasses import dataclass, field


#: diffusers에 없는 커스텀 스케줄러 레지스트리
#: {스케줄러명: (모듈 경로, 클래스명)}
_CUSTOM_SCHEDULERS = {
    "ResMultistepScheduler": (
        "atelier.engine.res_multistep", "ResMultistepScheduler"),
}


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
    # res_multistep은 나중에 ComfyUI/k_diffusion 로직 직접 내장 예정이었으나,
    # exponential multistep 수식을 클린룸으로 재구현하여 내장함
    # (atelier/engine/res_multistep.py — 원본 코드 복사 없음)
    "flow_euler": SamplerSpec(
        scheduler="FlowMatchEulerDiscreteScheduler",
        kwargs={"shift": 3.0},
        note="정확 대응. Anima-Base-v1.0-Diffusers 공식 기본 스케줄 "
             "(flow_diffusers_linear_shift, shift 3.0). rectified-flow 2b용.",
    ),
    "res_multistep": SamplerSpec(
        scheduler="ResMultistepScheduler",
        note="직접 내장 (클린룸 구현). RES res_2m 계열의 exponential "
             "integrator + phi 가중 2차 multistep. GPU 실측 필요.",
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
    """스케줄러 인스턴스를 생성한다.

    diffusers 네이티브 스케줄러는 클래스명으로 가져오고,
    _CUSTOM_SCHEDULERS에 등록된 것은 직접 구현 모듈에서 가져온다.
    base_config가 주어지면(보통 파이프라인의 기존 스케줄러 config)
    diffusers 스케줄러는 from_config로 베타 스케줄 등을 계승한다
    (커스텀 스케줄러는 base_config 무시).
    torch/diffusers는 여기서 lazy import하므로, 이 모듈 자체는
    torch/diffusers 없이 import 가능하다.
    """
    spec = resolve_sampler(name)
    if spec.scheduler in _CUSTOM_SCHEDULERS:
        mod_name, cls_name = _CUSTOM_SCHEDULERS[spec.scheduler]
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(**spec.kwargs)

    import diffusers  # lazy import: 테스트 환경엔 없어도 됨

    cls = getattr(diffusers, spec.scheduler)
    if base_config is not None:
        return cls.from_config(base_config, **spec.kwargs)
    return cls(**spec.kwargs)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 노이즈 스케줄 (ComfyUI식 스케줄러 선택)
# ---------------------------------------------------------------------------
#
# ComfyUI의 스케줄러는 sigma 스케줄을 바꾸는 것이다. 아래 함수들은
# ComfyUI/RES4LYF의 정의를 diffusers 스케줄러의 네이티브 sigma 스케줄에
# 맞게 옮긴 것이다 (원본 코드 복사 없음, 수식만 참고).
# ComfyUI의 네이티브 sigmas는 오름차순이므로 diffusers(내림차순) 것을
# 뒤집어(asc) ComfyUI와 같은 관례로 계산한다.
#
# - normal: timestep 공간에서 균등 (네이티브 곡선 위 보간)
# - karras: k-diffusion식 rho=7 곡선
# - simple: 네이티브 스케줄을 균등 간격으로 솎아냄
# - sgm_uniform: timestep 공간에서 균등, 끝점 제외
# - exponential: 지수 곡선
# - ddim_uniform: 네이티브 스케줄을 균등 간격으로 솎아냄 (뒤집기 변형)
# - beta: 베타분포 분위수로 네이티브 인덱스 선택 (수치해석으로 ppf 계산)
# - linear_quadratic: 앞쪽은 선형, 뒤쪽은 2차 (mochi 방식)
# - bong_tangent: 2단계 탄젠트 곡선 (RES4LYF)

NOISE_SCHEDULES = ("auto", "normal", "karras", "simple", "sgm_uniform",
                   "exponential", "ddim_uniform", "beta",
                   "linear_quadratic", "bong_tangent")


def _betacf(a: float, b: float, x: float) -> float:
    """불완전 베타함수의 연분수 전개 (Numerical Recipes식)."""
    maxit, eps, fpmin = 200, 3e-7, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        dele = d * c
        h *= dele
        if abs(dele - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """정규화 불완전 베타함수 I_x(a, b)."""
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """베타분포 분위수함수 (이분법으로 역함산)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _bong_tangent_values(num_steps: int) -> list:
    """RES4LYF식 2단계 탄젠트 스케줄 (정규화 1.0→0.0, 길이 num_steps+1)."""
    import math

    def stage(steps, slope, pivot, start, end):
        smax = ((2 / math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
        smin = ((2 / math.pi) * math.atan(-slope * ((steps - 1) - pivot)) + 1) / 2
        srange = smax - smin
        sscale = start - end
        return [((((2 / math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2)
                 - smin) * (1 / srange) * sscale + end
                for x in range(steps)]

    steps = num_steps + 2
    pivot_1, pivot_2 = 0.6, 0.6
    slope_1, slope_2 = 0.2, 0.2
    midpoint = int((steps * pivot_1 + steps * pivot_2) / 2)
    p1 = int(steps * pivot_1)
    s1 = slope_1 / (steps / 40)
    s2 = slope_2 / (steps / 40)
    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len
    t1 = stage(stage_1_len, s1, p1, 1.0, 0.5)[:-1]
    t2 = stage(stage_2_len, s2, int(steps * pivot_2) - stage_1_len, 0.5, 0.0)
    return t1 + t2


def make_noise_sigmas(schedule: str, native_sigmas, num_steps: int):
    """네이티브 sigma 스케줄(내림차순, 길이 num_steps+1)에서
    커스텀 sigma 스케줄(길이 num_steps+1, 마지막은 0)을 만든다."""
    import math

    import torch

    native = [float(s) for s in native_sigmas]  # 내림차순 (diffusers)
    asc = native[::-1]  # 오름차순 (ComfyUI 관례)
    n = len(asc)
    sigma_max = native[0]
    sigma_min = native[-1] if native[-1] > 0 else 1e-3

    def _at(idx: float) -> float:
        lo = max(0, min(n - 2, int(math.floor(idx))))
        frac = max(0.0, min(1.0, idx - lo))
        return asc[lo] * (1 - frac) + asc[lo + 1] * frac

    if schedule == "normal":
        steps = num_steps
        append_zero = True
        if math.isclose(asc[0], 0.0, abs_tol=1e-5):
            steps += 1
            append_zero = False
        sigs = [_at(i) for i in torch.linspace(n - 1, 0, steps).tolist()]
        if append_zero:
            sigs.append(0.0)
        return sigs

    if schedule == "karras":
        rho = 7.0
        ramp = torch.linspace(0, 1, num_steps).tolist()
        min_ir = sigma_min ** (1 / rho)
        max_ir = sigma_max ** (1 / rho)
        sigs = [(max_ir + r * (min_ir - max_ir)) ** rho for r in ramp]
        sigs.append(0.0)
        return sigs

    if schedule == "simple":
        ss = n / num_steps
        sigs = [asc[-(1 + int(x * ss))] for x in range(num_steps)]
        sigs.append(0.0)
        return sigs

    if schedule == "sgm_uniform":
        idx = torch.linspace(n - 1, 0, num_steps + 1)[:-1].tolist()
        sigs = [_at(i) for i in idx]
        sigs.append(0.0)
        return sigs

    if schedule == "exponential":
        ramp = torch.linspace(0, 1, num_steps).tolist()
        sigs = [sigma_max * (sigma_min / sigma_max) ** r for r in ramp]
        sigs.append(0.0)
        return sigs

    if schedule == "ddim_uniform":
        sigs = []
        x = 1
        steps = num_steps
        if math.isclose(asc[x], 0.0, abs_tol=1e-5):
            steps += 1
        else:
            sigs = [0.0]
        ss = max(n // steps, 1)
        while x < n:
            sigs.append(asc[x])
            x += ss
        return sigs[::-1]

    if schedule == "beta":
        alpha = beta_ = 0.6
        total = n - 1
        sigs = []
        last_t = -1
        for i in range(num_steps):
            t = 1.0 - i / num_steps
            ti = int(round(_beta_ppf(t, alpha, beta_) * total))
            if ti != last_t:
                sigs.append(asc[ti])
            last_t = ti
        sigs.append(0.0)
        return sigs

    if schedule == "linear_quadratic":
        threshold_noise = 0.025
        if num_steps == 1:
            sched = [1.0, 0.0]
        else:
            linear_steps = num_steps // 2
            linear = [i * threshold_noise / linear_steps
                      for i in range(linear_steps)]
            diff = linear_steps - threshold_noise * num_steps
            quad_steps = num_steps - linear_steps
            qcoef = diff / (linear_steps * quad_steps ** 2)
            lcoef = (threshold_noise / linear_steps
                     - 2 * diff / (quad_steps ** 2))
            const = qcoef * (linear_steps ** 2)
            quad = [qcoef * (i ** 2) + lcoef * i + const
                    for i in range(linear_steps, num_steps)]
            sched = linear + quad + [1.0]
            sched = [1.0 - x for x in sched]
        return [s * sigma_max for s in sched]

    if schedule == "bong_tangent":
        return [s * sigma_max for s in _bong_tangent_values(num_steps)]

    raise ValueError(f"알 수 없는 노이즈 스케줄: {schedule}")
class _NoiseScheduleWrapper:
    """set_timesteps 호출을 가로채 커스텀 sigma 스케줄을 적용하는 래퍼.

    파이프라인이 나중에 scheduler.set_timesteps(steps)를 호출하면
    네이티브 스케줄을 먼저 만든 뒤 커스텀 sigmas로 다시 설정한다
    (diffusers의 set_timesteps(sigmas=...)가 timesteps까지 계산).
    """

    def __init__(self, scheduler, schedule: str):
        self._sched = scheduler
        self._schedule = schedule

    def set_timesteps(self, num_inference_steps: int, **kwargs):
        if self._schedule in (None, "auto"):
            return self._sched.set_timesteps(num_inference_steps, **kwargs)
        # 1. 네이티브 스케줄 확보
        self._sched.set_timesteps(num_inference_steps, **kwargs)
        native = self._sched.sigmas
        sigmas = make_noise_sigmas(self._schedule, native,
                                   num_inference_steps)
        # 2. 커스텀 sigmas로 다시 설정
        try:
            return self._sched.set_timesteps(sigmas=sigmas)
        except TypeError:
            # sigmas 파라미터 미지원 스케줄러는 직접 교체
            import torch

            self._sched.sigmas = torch.tensor(sigmas,
                                              dtype=torch.float32)
            n = self._sched.num_train_timesteps \
                if hasattr(self._sched, "num_train_timesteps") else 1000
            self._sched.timesteps = (
                self._sched.sigmas[:-1] / max(sigmas[0], 1e-12) * n
            ).long()
            return self._sched

    def __getattr__(self, name):
        return getattr(self._sched, name)


def apply_scheduler(pipe, name: str, noise_schedule: str = "auto"):
    """파이프라인의 스케줄러를 교체하고 pipe를 반환한다.

    noise_schedule이 auto가 아니면 커스텀 sigma 스케줄을 적용한다.
    """
    sched = build_scheduler(name, base_config=pipe.scheduler.config)
    if noise_schedule not in (None, "auto"):
        sched = _NoiseScheduleWrapper(sched, noise_schedule)
    pipe.scheduler = sched
    return pipe
