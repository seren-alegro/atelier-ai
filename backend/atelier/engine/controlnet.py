"""ControlNet 로딩 (포즈 지정용).

- SDXL 기반이므로 SDXL용 ControlNet을 쓴다
  (예: xinsir/controlnet-openpose-sdxl-1.0).
- 로컬 파일(.safetensors, models/controlnet/ 하위) 또는 HF repo id를 받는다.
- torch / diffusers는 함수 안에서 lazy import하므로,
  이 모듈 자체는 GPU·torch 없이 import 가능하다.

주의: GPU 실측 전. 실제 Anima 체크포인트와의 궁합,
from_single_file/from_pretrained 분기, dtype 등은 실측 후 조정 필요.
"""

from __future__ import annotations

import os

DEFAULT_CONTROLNET = "xinsir/controlnet-openpose-sdxl-1.0"


def controlnet_models_dir() -> str:
    """ControlNet 모델 폴더. 없으면 만들어 둔다."""
    base = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    d = os.path.join(base, "controlnet")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_controlnet_ref(model_ref: str | None) -> str:
    """ControlNet 지정값을 diffusers가 받을 수 있는 ref로 바꾼다.

    - 비어 있으면: HF 기본값 (openpose SDXL)
    - 존재하는 파일 경로면: 그대로
    - controlnet 폴더 안의 파일명이면: 절대 경로로
    - models 폴더 기준 상대경로("controlnet/foo.safetensors")면: 절대 경로로
    - 그 외: HF repo id로 간주
    """
    ref = (model_ref or "").strip()
    if not ref:
        return DEFAULT_CONTROLNET
    if os.path.isfile(ref):
        return ref
    cn_dir = controlnet_models_dir()
    cand = os.path.join(cn_dir, ref)
    if os.path.isfile(cand):
        return cand
    # 프론트 드롭다운은 models dir 기준 상대경로를 보낸다
    base = os.path.dirname(cn_dir)
    cand2 = os.path.join(base, ref)
    if os.path.isfile(cand2):
        return cand2
    return ref


def load_controlnet(model_ref: str | None, torch_dtype=None):
    """ControlNetModel을 로드한다.

    - 로컬 .safetensors 파일이면 from_single_file
    - 그 외(폴더·HF repo id)는 from_pretrained
    """
    try:
        from diffusers import ControlNetModel
    except ImportError:
        raise RuntimeError(
            "ControlNet을 쓰려면 diffusers가 필요해요. "
            "백엔드 가상환경에 diffusers를 설치해주세요."
        )
    ref = resolve_controlnet_ref(model_ref)
    if ref.lower().endswith(".safetensors") and os.path.isfile(ref):
        # 단일 파일 가중치. 실측 필요: from_single_file의 config 추론이
        # ControlNet 변형(openpose/depth 등)을 제대로 구분하는지 확인 필요.
        return ControlNetModel.from_single_file(ref, torch_dtype=torch_dtype)
    return ControlNetModel.from_pretrained(ref, torch_dtype=torch_dtype)
