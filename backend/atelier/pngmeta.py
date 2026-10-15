"""PNG 메타데이터 왕복.

- 저장: 생성 설정을 PNG tEXt 청크(`atelier:params` 키, JSON)로 기록.
  Pillow의 PngInfo에 넣고 `img.save(path, pnginfo=...)`로 저장한다.
- 읽기: `atelier:params` JSON을 우선 읽고, 없으면 A1111식 `parameters`
  텍스트 청크를 파싱한다. PNG가 아니거나 파싱에 실패하면 None.

모든 읽기 함수는 예외를 밖으로 던지지 않는다.
"""

from __future__ import annotations

import io
import json
import re

from PIL import Image
from PIL.PngImagePlugin import PngInfo

#: Atelier AI가 생성 설정을 기록하는 tEXt 청크 키
PARAM_KEY = "atelier:params"


def build_pnginfo(params: dict) -> PngInfo:
    """생성 설정 dict → Pillow PngInfo. img.save(pnginfo=...)에 그대로 쓴다.

    포함 키: prompt, negative_prompt, steps, cfg_scale, sampler, seed,
    width, height, model(있으면), app="atelier-ai".
    """
    payload = dict(params)
    payload["app"] = "atelier-ai"
    info = PngInfo()
    info.add_text(PARAM_KEY, json.dumps(payload, ensure_ascii=False))
    return info


def build_meta_from_kwargs(
    kwargs: dict, seed: int, sampler_name: str
) -> dict:
    """_common_kwargs() 결과 + 실제 시드 + 샘플러 이름 → PNG 기록용 dict.

    kwargs: prompt, negative_prompt, num_inference_steps, guidance_scale,
    width, height 키를 쓴다. 모델 이름은 현재 활성 모델에서 가져온다.
    """
    meta = {
        "prompt": kwargs["prompt"],
        "negative_prompt": kwargs["negative_prompt"],
        "steps": kwargs["num_inference_steps"],
        "cfg_scale": kwargs["guidance_scale"],
        "sampler": sampler_name,
        "seed": seed,
        "width": kwargs["width"],
        "height": kwargs["height"],
    }
    try:
        # 순환 import를 피하려고 함수 안에서 가져온다.
        from atelier.api.routes import models as models_module

        name = models_module.get_active_name()
    except Exception:  # noqa: BLE001 - 모델 이름은 부가 정보라 실패해도 무시
        name = None
    if name:
        meta["model"] = name
    return meta


def read_params_from_bytes(data: bytes) -> dict | None:
    """PNG 바이트에서 생성 설정을 읽는다. 실패하면 None (예외 전파 없음)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            meta = img.info or {}
            raw = meta.get(PARAM_KEY)
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
            text = meta.get("parameters")
            if isinstance(text, str):
                return _parse_a1111_params(text)
    except Exception:  # noqa: BLE001 - 깨진 파일도 조용히 None
        pass
    return None


def _to_int(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_a1111_params(text: str) -> dict | None:
    """A1111/SD-webui식 `parameters` 청크 파싱.

    형식 예:
        masterpiece, 1girl, solo
        Negative prompt: lowres, bad anatomy
        Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 123,
        Size: 512x768, Model: my_model
    """
    text = text.strip()
    if not text:
        return None

    prompt = ""
    negative = ""
    settings_str = ""

    head, sep, tail = text.partition("\nNegative prompt: ")
    if sep:
        prompt = head.strip()
        # negative는 "\nSteps: " 직전까지 (멀티라인 가능)
        neg_head, sep2, settings = tail.partition("\nSteps: ")
        if sep2:
            negative = neg_head.strip()
            settings_str = "Steps: " + settings
        else:
            negative = tail.strip()
    else:
        prompt_head, sep2, settings = head.partition("\nSteps: ")
        if sep2:
            prompt = prompt_head.strip()
            settings_str = "Steps: " + settings
        else:
            prompt = head.strip()

    result: dict = {"prompt": prompt, "negative_prompt": negative}

    if settings_str:
        # ", Key: " 패턴으로 쌍 분리 (값 안의 콤마는 보존)
        pairs = re.split(r", (?=[A-Za-z][A-Za-z ]*: )", settings_str)
        for pair in pairs:
            key, _, value = pair.partition(": ")
            key = key.strip()
            value = value.strip().strip('"')
            if key == "Steps":
                v = _to_int(value)
                if v is not None:
                    result["steps"] = v
            elif key == "Sampler":
                result["sampler"] = value
            elif key == "CFG scale":
                v = _to_float(value)
                if v is not None:
                    result["cfg_scale"] = v
            elif key == "Seed":
                v = _to_int(value)
                if v is not None:
                    result["seed"] = v
            elif key == "Size":
                m = re.match(r"(\d+)\s*x\s*(\d+)", value)
                if m:
                    result["width"] = int(m.group(1))
                    result["height"] = int(m.group(2))
            elif key == "Model":
                result["model"] = value

    if not result.get("prompt") and not result.get("negative_prompt"):
        return None
    return result
