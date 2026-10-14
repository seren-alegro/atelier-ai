"""생성 엔진 (diffusers 기반).

- AutoPipeline으로 체크포인트를 로드하므로 SD1.5 / SDXL / Anima 등
  diffusers 호환 체크포인트를 제네릭하게 다룬다.
- img2img / inpaint는 from_pipe로 가중치를 공유한다.
- torch / diffusers는 각 메서드에서 lazy import하므로,
  이 모듈 자체는 GPU·torch 없이 import 가능하다.

주의: strength(디노이즈 강도)는 settings.py의 denoise_strength 항목을 쓴다.
("strength" 키도 하위 호환으로 받는다.)
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from atelier.engine.samplers import apply_scheduler
from atelier.settings import defaults, validate

# settings.py에 strength 항목이 생기기 전까지 쓰는 기본값. 실측 후 조정.
DEFAULT_IMG2IMG_STRENGTH = 0.75
DEFAULT_INPAINT_STRENGTH = 0.75
DEFAULT_CONTROLNET_STRENGTH = 0.8


class Pipeline:
    """txt2img / img2img / inpaint 파이프라인."""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "outputs",
        device: str | None = None,
        torch_dtype: str | None = None,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device          # None이면 자동 판별
        self.torch_dtype = torch_dtype  # None이면 자동 판별
        self._pipe = None
        self._controlnet_pipes: dict = {}  # controlnet ref -> SDXL ControlNet 파이프라인

    # -- 로드 ----------------------------------------------------------

    def load(self) -> "Pipeline":
        """체크포인트를 로드한다. 여러 번 호출해도 한 번만 로드."""
        if self._pipe is not None:
            return self
        import torch
        from diffusers import AutoPipelineForText2Image

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = self.torch_dtype or ("float16" if device == "cuda" else "float32")
        dtype = getattr(torch, dtype_name)

        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=False,
        )
        pipe.to(device)
        self._pipe = pipe
        self.device = device
        return self

    def _require_pipe(self):
        if self._pipe is None:
            raise RuntimeError("파이프라인이 로드되지 않았어요. load()를 먼저 호출하세요.")
        return self._pipe

    # -- 파라미터 ------------------------------------------------------

    def _common_kwargs(self, params: dict):
        """settings 키 → diffusers 호출 인자 매핑."""
        import torch

        errors = validate(params)
        if errors:
            raise ValueError(f"파라미터 오류: {errors}")

        p = defaults()
        p.update(params)

        seed = p["seed"]
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return {
            "prompt": p["prompt"],
            "negative_prompt": p["negative_prompt"],
            "height": p["height"],
            "width": p["width"],
            "num_inference_steps": p["steps"],
            "guidance_scale": p["cfg_scale"],
            "generator": generator,
            "num_images_per_prompt": p["batch_size"],
        }, seed

    def _prepare(self, params: dict, sampler_key: str = "sampler"):
        """검증 + 스케줄러 적용 + 공통 인자 생성."""
        pipe = self._require_pipe()
        sampler_name = params.get(sampler_key, defaults()[sampler_key])
        apply_scheduler(pipe, sampler_name)
        kwargs, seed = self._common_kwargs(params)
        return pipe, kwargs, seed

    # -- 생성 ----------------------------------------------------------

    def txt2img(self, params: dict) -> list[str]:
        """텍스트 → 이미지. 저장된 파일 경로 리스트 반환."""
        pipe, kwargs, seed = self._prepare(params, "sampler")
        images = pipe(**kwargs).images
        return self._save(images, seed, "txt2img")

    def img2img(self, params: dict, image: str) -> list[str]:
        """이미지 → 이미지. strength는 denoise_strength 설정값 사용."""
        from diffusers import AutoPipelineForImage2Image
        from PIL import Image

        p = dict(params)
        strength = p.pop(
            "denoise_strength", p.pop("strength", DEFAULT_IMG2IMG_STRENGTH))
        pipe, kwargs, seed = self._prepare(p, "sampler")
        i2i = AutoPipelineForImage2Image.from_pipe(
            pipe, trust_remote_code=False,
        ).to(self.device)
        images = i2i(image=Image.open(image).convert("RGB"),
                      strength=strength, **kwargs).images
        return self._save(images, seed, "img2img")

    def inpaint(self, params: dict, image: str, mask: str) -> list[str]:
        """인페인트. image 원본, mask 흰색=다시 그릴 영역.
        strength는 denoise_strength 설정값 사용."""
        from diffusers import AutoPipelineForInpainting
        from PIL import Image

        p = dict(params)
        strength = p.pop(
            "denoise_strength", p.pop("strength", DEFAULT_INPAINT_STRENGTH))
        pipe, kwargs, seed = self._prepare(p, "sampler")
        inp = AutoPipelineForInpainting.from_pipe(
            pipe, trust_remote_code=False,
        ).to(self.device)
        images = inp(image=Image.open(image).convert("RGB"),
                      mask_image=Image.open(mask).convert("RGB"),
                      strength=strength, **kwargs).images
        return self._save(images, seed, "inpaint")

    def pose(self, params: dict, control_image: str) -> list[str]:
        """포즈 지정 생성 (ControlNet).
        control_image: 포즈맵(OpenPose 스켈레톤 PNG) 파일 경로.
        strength는 controlnet_strength 설정값 사용.

        주의: SDXL 전용 파이프라인을 쓴다 (Anima는 SDXL 기반).
        SD1.5 체크포인트에서는 StableDiffusionControlNetPipeline으로
        교체해야 함 — 실측 필요.
        """
        import torch
        from diffusers import StableDiffusionXLControlNetPipeline
        from PIL import Image

        from atelier.engine.controlnet import (
            load_controlnet,
            resolve_controlnet_ref,
        )

        p = dict(params)
        strength = p.pop("controlnet_strength", DEFAULT_CONTROLNET_STRENGTH)
        model_ref = p.pop("controlnet_model", "")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = self.torch_dtype or (
            "float16" if device == "cuda" else "float32")
        dtype = getattr(torch, dtype_name)
        self.device = device

        ref = resolve_controlnet_ref(model_ref)
        if ref not in self._controlnet_pipes:
            controlnet = load_controlnet(ref, torch_dtype=dtype)
            cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.model_path, controlnet=controlnet,
                torch_dtype=dtype, trust_remote_code=False,
            ).to(device)
            self._controlnet_pipes[ref] = cn_pipe
        pipe = self._controlnet_pipes[ref]

        sampler_name = p.get("sampler", defaults()["sampler"])
        apply_scheduler(pipe, sampler_name)
        kwargs, seed = self._common_kwargs(p)
        images = pipe(image=Image.open(control_image).convert("RGB"),
                      controlnet_conditioning_scale=strength,
                      **kwargs).images
        return self._save(images, seed, "pose")

    # -- 저장 ----------------------------------------------------------

    def _save(self, images, seed: int, tag: str) -> list[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(
                self.output_dir, f"{stamp}-{tag}-s{seed}-{i}.png")
            img.save(path)
            paths.append(path)
        # 같은 초에 여러 배치가 겹치면 파일명 충돌 방지
        time.sleep(0.01)
        return paths
