"""생성 엔진 (diffusers 기반).

- AutoPipeline으로 체크포인트를 로드하므로 SD1.5 / SDXL / Anima 등
  diffusers 호환 체크포인트를 제네릭하게 다룬다.
- img2img / inpaint는 from_pipe로 가중치를 공유한다.
- torch / diffusers는 각 메서드에서 lazy import하므로,
  이 모듈 자체는 GPU·torch 없이 import 가능하다.

주의: strength(디노이즈 강도)는 settings.py의 denoise_strength 항목을 쓴다.
(엔진을 직접 호출할 때는 "strength" 키도 받는다. API 경로에서는
validate()가 모르는 키를 422로 막으므로 denoise_strength를 쓸 것.)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from atelier.engine.samplers import apply_scheduler
from atelier.settings import defaults, validate

# settings.py에 strength 항목이 생기기 전까지 쓰는 기본값. 실측 후 조정.
DEFAULT_IMG2IMG_STRENGTH = 0.75
DEFAULT_INPAINT_STRENGTH = 0.75
DEFAULT_CONTROLNET_STRENGTH = 0.8


def _lora_already_loaded(pipe, adapter: str) -> bool:
    """UNet을 공유하는 파생 파이프(from_pipe)에는 이미 로드돼 있을 수 있다.
    get_list_adapters가 있으면 실제 로드 여부를 확인한다."""
    get_list = getattr(pipe, "get_list_adapters", None)
    if not callable(get_list):
        return False
    try:
        return adapter in get_list()
    except Exception:  # noqa: BLE001
        return False


def _default_device() -> str:
    """cuda → mps(Apple Silicon Metal) 순서로 자동 판별.

    GPU가 없으면 에러를 낸다. CPU 폴백은 허용하지 않는다
    (SDXL을 CPU로 돌리면 한 장에 수십 분이 걸려 실용성이 없음).
    device를 명시적으로 넘기면(예: 테스트용 "cpu") 그 값을 그대로 쓴다.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    raise RuntimeError(
        "GPU를 찾을 수 없습니다. Atelier AI는 NVIDIA GPU(CUDA) 또는 "
        "Apple Silicon(Metal/MPS)이 필요합니다."
    )


def _default_dtype(device: str) -> str:
    return "float16" if device in ("cuda", "mps") else "float32"


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

        device = self.device or _default_device()
        dtype_name = self.torch_dtype or _default_dtype(device)
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

    def _resolve_lora(self, name: str) -> str | None:
        """LoRA 이름(모델 목록의 상대경로)을 절대 경로로 바꾼다.
        models dir 밖이거나 없는 파일이면 None (조용히 건너뜀)."""
        from atelier.api.routes import models as models_module
        try:
            return models_module._resolve(name)
        except Exception:  # noqa: BLE001 - 경로 조작/없는 파일은 건너뜀
            return None

    def _apply_loras(self, pipe, loras: dict | None) -> None:
        """요청의 LoRA 선택을 파이프라인에 적용한다.

        loras: {이름: {"enabled": bool, "strength": float}}
        이름은 모델 목록의 name(상대경로). 파이프라인별 로드 캐시를 두어
        같은 파일은 한 번만 로드한다. 아무것도 켜져 있지 않으면 비활성화.
        """
        cache = getattr(pipe, "_atelier_lora_cache", None)
        if cache is None:
            cache = {}
            pipe._atelier_lora_cache = cache  # type: ignore[attr-defined]

        active: list[tuple[str, float]] = []
        if loras:
            for name, info in loras.items():
                if not isinstance(info, dict) or not info.get("enabled"):
                    continue
                try:
                    strength = float(info.get("strength", 0.8))
                except (TypeError, ValueError):
                    strength = 0.8
                path = self._resolve_lora(name)
                if path is None:
                    continue
                adapter = cache.get(path)
                if adapter is None:
                    adapter = f"atelier_lora_{len(cache)}"
                    # from_pipe 파생 파이프는 UNet을 공유하므로
                    # 이미 로드된 어댑터면 다시 로드하지 않는다.
                    if not _lora_already_loaded(pipe, adapter):
                        try:
                            pipe.load_lora_weights(path, adapter_name=adapter)
                        except Exception as e:  # noqa: BLE001 - 깨진 LoRA는 건너뜀
                            print(f"[atelier] LoRA 로드 실패 ({name}): {e}")
                            continue
                    cache[path] = adapter
                active.append((adapter, strength))

        if not active:
            disable = getattr(pipe, "disable_lora", None)
            if callable(disable):
                try:
                    disable()
                except Exception:  # noqa: BLE001
                    pass
            return

        names = [a for a, _ in active]
        weights = [w for _, w in active]
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except TypeError:
            # 구버전 diffusers: 가중치 미지원
            pipe.set_adapters(names)

    def txt2img(self, params: dict, loras: dict | None = None) -> list[str]:
        """텍스트 → 이미지. 저장된 파일 경로 리스트 반환."""
        pipe, kwargs, seed = self._prepare(params, "sampler")
        self._apply_loras(pipe, loras)
        images = pipe(**kwargs).images
        return self._save(images, seed, "txt2img")

    def img2img(self, params: dict, image: str,
                loras: dict | None = None) -> list[str]:
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
        self._apply_loras(i2i, loras)
        images = i2i(image=Image.open(image).convert("RGB"),
                      strength=strength, **kwargs).images
        return self._save(images, seed, "img2img")

    def inpaint(self, params: dict, image: str, mask: str,
                loras: dict | None = None) -> list[str]:
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
        self._apply_loras(inp, loras)
        images = inp(image=Image.open(image).convert("RGB"),
                      mask_image=Image.open(mask).convert("RGB"),
                      strength=strength, **kwargs).images
        return self._save(images, seed, "inpaint")

    def pose(self, params: dict, control_image: str,
             loras: dict | None = None) -> list[str]:
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

        device = self.device or _default_device()
        dtype_name = self.torch_dtype or _default_dtype(device)
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
        self._apply_loras(pipe, loras)
        kwargs, seed = self._common_kwargs(p)
        images = pipe(image=Image.open(control_image).convert("RGB"),
                      controlnet_conditioning_scale=strength,
                      **kwargs).images
        return self._save(images, seed, "pose")

    # -- 저장 ----------------------------------------------------------

    def _save(self, images, seed: int, tag: str) -> list[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        uniq = uuid.uuid4().hex[:6]  # 같은 초·같은 seed여도 충돌 방지
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(
                self.output_dir, f"{stamp}-{tag}-s{seed}-{i}-{uniq}.png")
            img.save(path)
            paths.append(path)
        return paths
