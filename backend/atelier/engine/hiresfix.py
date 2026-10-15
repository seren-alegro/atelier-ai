"""Hires Fix (고해상도 보정).

흐름: base 해상도로 txt2img → Lanczos로 scale배 업스케일
      → img2img(denoise=hires_fix_denoise)로 살짝 다시 그리기.

RealESRGAN이 아니라 Lanczos를 쓰는 이유: Hires Fix의 1단계는
단순 확대면 충분하고, 디테일은 2단계 img2img에서 만들기 때문이다.

torch/diffusers는 직접 import하지 않는다 (파이프라인이 lazy import).
"""

from __future__ import annotations

import os
import tempfile
import uuid


def apply_hires_fix(pipe, params: dict, loras: dict | None,
                    image_paths: list[str]) -> list[str]:
    """base 생성 결과들에 Hires Fix를 적용한다.

    pipe: Pipeline (txt2img 결과를 낸 그 파이프라인, img2img 재사용).
    params: 생성 params (hires_fix_scale, hires_fix_denoise 참조).
    image_paths: 1단계 txt2img 결과 파일 경로들.
    반환: 2단계 img2img 결과 파일 경로들.
    """
    from PIL import Image

    scale = params.get("hires_fix_scale", 2.0)
    denoise = params.get("hires_fix_denoise", 0.35)

    out_paths: list[str] = []
    for src in image_paths:
        img = Image.open(src).convert("RGB")
        w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        upscaled = img.resize((w, h), Image.LANCZOS)

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"atelier-hiresfix-{uuid.uuid4().hex[:6]}-",
                suffix=".png",
            )
            os.close(fd)
            upscaled.save(tmp_path)

            p = dict(params)
            p["width"] = w
            p["height"] = h
            # img2img는 denoise_strength를 strength로 쓴다 (pipeline.py 패턴)
            p["denoise_strength"] = denoise
            # hires_fix_* 키는 _common_kwargs가 무시하므로 둬도 되지만,
            # 2단계 기록이 깔끔하게 남도록 유지한다.
            out_paths.extend(pipe.img2img(p, tmp_path, loras))
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    return out_paths
