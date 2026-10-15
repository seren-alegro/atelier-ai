"""아웃페인팅 (캔버스 확장).

원본을 더 큰 캔버스 중앙에 배치하고, 새로 생긴 영역을 마스킹해
pipeline.inpaint를 한 번 호출한다. 마스크 경계는 블러로 feathering.

torch/diffusers는 직접 import하지 않는다 (파이프라인이 lazy import).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from io import BytesIO

MIN_EXPAND = 64
MAX_EXPAND = 512
DEFAULT_DENOISE = 0.6
DEFAULT_BLUR = 16
FALLBACK_PROMPT = "high quality, detailed background"


def validate_expand(expand: dict) -> dict[str, int]:
    """expand {up, down, left, right} 검증. 불가하면 ValueError."""
    out: dict[str, int] = {}
    for key in ("up", "down", "left", "right"):
        v = expand.get(key, 0)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"확장 값 '{key}'은(는) {MIN_EXPAND}~{MAX_EXPAND}px "
                f"정수여야 해요.")
        if not (MIN_EXPAND <= v <= MAX_EXPAND):
            raise ValueError(
                f"확장 값 '{key}'은(는) {MIN_EXPAND}~{MAX_EXPAND}px "
                f"사이여야 해요.")
        out[key] = v
    return out


def outpaint_image(
    image_bytes: bytes,
    expand: dict,
    pipe,
    prompt: str = "",
    negative_prompt: str = "",
    denoise_strength: float = DEFAULT_DENOISE,
    mask_blur: int = DEFAULT_BLUR,
    loras: dict | None = None,
) -> bytes:
    """이미지 바깥을 확장해 다시 그린다. PNG 바이트 반환.

    pipe: 로드된 Pipeline (inpaint 사용). get_pipeline()으로 미리 확보.
    """
    from PIL import Image, ImageDraw, ImageFilter

    ex = validate_expand(expand)
    if not (0.0 <= denoise_strength <= 1.0):
        raise ValueError("디노이즈 강도는 0~1 사이여야 해요.")
    if mask_blur < 0 or mask_blur > 64:
        raise ValueError("마스크 블러는 0~64 사이여야 해요.")

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    left, right, up, down = ex["left"], ex["right"], ex["up"], ex["down"]
    cw, ch = w + left + right, h + up + down

    canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
    canvas.paste(img, (left, up))

    # 마스크: 흰색=다시 그릴 영역(새 영역), 검은색=원본 유지
    mask = Image.new("L", (cw, ch), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, up, left + w - 1, up + h - 1], fill=0)
    if mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))

    tmp_image = tmp_mask = None
    try:
        fd, tmp_image = tempfile.mkstemp(
            prefix=f"atelier-outpaint-{uuid.uuid4().hex[:6]}-", suffix=".png")
        os.close(fd)
        canvas.save(tmp_image)
        fd, tmp_mask = tempfile.mkstemp(
            prefix=f"atelier-outpaint-{uuid.uuid4().hex[:6]}-mask-",
            suffix=".png")
        os.close(fd)
        mask.save(tmp_mask)

        params = {
            "prompt": prompt.strip() or FALLBACK_PROMPT,
            "negative_prompt": negative_prompt,
            "width": cw,
            "height": ch,
            "denoise_strength": denoise_strength,
        }
        paths = pipe.inpaint(params, tmp_image, tmp_mask, loras)
        if not paths:
            raise RuntimeError("아웃페인팅 결과를 받지 못했어요.")
        with open(paths[0], "rb") as f:
            return f.read()
    finally:
        for p in (tmp_image, tmp_mask):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
