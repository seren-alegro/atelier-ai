"""RealESRGAN x4plus anime 6B 업스케일 (4x, 애니메이션 특화).

- RRDBNet 아키텍처를 직접 구현한다
  (num_feat=64, num_block=6, num_grow_ch=32, scale=4 —
  xinntao/Real-ESRGAN의 RealESRGAN_x4plus_anime_6B 설정과 동일).
- 모델 파일은 첫 사용 시 GitHub 릴리스에서
  ~/.atelier-ai/models/upscale/ 로 자동 다운로드한다 (약 18MB,
  pose_detect.py의 _download 패턴 재사용).
- 강도 블렌드: result = strength * esrgan + (1 - strength) * lanczos
  (1.0 = 풀 업스케일, 0.5 = 절반만 적용).
- 큰 이미지는 타일(기본 512px, overlap 16)로 나눠 처리해 메모리를 보호한다.
- GPU가 없으면 pipeline.py의 _default_device()가 RuntimeError를 내고,
  API는 이를 503으로 전달한다.
- torch는 함수 안에서 lazy import하므로 이 모듈 자체는
  GPU·torch 없이 import 가능하다 (pipeline.py 패턴).
"""

from __future__ import annotations

import io
import os
import threading

import numpy as np
from PIL import Image

from atelier.engine.pipeline import _default_device
from atelier.engine.pose_detect import _download

# ---------------------------------------------------------------------------
# 모델 파일
# ---------------------------------------------------------------------------

_SCALE = 4
_MODEL_FILE = "RealESRGAN_x4plus_anime_6B.pth"
_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/"
    + _MODEL_FILE
)

MIN_STRENGTH = 0.5
MAX_STRENGTH = 1.0


def upscale_models_dir() -> str:
    """업스케일 모델 폴더. 없으면 만들어 둔다."""
    base = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    d = os.path.join(base, "upscale")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_upscale_model() -> str:
    """RealESRGAN_x4plus_anime_6B.pth가 있는지 확인하고, 없으면 받는다."""
    d = upscale_models_dir()
    path = os.path.join(d, _MODEL_FILE)
    if not os.path.isfile(path):
        print("[atelier] 업스케일 모델을 내려받습니다 (첫 실행 시 약 18MB)...")
    _download(_MODEL_URL, path)
    return path


# ---------------------------------------------------------------------------
# RRDBNet (RealESRGAN 공식 아키텍처 — torch는 함수 내 lazy import)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_upscaler = None  # (model, device) 캐시


def _build_rrdbnet():
    """torch lazy import + RRDBNet 클래스 정의/인스턴스 생성."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(
                num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(
                num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(
                num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    def _make_layer(block, n_layers, **kwargs):
        return nn.Sequential(
            *(block(**kwargs) for _ in range(n_layers)))

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                     num_feat=64, num_block=6, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = _make_layer(
                ResidualDenseBlock, num_block,
                num_feat=num_feat, num_grow_ch=num_grow_ch)
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # 업샘플: nearest 보간 + conv (공식 RRDBNet 방식)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(
                F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(
                F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet(num_in_ch=3, num_out_ch=3, scale=_SCALE,
                   num_feat=64, num_block=6, num_grow_ch=32)


def get_upscaler(device: str):
    """RealESRGAN 모델 (캐시). device가 바뀌면 다시 로드한다."""
    global _upscaler
    with _lock:
        if _upscaler is not None and _upscaler[1] == device:
            return _upscaler[0]
        import torch
        model = _build_rrdbnet()
        state = torch.load(
            ensure_upscale_model(), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        _upscaler = (model, device)
        return model


def reset_upscaler() -> None:
    """모델 캐시 초기화. 테스트용."""
    global _upscaler
    with _lock:
        _upscaler = None


# ---------------------------------------------------------------------------
# 타일 분할 추론
# ---------------------------------------------------------------------------

def _run_tiled(model, img: np.ndarray, device: str,
               tile: int, overlap: int) -> np.ndarray:
    """RGB float32 0..1 numpy (H,W,3) → 4x 업스케일 float32 numpy.

    타일(inner step = tile - 2*overlap)마다 overlap 마진을 붙여 추론하고
    안쪽만 잘라 붙인다. 가장자리는 클램프한다.
    """
    import torch

    h, w = img.shape[:2]
    out = np.zeros((h * _SCALE, w * _SCALE, 3), dtype=np.float32)
    step = tile - 2 * overlap

    with torch.no_grad():
        for y in range(0, h, step):
            for x in range(0, w, step):
                y0 = max(0, y - overlap)
                x0 = max(0, x - overlap)
                y1 = min(h, y + step + overlap)
                x1 = min(w, x + step + overlap)
                tile_img = img[y0:y1, x0:x1]
                t = torch.from_numpy(
                    tile_img.transpose(2, 0, 1)[None]).to(device)
                pred = model(t).clamp_(0, 1).cpu().numpy()[0]
                pred = pred.transpose(1, 2, 0)  # (4*th, 4*tw, 3)

                # 이 타일이 맡는 출력 영역 (안쪽만)
                oy0, ox0 = y * _SCALE, x * _SCALE
                oy1 = min(h, y + step) * _SCALE
                ox1 = min(w, x + step) * _SCALE
                # pred 안에서의 오프셋 (입력 클램프 마진 × 4)
                py0 = (y - y0) * _SCALE
                px0 = (x - x0) * _SCALE
                out[oy0:oy1, ox0:ox1] = pred[
                    py0:py0 + (oy1 - oy0),
                    px0:px0 + (ox1 - ox0)]
    return out


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def upscale_image(image_bytes: bytes, strength: float = 0.5,
                  tile: int = 512, tile_overlap: int = 16,
                  device: str | None = None) -> bytes:
    """이미지 바이트 → 4x 업스케일 PNG 바이트.

    Raises:
        ValueError: 이미지 오류 / 강도 범위(0.5~1.0) 밖
        RuntimeError: GPU 없음 (_default_device가 냄)
    """
    if not (MIN_STRENGTH <= strength <= MAX_STRENGTH):
        raise ValueError(
            f"업스케일 강도는 {MIN_STRENGTH}~{MAX_STRENGTH} 사이여야 해요.")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("이미지 파일이 올바르지 않아요.")

    device = device or _default_device()
    model = get_upscaler(device)

    img_f = np.asarray(img).astype(np.float32) / 255.0
    esrgan = _run_tiled(model, img_f, device, tile, tile_overlap)

    # Lanczos 기준선: 원본을 4x로 단순 확대
    w, h = img.size
    lanczos = np.asarray(
        img.resize((w * _SCALE, h * _SCALE), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    blended = strength * esrgan + (1.0 - strength) * lanczos
    blended = np.clip(blended, 0, 1)
    out_img = Image.fromarray((blended * 255.0 + 0.5).astype(np.uint8))

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()
