"""ADetailer 스타일 얼굴 보정 (얼굴 검출 + 인페인트).

- face_yolov8n.pt (Bingsu/adetailer, YOLOv8n 얼굴 검출, 약 6MB)를
  ultralytics로 실행한다.
- 모델 파일은 첫 사용 시 HuggingFace에서
  ~/.atelier-ai/models/adetailer/ 로 자동 다운로드한다
  (pose_detect.py의 _download 패턴 재사용).
- 검출된 모든 얼굴 bbox를 padding으로 확장하고 blur로 feathering한
  결합 마스크를 만들어 pipeline.inpaint를 한 번 호출한다
  (ADetailer의 "모든 얼굴을 하나의 마스크로" 방식).
- ultralytics는 함수 안에서 lazy import하므로 이 모듈 자체는
  ultralytics/torch 없이 import 가능하다 (pipeline.py 패턴).
- GPU가 없으면 pipeline.py의 _default_device()가 RuntimeError를 내고,
  API는 이를 503으로 전달한다.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from atelier.engine.pipeline import _default_device
from atelier.engine.pose_detect import _download

# ---------------------------------------------------------------------------
# 모델 파일
# ---------------------------------------------------------------------------

_MODEL_FILE = "face_yolov8n.pt"
_MODEL_URL = (
    "https://huggingface.co/Bingsu/adetailer/resolve/main/" + _MODEL_FILE
)

# 디노이즈 강도 (0=원본 유지, 1=완전히 다시 그림)
MIN_DENOISING = 0.0
MAX_DENOISING = 1.0
DEFAULT_DENOISING = 0.4

# 얼굴 검출 신뢰도
MIN_CONF = 0.0
MAX_CONF = 1.0
DEFAULT_CONF = 0.5

# 마스크 확장·feathering
DEFAULT_PADDING = 32
DEFAULT_BLUR = 8

# 프롬프트가 비어 있을 때 쓰는 폴백 (원본 프롬프트를 모를 때)
FALLBACK_PROMPT = "detailed face, beautiful detailed eyes, high quality"


def adetailer_models_dir() -> str:
    """ADetailer 모델 폴더. 없으면 만들어 둔다."""
    base = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    d = os.path.join(base, "adetailer")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_adetailer_model() -> str:
    """face_yolov8n.pt가 있는지 확인하고, 없으면 받는다."""
    d = adetailer_models_dir()
    path = os.path.join(d, _MODEL_FILE)
    if not os.path.isfile(path):
        print("[atelier] 얼굴 검출 모델을 내려받습니다 (첫 실행 시 약 6MB)...")
    _download(_MODEL_URL, path)
    return path


# ---------------------------------------------------------------------------
# YOLO 얼굴 검출 (lazy, 스레드 안전)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_detector = None  # (model, device) 캐시


def _ultralytics():
    try:
        import ultralytics
    except ImportError:
        raise RuntimeError(
            "얼굴 보정에는 ultralytics가 필요해요. "
            "백엔드 가상환경에 ultralytics를 설치해주세요."
        )
    return ultralytics


def get_detector(device: str):
    """face_yolov8n YOLO 모델 (캐시). device가 바뀌면 다시 로드한다."""
    global _detector
    with _lock:
        if _detector is not None and _detector[1] == device:
            return _detector[0]
        YOLO = _ultralytics().YOLO
        model = YOLO(ensure_adetailer_model())
        _detector = (model, device)
        return model


def reset_detector() -> None:
    """모델 캐시 초기화. 테스트용."""
    global _detector
    with _lock:
        _detector = None


def detect_faces(image_bytes: bytes, conf: float = DEFAULT_CONF,
                 device: str | None = None
                 ) -> list[tuple[float, float, float, float, float]]:
    """이미지 바이트 → 얼굴 bbox [(x1, y1, x2, y2, conf)] (원본 좌표).

    Raises:
        ValueError: 이미지 오류
        RuntimeError: GPU 없음 / ultralytics 없음
    """
    if not (MIN_CONF <= conf <= MAX_CONF):
        raise ValueError(
            f"검출 신뢰도는 {MIN_CONF}~{MAX_CONF} 사이여야 해요.")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("이미지 파일이 올바르지 않아요.")

    device = device or _default_device()
    model = get_detector(device)

    results = model.predict(np.asarray(img), conf=conf,
                            device=device, verbose=False)
    boxes: list[tuple[float, float, float, float, float]] = []
    for r in results:
        b = getattr(r, "boxes", None)
        if b is None:
            continue
        xyxy = np.asarray(b.xyxy)
        confs = np.asarray(b.conf)
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append(
                (float(x1), float(y1), float(x2), float(y2), float(c)))
    # 신뢰도 높은 순
    boxes.sort(key=lambda t: t[4], reverse=True)
    return boxes


# ---------------------------------------------------------------------------
# 마스크
# ---------------------------------------------------------------------------

def build_face_mask(img: Image.Image,
                    boxes: list[tuple[float, float, float, float, float]],
                    padding: int = DEFAULT_PADDING,
                    blur: int = DEFAULT_BLUR) -> Image.Image:
    """모든 얼굴 bbox를 padding으로 확장한 결합 마스크 (L 모드).

    흰색=다시 그릴 영역. blur로 가장자리를 feathering한다.
    """
    if padding < 0:
        raise ValueError("마스크 확장은 0 이상이어야 해요.")
    if blur < 0:
        raise ValueError("마스크 블러는 0 이상이어야 해요.")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2, _c in boxes:
        draw.rectangle(
            [x1 - padding, y1 - padding, x2 + padding, y2 + padding],
            fill=255,
        )
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def retouch(image_bytes: bytes, prompt: str,
            negative_prompt: str = "",
            denoising_strength: float = DEFAULT_DENOISING,
            conf: float = DEFAULT_CONF,
            mask_padding: int = DEFAULT_PADDING,
            mask_blur: int = DEFAULT_BLUR,
            pipeline=None, loras: dict | None = None,
            device: str | None = None) -> tuple[str | None, int]:
    """얼굴 보정: 검출된 얼굴 영역만 인페인트로 다시 그린다.

    Args:
        image_bytes: 원본 이미지 바이트.
        prompt: 인페인트 프롬프트. 비어 있으면 폴백 프롬프트 사용.
        pipeline: atelier.engine.pipeline.Pipeline (inpaint 호출용).
            None이면 RuntimeError.
        loras: pipeline.inpaint에 그대로 전달.

    Returns:
        (결과 파일 경로, 검출된 얼굴 수).
        얼굴이 없으면 (None, 0) — 원본을 그대로 쓰라는 뜻.

    Raises:
        ValueError: 이미지 오류 / 파라미터 범위 밖
        RuntimeError: GPU 없음 (→ API 503)
    """
    if not (MIN_DENOISING <= denoising_strength <= MAX_DENOISING):
        raise ValueError(
            f"디노이즈 강도는 {MIN_DENOISING}~{MAX_DENOISING} 사이여야 해요.")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("이미지 파일이 올바르지 않아요.")

    device = device or _default_device()
    boxes = detect_faces(image_bytes, conf=conf, device=device)
    if not boxes:
        return None, 0
    if pipeline is None:
        raise RuntimeError(
            "얼굴 보정에는 생성 파이프라인이 필요해요.")

    mask = build_face_mask(img, boxes,
                           padding=mask_padding, blur=mask_blur)
    if not (prompt or "").strip():
        prompt = FALLBACK_PROMPT

    # inpaint는 파일 경로를 받으므로 임시 파일로 저장한다.
    # height/width는 원본 크기로 고정 (기본값으로 바뀌면 화면비가 깨짐).
    w, h = img.size
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "denoise_strength": denoising_strength,
        "height": h,
        "width": w,
    }
    with tempfile.TemporaryDirectory(prefix="adetailer-") as tmp:
        img_path = os.path.join(tmp, "image.png")
        mask_path = os.path.join(tmp, "mask.png")
        img.save(img_path, format="PNG")
        mask.save(mask_path, format="PNG")
        paths = pipeline.inpaint(params, img_path, mask_path, loras)
    if not paths:
        raise RuntimeError("얼굴 보정에 실패했어요.")
    return paths[0], len(boxes)
