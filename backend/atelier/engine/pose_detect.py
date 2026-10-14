"""사진 → OpenPose 스켈레톤 PNG (포즈 자동 추출).

- DWPose ONNX 2종을 onnxruntime으로 실행한다 (torch/diffusers 불필요).
  - yolox_l.onnx: 사람 검출 (입력 1x3x640x640, 출력 1x8400x85)
  - dw-ll_ucoco_384.onnx: 포즈 추정 (입력 1x3x384x288,
    SimCC 출력 → 133 키포인트(COCO-WholeBody))
- 모델 파일은 첫 사용 시 HuggingFace(yzd-v/DWPose)에서
  ~/.atelier-ai/models/pose/ 로 자동 다운로드한다 (약 350MB).
- 출력: 검정 배경 + 흰색 스켈레톤 선 PNG (OpenPose COCO-18 연결선).
  그리기는 PIL/numpy만 사용 (opencv 없음).
- 전처리·디코딩은 DWPose 공식 onnx 추론 코드와 동일한 방식:
  YOLOX letterbox(114 패딩) + stride 디코딩,
  포즈는 aspect-ratio 유지 affine warp + BGR ImageNet 정규화,
  SimCC는 argmax/2 디코딩 + raw logit 점수.
- 이 모듈은 onnxruntime 없이도 import 가능해야 한다
  (onnxruntime은 세션 생성 시점에 lazy import).
"""

from __future__ import annotations

import io
import os
import threading
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# 모델 파일
# ---------------------------------------------------------------------------

_HF_REPO = "yzd-v/DWPose"
_DET_FILE = "yolox_l.onnx"
_POSE_FILE = "dw-ll_ucoco_384.onnx"

_DET_INPUT = 640          # YOLOX 입력 한 변
_POSE_W, _POSE_H = 288, 384  # DWPose 입력 (W, H)
_SIMCC_SPLIT = 2.0        # SimCC 디코딩: argmax / 2

# ImageNet 정규화 (RTMPose/DWPose 표준 전처리)
_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# 스켈레톤을 그릴 때 이 점수 미만인 관절은 연결선에서 제외
# (score는 SimCC raw logit 최대값. 공식 코드의 가시성 기준 0.3)
MIN_LIMB_SCORE = 0.3


class NoPersonError(ValueError):
    """사진에서 사람을 찾지 못했을 때."""


def pose_models_dir() -> str:
    """포즈 모델 폴더. 없으면 만들어 둔다."""
    base = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models"
    )
    d = os.path.join(base, "pose")
    os.makedirs(d, exist_ok=True)
    return d


def _download(url: str, path: str) -> None:
    """파일이 없으면 다운로드한다."""
    if os.path.isfile(path):
        return
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)


def ensure_pose_models() -> tuple[str, str]:
    """두 ONNX 파일이 있는지 확인하고, 없으면 HF에서 받는다.

    Returns:
        (yolox 경로, dwpose 경로)
    """
    d = pose_models_dir()
    det_path = os.path.join(d, _DET_FILE)
    pose_path = os.path.join(d, _POSE_FILE)
    if not (os.path.isfile(det_path) and os.path.isfile(pose_path)):
        print("[atelier] 포즈 모델을 내려받습니다 (첫 실행 시 약 350MB)...")
    _download(f"https://huggingface.co/{_HF_REPO}/resolve/main/{_DET_FILE}", det_path)
    _download(f"https://huggingface.co/{_HF_REPO}/resolve/main/{_POSE_FILE}", pose_path)
    return det_path, pose_path


# ---------------------------------------------------------------------------
# ONNX 세션 (lazy, 스레드 안전)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_det_session = None
_pose_session = None


def _ort():
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "포즈 추출에는 onnxruntime이 필요해요. "
            "백엔드 가상환경에 onnxruntime을 설치해주세요."
        )
    return ort


def get_detector():
    """YOLOX 사람 검출 세션 (캐시)."""
    global _det_session
    if _det_session is None:
        with _lock:
            if _det_session is None:
                det_path, _ = ensure_pose_models()
                _det_session = _ort().InferenceSession(
                    det_path, providers=["CPUExecutionProvider"]
                )
    return _det_session


def get_pose_estimator():
    """DWPose 포즈 추정 세션 (캐시)."""
    global _pose_session
    if _pose_session is None:
        with _lock:
            if _pose_session is None:
                _, pose_path = ensure_pose_models()
                _pose_session = _ort().InferenceSession(
                    pose_path, providers=["CPUExecutionProvider"]
                )
    return _pose_session


def reset_sessions() -> None:
    """세션 캐시 초기화. 테스트용."""
    global _det_session, _pose_session
    with _lock:
        _det_session = None
        _pose_session = None


# ---------------------------------------------------------------------------
# YOLOX 사람 검출
# ---------------------------------------------------------------------------

def _letterbox(img_bgr: np.ndarray, size: int = _DET_INPUT):
    """114 회색 패딩으로 정사각형 letterbox. (패드 이미지, 비율) 반환."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(w * r), int(h * r)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = np.array(Image.fromarray(img_bgr).resize((nw, nh), Image.BILINEAR))
    canvas[:nh, :nw] = resized
    return canvas, r


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5):
    """numpy NMS. boxes: (N,4) x1y1x2y2. 유지할 인덱스 반환."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a))


def _decode_yolox_raw(pred: np.ndarray) -> np.ndarray:
    """YOLOX raw 출력 (8400,85) → 디코딩된 (8400,85).

    출력은 stride 8/16/32 레벨 순으로 이어져 있으며,
    박스는 grid-cell 기준 raw 값이므로 sigmoid/exp 디코딩이 필요하다.
    """
    out = pred.copy()
    pos = 0
    for stride in (8, 16, 32):
        n = (_DET_INPUT // stride) ** 2
        seg = out[pos:pos + n]
        gx, gy = np.meshgrid(np.arange(_DET_INPUT // stride),
                             np.arange(_DET_INPUT // stride))
        seg[:, 0] = (_sigmoid(seg[:, 0]) + gx.ravel()) * stride
        seg[:, 1] = (_sigmoid(seg[:, 1]) + gy.ravel()) * stride
        seg[:, 2] = np.exp(seg[:, 2]) * stride
        seg[:, 3] = np.exp(seg[:, 3]) * stride
        pos += n
    return out


def decode_yolox(output: np.ndarray, ratio: float,
                 img_w: int, img_h: int,
                 conf_thres: float = 0.3) -> list[tuple[float, float, float, float]]:
    """YOLOX 출력 (1,8400,85) → 사람 박스 [(x1,y1,x2,y2)] (원본 좌표)."""
    pred = _decode_yolox_raw(np.asarray(output[0], dtype=np.float64))
    boxes = pred[:, :4]
    # 주의: 이 ONNX 익스포트는 박스만 raw이고 obj/cls는 이미 sigmoid된 확률이다.
    scores = pred[:, 4] * pred[:, 5:].max(axis=1)
    labels = pred[:, 5:].argmax(axis=1)
    keep = (labels == 0) & (scores > conf_thres)  # COCO class 0 = person
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return []
    # cxcywh(640 공간) → x1y1x2y2(원본 좌표)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) / ratio
    y1 = (cy - h / 2) / ratio
    x2 = (cx + w / 2) / ratio
    y2 = (cy + h / 2) / ratio
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)
    idx = _nms(xyxy, scores)
    return [tuple(map(float, xyxy[i])) for i in idx]


def detect_persons(img_rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """RGB numpy 이미지에서 사람 박스를 찾는다."""
    h, w = img_rgb.shape[:2]
    # YOLOX 공식 전처리와 동일: BGR, 0~255, 정규화 없음
    img_bgr = img_rgb[:, :, ::-1]
    canvas, ratio = _letterbox(img_bgr)
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    sess = get_detector()
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    return decode_yolox(np.asarray(out), ratio, w, h)


# ---------------------------------------------------------------------------
# DWPose 포즈 추정 (SimCC 디코딩)
# ---------------------------------------------------------------------------

def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
    """SimCC 출력 → (K,3) [x, y, score] (288x384 입력 공간 좌표).

    DWPose 공식 onnx 추론 코드와 동일한 디코딩:
    - 좌표 = argmax / split_ratio(2.0)
    - score = x/y raw logit 최대값 중 작은 쪽 (softmax 없음)
    - score <= 0 이면 무효 (-1)
    """
    x_idx = simcc_x.argmax(axis=1).astype(np.float32)
    y_idx = simcc_y.argmax(axis=1).astype(np.float32)
    score = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1))
    x = x_idx / _SIMCC_SPLIT
    y = y_idx / _SIMCC_SPLIT
    invalid = score <= 0
    x[invalid] = -1.0
    y[invalid] = -1.0
    return np.stack([x, y, score], axis=1)


def _warp_crop(img_rgb: np.ndarray,
               box: tuple[float, float, float, float]
               ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """박스를 aspect-ratio 유지 affine warp로 288x384 크롭으로 만든다.

    DWPose 공식 onnx 추론 코드(top_down_affine)와 동일한 방식:
    박스를 1.25배 확장한 뒤, 288/384 비율에 맞춰 짧은 쪽을 늘리고
    warp한다 (찌그러뜨리지 않음).
    Returns: (크롭 RGB (384,288,3), (ex1, ey1, bw, bh))
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * 1.25, (y2 - y1) * 1.25
    target_ar = _POSE_W / _POSE_H
    if bw > bh * target_ar:
        bh = bw / target_ar
    else:
        bw = bh * target_ar
    ex1, ey1 = cx - bw / 2.0, cy - bh / 2.0
    # PIL AFFINE은 output→input 역행렬:
    # input_x = ex1 + x_out/288*bw, input_y = ey1 + y_out/384*bh
    crop = Image.fromarray(img_rgb).transform(
        (_POSE_W, _POSE_H), Image.AFFINE,
        (bw / _POSE_W, 0.0, ex1, 0.0, bh / _POSE_H, ey1),
        resample=Image.BILINEAR,
    )
    return np.asarray(crop), (ex1, ey1, bw, bh)


def estimate_pose(img_rgb: np.ndarray,
                  box: tuple[float, float, float, float]) -> np.ndarray:
    """박스 안의 사람을 133 키포인트로 추정한다. (133,3) [x,y,score] (원본 좌표).

    전처리(DWPose 공식 onnx 추론 코드와 동일):
    BGR 변환 없이... 가 아니라, BGR 순서 그대로 ImageNet 정규화
    (공식 cv2 코드는 cv2.imread(BGR)를 채널 스왑 없이 정규화한다).
    """
    crop, (ex1, ey1, bw, bh) = _warp_crop(img_rgb, box)
    bgr = crop[:, :, ::-1].astype(np.float32)
    normed = (bgr - _IMAGENET_MEAN) / _IMAGENET_STD
    blob = normed.transpose(2, 0, 1)[None].astype(np.float32)

    sess = get_pose_estimator()
    in_name = sess.get_inputs()[0].name
    sx, sy = (np.asarray(o) for o in sess.run(None, {in_name: blob}))
    kpts = decode_simcc(sx[0], sy[0])  # (133,3), 288x384 공간

    # 288x384 → 원본 좌표 (공식 후처리와 동일)
    kpts[:, 0] = ex1 + kpts[:, 0] / _POSE_W * bw
    kpts[:, 1] = ey1 + kpts[:, 1] / _POSE_H * bh
    return kpts


# ---------------------------------------------------------------------------
# WholeBody-133 → OpenPose COCO-18
# ---------------------------------------------------------------------------

# COCO-17 인덱스 (wholebody 앞 17개와 동일):
# 0 코, 1 왼쪽눈, 2 오른쪽눈, 3 왼쪽귀, 4 오른쪽귀,
# 5 왼쪽어깨, 6 오른쪽어깨, 7 왼쪽팔꿈치, 8 오른쪽팔꿈치,
# 9 왼쪽손목, 10 오른쪽손목, 11 왼쪽골반, 12 오른쪽골반,
# 13 왼쪽무릎, 14 오른쪽무릎, 15 왼쪽발목, 16 오른쪽발목
# (좌/우는 사람 기준)

def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def wholebody_to_coco18(kpts133: np.ndarray) -> np.ndarray:
    """(133,3) → (18,3) OpenPose COCO-18 순서.

    0 코, 1 목, 2 오_어깨, 3 오_팔꿈치, 4 오_손목,
    5 왼_어깨, 6 왼_팔꿈치, 7 왼_손목, 8 골반중앙,
    9 오_골반, 10 오_무릎, 11 오_발목,
    12 왼_골반, 13 왼_무릎, 14 왼_발목,
    15 오_눈, 16 왼_눈, 17 오_귀
    """
    wb = kpts133
    neck = _mid(wb[5], wb[6])
    midhip = _mid(wb[11], wb[12])
    pts = np.stack([
        wb[0], neck,
        wb[6], wb[8], wb[10],
        wb[5], wb[7], wb[9],
        midhip,
        wb[12], wb[14], wb[16],
        wb[11], wb[13], wb[15],
        wb[2], wb[1], wb[4],
    ], axis=0)
    # 중간점(목/골반중앙)의 score는 양쪽 평균
    pts[1, 2] = (wb[5, 2] + wb[6, 2]) / 2.0
    pts[8, 2] = (wb[11, 2] + wb[12, 2]) / 2.0
    return pts


COCO18_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (0, 16),
]


# ---------------------------------------------------------------------------
# 스켈레톤 그리기
# ---------------------------------------------------------------------------

def draw_skeleton(image_size: tuple[int, int],
                  persons: list[np.ndarray]) -> Image.Image:
    """검정 배경에 흰색 스켈레톤을 그린다. persons: [(18,3)] 리스트."""
    img = Image.new("RGB", image_size, (0, 0, 0))
    if not persons:
        return img
    w, h = image_size
    lw = max(2, round(min(w, h) / 200))  # 선 굵기: 이미지 크기에 비례
    r = lw  # 관절 원 반지름
    draw = ImageDraw.Draw(img)
    for pts in persons:
        for a, b in COCO18_PAIRS:
            if pts[a, 2] < MIN_LIMB_SCORE or pts[b, 2] < MIN_LIMB_SCORE:
                continue
            draw.line([tuple(pts[a, :2]), tuple(pts[b, :2])],
                      fill=(255, 255, 255), width=lw)
        for x, y, s in pts:
            if s < MIN_LIMB_SCORE:
                continue
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    return img


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def detect_pose_skeleton(image_bytes: bytes, max_side: int = 1024) -> bytes:
    """사진 바이트 → 스켈레톤 PNG 바이트.

    Raises:
        NoPersonError: 사람 검출 실패 (한국어 메시지)
        RuntimeError: onnxruntime 없음 등
        ValueError: 이미지가 아님
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("이미지 파일이 올바르지 않아요.")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        w, h = img.size
    img_rgb = np.asarray(img)

    boxes = detect_persons(img_rgb)
    if not boxes:
        raise NoPersonError(
            "사진에서 사람을 찾지 못했어요. "
            "인물이 잘 보이는 사진으로 다시 시도해주세요."
        )

    persons = [wholebody_to_coco18(estimate_pose(img_rgb, b)) for b in boxes]
    # 스켈레톤 좌표는 이미 원본(축소 반영) 좌표계이므로 그대로 그린다
    out = draw_skeleton((w, h), persons)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
