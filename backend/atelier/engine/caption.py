"""자동 캡션 (학습 데이터셋용).

- wd14: WD14 v3 ONNX (단보루 태그). onnxruntime로 CPU 실행.
- blip: Salesforce BLIP (자연어 문장). transformers로 실행.
- 모델 파일은 첫 사용 시 HuggingFace에서 내려받아 models/caption/ 에 둔다.
- numpy / onnxruntime / transformers는 함수 안에서 lazy import.
"""

from __future__ import annotations

import csv
import os
import threading
import urllib.request

_WD14_REPO = "SmilingWolf/wd-vit-tagger-v3"
_WD14_FILES = ("model.onnx", "selected_tags.csv")
_BLIP_REPO = "Salesforce/blip-image-captioning-base"

_lock = threading.Lock()
_wd14_session = None
_wd14_tags: tuple[list[str], dict[int, int]] | None = None  # (태그명, 카테고리)
_blip = None  # (processor, model, device)


def caption_models_dir() -> str:
    base = os.environ.get("ATELIER_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".atelier-ai", "models")
    d = os.path.join(base, "caption")
    os.makedirs(d, exist_ok=True)
    return d


def _download(url: str, path: str) -> None:
    if os.path.isfile(path):
        return
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
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


def _ensure_wd14() -> None:
    """WD14 ONNX 세션과 태그 목록을 준비한다."""
    global _wd14_session, _wd14_tags
    if _wd14_session is not None:
        return
    with _lock:
        if _wd14_session is not None:
            return
        import onnxruntime as ort

        d = caption_models_dir()
        for name in _WD14_FILES:
            _download(f"https://huggingface.co/{_WD14_REPO}/resolve/main/{name}",
                      os.path.join(d, name))

        tags: list[str] = []          # idx → "tag name" (공백 변환됨)
        cats: dict[int, int] = {}     # idx → category (9=캐릭터)
        with open(os.path.join(d, "selected_tags.csv"),
                  encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                tags.append(row["name"].replace("_", " "))
                cats[i] = int(row["category"])

        sess = ort.InferenceSession(
            os.path.join(d, "model.onnx"),
            providers=["CPUExecutionProvider"])
        _wd14_session = sess
        _wd14_tags = (tags, cats)


def caption_wd14(image_path: str,
                 general_thresh: float = 0.35,
                 char_thresh: float = 0.85) -> str:
    """WD14 v3로 단보루 태그를 뽑는다. 쉼표 구분 문자열 반환."""
    import numpy as np
    from PIL import Image

    _ensure_wd14()
    sess = _wd14_session
    tag_names, cats = _wd14_tags

    img = Image.open(image_path).convert("RGB").resize((448, 448))
    arr = np.asarray(img, dtype=np.float32)[:, :, ::-1]  # RGB → BGR
    arr = np.expand_dims(arr, 0)

    probs = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
    tags = []
    for i, p in enumerate(probs):
        if i >= len(tag_names):
            break
        cat = cats.get(i, 0)
        thresh = char_thresh if cat == 9 else general_thresh
        if float(p) >= thresh:
            # 점수 순 정렬용으로 (점수, 태그) 저장
            tags.append((float(p), tag_names[i]))
    tags.sort(reverse=True)
    return ", ".join(t for _, t in tags)


def _ensure_blip():
    global _blip
    if _blip is not None:
        return _blip
    with _lock:
        if _blip is not None:
            return _blip
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from atelier.engine.pipeline import _default_device

        # GPU 없으면 캡션도 막는다 (학습 전제 기능이므로)
        device = _default_device()
        processor = BlipProcessor.from_pretrained(_BLIP_REPO)
        model = BlipForConditionalGeneration.from_pretrained(_BLIP_REPO)
        model.to(device)
        model.eval()
        _blip = (processor, model, device)
        return _blip


def caption_blip(image_path: str) -> str:
    """BLIP으로 자연어 캡션을 만든다."""
    import torch
    from PIL import Image

    processor, model, device = _ensure_blip()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_image(image_path: str, method: str = "wd14") -> str:
    """method: 'wd14' | 'blip'."""
    if method == "blip":
        return caption_blip(image_path)
    return caption_wd14(image_path)
