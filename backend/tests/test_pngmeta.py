"""PNG 메타데이터 왕복 테스트: build→read, A1111 파싱, 깨진 파일→None."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from atelier.api import app as app_module
from atelier.pngmeta import (
    build_pnginfo,
    build_meta_from_kwargs,
    read_params_from_bytes,
)


@pytest.fixture()
def client():
    return TestClient(app_module.app)


def _make_png(**text_chunks) -> bytes:
    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    buf = io.BytesIO()
    if text_chunks:
        info = PngInfo()
        for k, v in text_chunks.items():
            info.add_text(k, v)
        img.save(buf, format="PNG", pnginfo=info)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def test_build_read_roundtrip():
    params = {
        "prompt": "1girl, solo, long hair",
        "negative_prompt": "lowres, bad anatomy",
        "steps": 30,
        "cfg_scale": 7.0,
        "sampler": "er_sde",
        "seed": 12345,
        "width": 768,
        "height": 1024,
        "model": "anima-2b.safetensors",
    }
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=build_pnginfo(params))
    back = read_params_from_bytes(buf.getvalue())
    assert back is not None
    for k, v in params.items():
        assert back[k] == v, k
    assert back["app"] == "atelier-ai"


def test_read_no_meta_returns_none():
    assert read_params_from_bytes(_make_png()) is None


def test_read_broken_file_returns_none():
    assert read_params_from_bytes(b"not a png at all") is None
    assert read_params_from_bytes(b"") is None
    assert read_params_from_bytes(b"\x89PNG\r\n\x1a\n\x00") is None


def test_read_jpeg_returns_none():
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    assert read_params_from_bytes(buf.getvalue()) is None


def test_a1111_parameters_parsing():
    text = (
        "masterpiece, 1girl, solo, long hair\n"
        "Negative prompt: lowres, bad anatomy, bad hands\n"
        "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, "
        "Seed: 987654, Size: 768x1024, Model: anima-2b"
    )
    back = read_params_from_bytes(_make_png(parameters=text))
    assert back is not None
    assert back["prompt"] == "masterpiece, 1girl, solo, long hair"
    assert back["negative_prompt"] == "lowres, bad anatomy, bad hands"
    assert back["steps"] == 28
    assert back["sampler"] == "DPM++ 2M Karras"
    assert back["cfg_scale"] == 7
    assert back["seed"] == 987654
    assert back["width"] == 768
    assert back["height"] == 1024
    assert back["model"] == "anima-2b"


def test_a1111_parameters_without_negative():
    text = "1girl, solo\nSteps: 20, Seed: 42, Size: 512x512"
    back = read_params_from_bytes(_make_png(parameters=text))
    assert back is not None
    assert back["prompt"] == "1girl, solo"
    assert back["negative_prompt"] == ""
    assert back["steps"] == 20
    assert back["seed"] == 42


def test_a1111_multiline_negative():
    text = (
        "1girl\n"
        "Negative prompt: lowres,\nbad anatomy\n"
        "Steps: 20, Seed: 1"
    )
    back = read_params_from_bytes(_make_png(parameters=text))
    assert back is not None
    assert back["negative_prompt"] == "lowres,\nbad anatomy"
    assert back["steps"] == 20


def test_atelier_params_preferred_over_a1111():
    """두 청크가 다 있으면 atelier:params를 쓴다."""
    own = json.dumps({"prompt": "ours", "seed": 7}, ensure_ascii=False)
    back = read_params_from_bytes(
        _make_png(**{"atelier:params": own, "parameters": "1girl\nSteps: 20"})
    )
    assert back is not None
    assert back["prompt"] == "ours"


def test_build_meta_from_kwargs():
    kwargs = {
        "prompt": "1girl",
        "negative_prompt": "lowres",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 768,
    }
    meta = build_meta_from_kwargs(kwargs, seed=99, sampler_name="ddim")
    assert meta == {
        "prompt": "1girl",
        "negative_prompt": "lowres",
        "steps": 30,
        "cfg_scale": 7.5,
        "sampler": "ddim",
        "seed": 99,
        "width": 512,
        "height": 768,
    }


def test_png_info_endpoint_roundtrip(client):
    """POST /api/png-info: 메타 있는 PNG → found=true, 없으면 found=false."""
    params = {
        "prompt": "1girl, solo",
        "negative_prompt": "lowres",
        "steps": 20,
        "cfg_scale": 7.0,
        "sampler": "ddim",
        "seed": 1,
        "width": 512,
        "height": 512,
    }
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=build_pnginfo(params))
    r = client.post(
        "/api/png-info",
        files={"file": ("gen.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["params"]["prompt"] == "1girl, solo"
    assert body["params"]["seed"] == 1

    # 메타 없는 PNG
    plain = io.BytesIO()
    Image.new("RGB", (16, 16)).save(plain, format="PNG")
    r = client.post(
        "/api/png-info",
        files={"file": ("plain.png", plain.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False
    assert body["params"] is None
