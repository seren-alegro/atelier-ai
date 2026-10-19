"""모델 다운로드 API 테스트.

ATELIER_MODELS_DIR를 tmp로 격리한다.
실제 네트워크 다운로드는 로컬 HTTP 서버로 모킹한다.
"""

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from atelier.api import app as app_module
from atelier.api.routes import downloads
from atelier.api.routes.downloads import (
    classify_url,
    filename_from_url,
    safe_filename,
)


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    """샌드박스의 깨진 프록시 환경변수를 테스트에서 제거."""
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ATELIER_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ATELIER_DATA_DIR", str(tmp_path / "data"))
    yield TestClient(app_module.app)


@pytest.fixture()
def file_server(tmp_path):
    """작은 바이너리를 서빙하는 로컬 HTTP 서버."""
    payload = b"FAKEMODEL" * 1024  # 9KB

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = payload
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Disposition",
                'attachment; filename="served-model.safetensors"',
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # noqa: D102
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _wait_done(client, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/downloads/{job_id}").json()
        if st["status"] != "downloading":
            return st
        time.sleep(0.2)
    raise AssertionError("다운로드가 끝나지 않음")


# -- URL 판별 -------------------------------------------------------------


def test_classify_civitai():
    src, info = classify_url("https://civitai.com/models/12345/my-model")
    assert src == "civitai"
    assert info == {"model_id": "12345", "version_id": None}


def test_classify_civitai_version():
    src, info = classify_url(
        "https://civitai.com/models/12345/x?modelVersionId=67890"
    )
    assert src == "civitai"
    assert info["version_id"] == "67890"


def test_classify_civitai_not_model_page():
    with pytest.raises(Exception):
        classify_url("https://civitai.com/images/999")


def test_classify_huggingface_blob_to_resolve():
    src, info = classify_url(
        "https://huggingface.co/user/repo/blob/main/model.safetensors"
    )
    assert src == "huggingface"
    assert "/resolve/main/model.safetensors" in info["direct_url"]
    assert "/blob/" not in info["direct_url"]


def test_classify_direct():
    src, info = classify_url("https://example.com/a/model.ckpt")
    assert src == "direct"
    assert info["direct_url"].endswith("model.ckpt")


def test_classify_bad_scheme():
    with pytest.raises(Exception):
        classify_url("ftp://example.com/model.safetensors")


# -- 파일명 안전화 ----------------------------------------------------------


def test_safe_filename_ok():
    assert safe_filename("my-model.safetensors") == "my-model.safetensors"
    assert safe_filename("a%20b.ckpt") == "a b.ckpt"


def test_safe_filename_strips_path():
    # 경로 조작 시도는 basename만 남는다 (탈출 불가)
    assert safe_filename("../../evil.safetensors") == "evil.safetensors"
    assert safe_filename("sub/dir/model.safetensors") == "model.safetensors"


def test_safe_filename_bad_ext():
    with pytest.raises(Exception):
        safe_filename("model.bin")


def test_safe_filename_empty():
    with pytest.raises(Exception):
        safe_filename("")


def test_filename_from_url():
    assert (
        filename_from_url("https://x.com/a/b.safetensors?dl=1")
        == "b.safetensors"
    )


# -- 다운로드 흐름 (로컬 서버) -----------------------------------------------


def test_download_checkpoint_auto(client, file_server, tmp_path):
    r = client.post(
        "/api/downloads",
        json={"url": f"{file_server}/some-model.safetensors", "kind": "auto"},
    )
    assert r.status_code == 202
    st = _wait_done(client, r.json()["job_id"])
    assert st["status"] == "done"
    # Content-Disposition 파일명으로 저장됨
    assert st["filename"] == "served-model.safetensors"
    dest = tmp_path / "models" / "served-model.safetensors"
    assert dest.is_file()
    assert not (tmp_path / "models" / "served-model.safetensors.part").exists()


def test_download_lora_kind(client, file_server, tmp_path):
    r = client.post(
        "/api/downloads",
        json={"url": f"{file_server}/m.safetensors", "kind": "lora"},
    )
    assert r.status_code == 202
    st = _wait_done(client, r.json()["job_id"])
    assert st["status"] == "done"
    assert (tmp_path / "models" / "lora" / "served-model.safetensors").is_file()


def test_download_kind_auto_lora_by_name(client, file_server, tmp_path):
    r = client.post(
        "/api/downloads",
        json={"url": f"{file_server}/cool-lora.safetensors", "kind": "auto"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # 파일명에 lora가 있으면 lora 폴더로 (CD 파일명보다 URL 기준 판정)
    st = _wait_done(client, job_id)
    assert st["status"] == "done"
    assert (tmp_path / "models" / "lora" / st["filename"]).is_file()


def test_download_bad_scheme_rejected(client):
    r = client.post(
        "/api/downloads",
        json={"url": "ftp://x.com/m.safetensors", "kind": "auto"},
    )
    assert r.status_code == 400


def test_download_bad_kind(client, file_server):
    r = client.post(
        "/api/downloads",
        json={"url": f"{file_server}/m.safetensors", "kind": "nope"},
    )
    assert r.status_code == 400


def test_download_pack_dir(client, file_server, tmp_path, monkeypatch):
    pack = tmp_path / "models" / "anima38-test"
    pack.mkdir(parents=True)
    # 팩 판정용 최소 파일 (qwen35_4b)
    (pack / "qwen35_4b.safetensors").write_bytes(b"x")
    r = client.post(
        "/api/downloads",
        json={
            "url": f"{file_server}/Anima-3.8B-v1.safetensors",
            "kind": "pack",
            "pack_dir": "anima38-test",
        },
    )
    assert r.status_code == 202
    st = _wait_done(client, r.json()["job_id"])
    assert st["status"] == "done"
    assert (pack / st["filename"]).is_file()


def test_download_pack_traversal_blocked(client, file_server):
    r = client.post(
        "/api/downloads",
        json={
            "url": f"{file_server}/m.safetensors",
            "kind": "pack",
            "pack_dir": "../evil",
        },
    )
    assert r.status_code == 400


def test_download_pack_missing_dir_param(client, file_server):
    r = client.post(
        "/api/downloads",
        json={"url": f"{file_server}/m.safetensors", "kind": "pack"},
    )
    assert r.status_code == 400


def test_download_status_404(client):
    assert client.get("/api/downloads/nope123").status_code == 404


# -- Civitai 판별 (API 모킹) --------------------------------------------------


def test_civitai_kind_auto_maps_type(client, monkeypatch):
    """Civitai LORA 타입 → kind auto가 lora로 판별되어 202를 반환."""
    monkeypatch.setattr(
        downloads, "civitai_resolve", lambda mid, vid, api_key=None: (
            "https://127.0.0.1:9/dl.safetensors", "LORA",
        )
    )
    r = client.post(
        "/api/downloads",
        json={"url": "https://civitai.com/models/1/x", "kind": "auto"},
    )
    # 매핑 성공 → 202 (백그라운드 다운로드는 실패해도 무방)
    assert r.status_code == 202
    assert downloads.CIVITAI_TYPE_MAP["Checkpoint"] == "checkpoint"


def test_civitai_unsupported_type(client, monkeypatch, file_server):
    monkeypatch.setattr(
        downloads, "civitai_resolve", lambda mid, vid, api_key=None: (
            f"{file_server}/m.safetensors", "TextualInversion",
        )
    )
    r = client.post(
        "/api/downloads",
        json={"url": "https://civitai.com/models/1/x", "kind": "auto"},
    )
    assert r.status_code == 400


# -- pack-missing ------------------------------------------------------------


def test_pack_missing(client, tmp_path):
    pack = tmp_path / "models" / "anima38p"
    pack.mkdir(parents=True)
    (pack / "qwen35_4b.safetensors").write_bytes(b"x")
    r = client.get("/api/pack-missing/anima38p")
    assert r.status_code == 200
    body = r.json()
    assert body["pack"] == "anima38p"
    assert body["status"]["qwen35_4b"] is True
    assert body["status"]["dit"] is None  # DiT 없음


def test_pack_missing_404(client):
    assert client.get("/api/pack-missing/nonexistent").status_code == 404


def test_auth_headers():
    from atelier.api.routes.downloads import _auth_headers
    assert _auth_headers(None) == {}
    assert _auth_headers("") == {}
    assert _auth_headers("  ") == {}
    h = _auth_headers("mykey123")
    assert h == {"Authorization": "Bearer mykey123"}


def test_download_request_accepts_api_key():
    from atelier.api.routes.downloads import DownloadRequest
    req = DownloadRequest(url="https://example.com/a.safetensors")
    assert req.civitai_api_key is None
    req2 = DownloadRequest(
        url="https://example.com/a.safetensors", civitai_api_key="k"
    )
    assert req2.civitai_api_key == "k"


def test_civitai_api_key_forwarded_to_resolve(client, monkeypatch):
    """POST의 civitai_api_key가 civitai_resolve에 전달되는지 확인."""
    seen = {}

    def fake_resolve(mid, vid, api_key=None):
        seen["api_key"] = api_key
        return ("https://127.0.0.1:9/dl.safetensors", "LORA")

    monkeypatch.setattr(downloads, "civitai_resolve", fake_resolve)
    r = client.post(
        "/api/downloads",
        json={
            "url": "https://civitai.com/models/1/x",
            "kind": "auto",
            "civitai_api_key": "secret123",
        },
    )
    assert r.status_code == 202
    assert seen.get("api_key") == "secret123"


def test_non_civitai_download_gets_no_key(client, monkeypatch, file_server):
    """Civitai가 아닌 URL에는 API 키를 전달하지 않는다 (유출 방지)."""
    done = threading.Event()
    seen = {}

    def fake_worker(job_id, url, dest_dir, filename, api_key=None):
        seen["api_key"] = api_key
        done.set()

    monkeypatch.setattr(downloads, "_download_worker", fake_worker)
    r = client.post(
        "/api/downloads",
        json={
            "url": f"{file_server}/m.safetensors",
            "kind": "checkpoint",
            "civitai_api_key": "secret123",
        },
    )
    assert r.status_code == 202
    assert done.wait(timeout=5)
    assert seen.get("api_key") is None


def test_civitai_download_gets_key(client, monkeypatch):
    """Civitai URL에는 API 키가 워커까지 전달된다."""
    done = threading.Event()
    seen = {}

    def fake_worker(job_id, url, dest_dir, filename, api_key=None):
        seen["api_key"] = api_key
        done.set()

    monkeypatch.setattr(downloads, "_download_worker", fake_worker)
    monkeypatch.setattr(
        downloads, "civitai_resolve", lambda mid, vid, api_key=None: (
            "https://127.0.0.1:9/dl.safetensors", "LORA",
        )
    )
    r = client.post(
        "/api/downloads",
        json={
            "url": "https://civitai.com/models/1/x",
            "kind": "auto",
            "civitai_api_key": "secret123",
        },
    )
    assert r.status_code == 202
    assert done.wait(timeout=5)
    assert seen.get("api_key") == "secret123"
