"""모델 다운로드 API (URL 붙여넣기 방식, SwarmUI식).

- POST /api/downloads: {url, kind, pack_dir?} → 백그라운드 다운로드 시작
- GET /api/downloads/{job_id}: 진행 상태 (프론트가 1초 간격 폴링)
- GET /api/pack-missing/{pack_name}: pack_status() 얇은 래퍼

지원 URL:
- Civitai 모델 페이지(civitai.com/models/<id>): API로 downloadUrl·타입 조회
- HuggingFace (blob/resolve): resolve로 변환해 직접 다운로드
- 그 외 직접 URL: 그대로 스트리밍 다운로드

저장 위치: checkpoint→models_dir()/, lora→models_dir()/lora/,
pack→models_dir()/<pack_dir>/ (기존 팩 디렉토리).
다운로드는 .part 임시 파일에 쓰고 완료 후 원자적 rename.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atelier.api.routes.models import _resolve_dir, models_dir

router = APIRouter()

ALLOWED_EXTS = (".safetensors", ".ckpt", ".pt")
KINDS = ("auto", "checkpoint", "lora", "pack")

CIVITAI_API = "https://civitai.com/api/v1"
#: Civitai 모델 타입 → 내부 kind
CIVITAI_TYPE_MAP = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "LoCon": "lora",
    "DoRA": "lora",
}

# job_id → {id, status, filename, downloaded_bytes, total_bytes, error}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "downloading",
            "filename": filename,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "error": None,
        }
    return job_id


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def classify_url(url: str) -> tuple[str, dict]:
    """URL 종류 판별 → (source, info).

    source: "civitai" | "huggingface" | "direct".
    http/https가 아니면 400.
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(
            status_code=400, detail="http/https URL만 받을 수 있어요."
        )
    host = p.hostname.lower()
    if "civitai.com" in host:
        m = re.search(r"/models/(\d+)", p.path)
        if not m:
            raise HTTPException(
                status_code=400,
                detail="Civitai 모델 페이지 URL이 아니에요.",
            )
        version_id = parse_qs(p.query).get("modelVersionId", [None])[0]
        return "civitai", {
            "model_id": m.group(1),
            "version_id": version_id,
        }
    if host in ("huggingface.co", "www.huggingface.co") or host.endswith(
        ".huggingface.co"
    ):
        # /blob/ → /resolve/ 변환
        path = re.sub(r"/blob/", "/resolve/", p.path, count=1)
        return "huggingface", {
            "direct_url": f"{p.scheme}://{p.hostname}{path}",
        }
    return "direct", {"direct_url": url}


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Civitai API 키가 있으면 Bearer 헤더 생성. 없으면 빈 dict."""
    key = (api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def civitai_resolve(
    model_id: str, version_id: str | None, api_key: str | None = None
) -> tuple[str, str]:
    """Civitai API로 실제 다운로드 URL과 모델 타입 조회.

    → (download_url, model_type). version_id가 없으면 최신 버전.
    """
    try:
        r = httpx.get(
            f"{CIVITAI_API}/models/{model_id}",
            headers=_auth_headers(api_key),
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - 네트워크 실패는 400으로
        raise HTTPException(
            status_code=400,
            detail=f"Civitai 정보를 가져오지 못했어요: {e}",
        )
    versions = data.get("modelVersions") or []
    ver = None
    if version_id:
        ver = next(
            (v for v in versions if str(v.get("id")) == str(version_id)),
            None,
        )
    if ver is None and versions:
        ver = versions[0]
    if not ver or not ver.get("downloadUrl"):
        raise HTTPException(
            status_code=400,
            detail="다운로드 가능한 버전을 찾지 못했어요.",
        )
    return ver["downloadUrl"], data.get("type", "")


def safe_filename(name: str) -> str:
    """파일명 안전화. 경로 조작·이상한 확장자를 차단한다."""
    name = unquote(name or "")
    name = os.path.basename(name.replace("\\", "/")).strip().strip(".")
    if not name or name in (".", ".."):
        raise HTTPException(
            status_code=400, detail="파일명을 알 수 없어요."
        )
    if os.path.splitext(name)[1].lower() not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"모델 파일({', '.join(ALLOWED_EXTS)})만 받을 수 있어요.",
        )
    return name


def filename_from_url(url: str) -> str:
    """URL 끝부분에서 파일명 추출."""
    p = urlparse(url)
    return safe_filename(os.path.basename(p.path.rstrip("/")))


def _filename_from_content_disposition(header: str) -> str | None:
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", header or "")
    if not m:
        return None
    try:
        return safe_filename(m.group(1))
    except HTTPException:
        return None


def _download_worker(
    job_id: str, url: str, dest_dir: str, filename: str,
    api_key: str | None = None,
) -> None:
    """백그라운드 다운로드. .part에 쓰고 완료 후 rename."""
    part = os.path.join(dest_dir, filename + ".part")
    final = os.path.join(dest_dir, filename)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        with httpx.stream(
            "GET", url, headers=_auth_headers(api_key),
            timeout=60, follow_redirects=True,
        ) as r:
            r.raise_for_status()
            # Content-Disposition에 더 정확한 파일명이 있으면 교체
            better = _filename_from_content_disposition(
                r.headers.get("content-disposition", "")
            )
            if better and better != filename:
                filename = better
                part = os.path.join(dest_dir, filename + ".part")
                final = os.path.join(dest_dir, filename)
                _set_job(job_id, filename=filename)
            total = r.headers.get("content-length")
            _set_job(
                job_id,
                total_bytes=int(total)
                if total and total.isdigit()
                else None,
            )
            downloaded = 0
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    _set_job(job_id, downloaded_bytes=downloaded)
        os.replace(part, final)
        _set_job(
            job_id,
            status="done",
            downloaded_bytes=os.path.getsize(final),
        )
    except Exception as e:  # noqa: BLE001 - 실패는 job에 기록
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        _set_job(job_id, status="error", error=str(e)[:300])


class DownloadRequest(BaseModel):
    url: str
    kind: str = "auto"  # auto | checkpoint | lora | pack
    pack_dir: str | None = None  # kind=pack일 때 기존 팩 디렉토리명
    civitai_api_key: str | None = None  # Civitai 다운로드용 본인 API 키


@router.post("/api/downloads", status_code=202)
def start_download(req: DownloadRequest):
    """URL 다운로드 시작. 백그라운드 스레드로 받고 {job_id} 반환."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(
            status_code=400, detail="URL을 입력해주세요."
        )
    kind = (req.kind or "auto").strip().lower()
    if kind not in KINDS:
        raise HTTPException(
            status_code=400,
            detail="kind는 auto/checkpoint/lora/pack 중 하나예요.",
        )
    source, info = classify_url(url)

    if source == "civitai":
        dl_url, model_type = civitai_resolve(
            info["model_id"], info["version_id"],
            api_key=req.civitai_api_key,
        )
        if kind == "auto":
            mapped = CIVITAI_TYPE_MAP.get(model_type)
            if not mapped:
                raise HTTPException(
                    status_code=400,
                    detail=f"Civitai 모델 타입 '{model_type}'은 "
                           "받을 수 없어요.",
                )
            kind = mapped
        url = dl_url
    else:
        url = info["direct_url"]

    try:
        filename = filename_from_url(url)
    except HTTPException:
        # Content-Disposition에서 살리기를 시도 (워커가 갱신)
        filename = "download.safetensors"

    if kind == "auto":
        kind = "lora" if "lora" in filename.lower() else "checkpoint"

    if kind == "pack":
        if not req.pack_dir:
            raise HTTPException(
                status_code=400,
                detail="pack에는 pack_dir(팩 디렉토리명)이 필요해요.",
            )
        dest_dir = _resolve_dir(req.pack_dir)  # 경로 검증 + 존재 확인
    elif kind == "lora":
        dest_dir = os.path.join(models_dir(), "lora")
    else:
        dest_dir = models_dir()

    job_id = _new_job(filename)
    # API 키는 Civitai 다운로드에만 사용 — 다른 호스트로 키 유출 방지
    worker_key = req.civitai_api_key if source == "civitai" else None
    thread = threading.Thread(
        target=_download_worker,
        args=(job_id, url, dest_dir, filename, worker_key),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@router.get("/api/downloads/{job_id}")
def download_status(job_id: str):
    """다운로드 진행 상태."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="다운로드 작업을 찾을 수 없어요."
        )
    return dict(job)


@router.get("/api/pack-missing/{pack_name:path}")
def pack_missing(pack_name: str):
    """팩 구성 요소 존재 여부 (pack_status 얇은 래퍼)."""
    from atelier.engine.anima38 import pack_status

    path = _resolve_dir(pack_name)
    return {"pack": pack_name, "status": pack_status(path)}
