"""배치 생성 큐 엔진.

프롬프트 목록을 받아 순서대로 txt2img를 실행한다.

- 동시에 실행되는 배치 job은 1개로 제한한다 (GPU 1개 기준 순차 처리).
  실행 중일 때 새 요청은 409로 거절한다.
- 각 아이템은 queued/running/done/error/cancelled 상태.
- 취소는 다음 아이템 시작 전에 중단 플래그를 확인한다.
- 시드가 고정값(>=0)이면 항목마다 +1씩 증가시킨다.
  랜덤(음수/None)이면 파이프라인이 매번 새로 뽑는다.

torch/diffusers는 여기서 직접 import하지 않는다 (파이프라인이 lazy import).
"""

from __future__ import annotations

import os
import threading
import time
import uuid

# 동시에 실행되는 배치 job은 1개로 제한 (GPU 메모리 보호).
_batch_run_lock = threading.Lock()

# job_id -> BatchJob. 서버 재시작 시 사라진다 (generate.py와 동일, MVP).
_jobs: dict[str, "BatchJob"] = {}

# 완료된 job 보관 시간 (1시간). 무제한 증가로 인한 메모리 누수 방지.
JOB_TTL_SECONDS = 3600

# 한 배치당 최대 프롬프트 수.
MAX_PROMPTS = 200


class BatchBusyError(RuntimeError):
    """다른 배치 작업이 실행 중일 때 발생."""


class BatchJob:
    """배치 생성 작업. 별도 스레드에서 순차 실행되며 상태를 기록한다."""

    def __init__(
        self,
        job_id: str,
        prompts: list[str],
        params: dict,
        loras: dict | None,
    ):
        self.job_id = job_id
        self.prompts = prompts
        self.params = params
        self.loras = loras
        self.status = "queued"  # queued/running/done/cancelled/error
        self.created_at = time.time()
        self.cancel_requested = False
        self.error: str | None = None
        self.items = [
            {"prompt": p, "status": "queued", "url": None, "error": None}
            for p in prompts
        ]

    @property
    def done_count(self) -> int:
        return sum(
            1 for i in self.items
            if i["status"] in ("done", "error", "cancelled")
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": len(self.items),
            "done": self.done_count,
            "created_at": self.created_at,
            "error": self.error,
            "items": [dict(i) for i in self.items],
        }


def _run_batch_sync(job: BatchJob, pipe) -> None:
    """배치 실행 (별도 스레드). generate.py의 _run_job_sync 패턴을 따른다."""
    # 순환 import를 피하려고 함수 안에서 가져온다.
    from atelier.api.routes import history as history_module

    job.status = "running"
    seed = job.params.get("seed")
    # bool은 int의 서브클래스라 제외 (True == 1 오판 방지)
    fixed_seed = (
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
    )
    try:
        for idx, item in enumerate(job.items):
            if job.cancel_requested:
                for rest in job.items[idx:]:
                    rest["status"] = "cancelled"
                job.status = "cancelled"
                return
            item["status"] = "running"
            try:
                p = dict(job.params)
                p["prompt"] = item["prompt"]
                if fixed_seed:
                    # 시드 고정 시 전부 같은 그림이 나오지 않게 항목마다 증가
                    p["seed"] = seed + idx
                paths = pipe.txt2img(p, job.loras)
                urls = ["/api/results/" + os.path.basename(x) for x in paths]
                item["status"] = "done"
                item["url"] = urls[0] if urls else None
                try:
                    history_module.append_record(
                        history_module.build_record(
                            f"{job.job_id}-{idx}", "txt2img",
                            p, urls, job.loras,
                        )
                    )
                except Exception as e:  # noqa: BLE001 - 기록 실패가 배치를 망치면 안 됨
                    print(f"[atelier] 배치 작업내역 저장 실패: {e}")
            except Exception as e:  # noqa: BLE001 - 한 항목 실패해도 다음 항목 계속
                item["status"] = "error"
                item["error"] = str(e)
        job.status = "cancelled" if job.cancel_requested else "done"
    except Exception as e:  # noqa: BLE001 - 예상 밖 오류는 job 단위로 기록
        job.status = "error"
        job.error = str(e)
    finally:
        _batch_run_lock.release()


def start_job(
    prompts: list[str],
    params: dict,
    loras: dict | None,
    pipe,
) -> BatchJob:
    """배치 작업을 시작한다. 실행 중인 배치가 있으면 BatchBusyError."""
    if not _batch_run_lock.acquire(blocking=False):
        raise BatchBusyError("다른 배치 작업이 실행 중이에요.")
    _prune_old_jobs()
    job = BatchJob(uuid.uuid4().hex, prompts, params, loras)
    _jobs[job.job_id] = job
    thread = threading.Thread(
        target=_run_batch_sync,
        args=(job, pipe),
        daemon=True,
        name=f"atelier-batch-{job.job_id[:8]}",
    )
    thread.start()
    return job


def get_job(job_id: str) -> BatchJob | None:
    return _jobs.get(job_id)


def cancel_job(job_id: str) -> BatchJob | None:
    """취소 요청. 다음 아이템 시작 전에 중단된다. 없는 id면 None."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    if job.status in ("queued", "running"):
        job.cancel_requested = True
    return job


def _prune_old_jobs() -> None:
    """끝난 지 오래된 job을 정리해 메모리 누수를 방지한다."""
    now = time.time()
    stale = [
        jid for jid, job in _jobs.items()
        if job.status in ("done", "cancelled", "error")
        and now - job.created_at > JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]
