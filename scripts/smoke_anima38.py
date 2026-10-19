#!/usr/bin/env python3
"""Anima 3.8b 팩 GPU 실측 스모크 테스트.

포드(RunPod 등 24GB+ GPU)에서 실행:
    backend/.venv/bin/python scripts/smoke_anima38.py /root/.atelier-ai/models/anima38/<팩이름>

단계별로 진단 정보를 출력하므로, 실패 시 출력 전체를 그대로 전달하면
원인 파악이 가능하다.
"""

import argparse
import os
import sys
import traceback

# repo root 기준 backend/ 를 sys.path에 (scripts/에서 실행 가정)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "backend"))


def stage(title):
    print(f"\n===== [단계] {title} =====", flush=True)


def ok(msg):
    print(f"  [OK] {msg}", flush=True)


def warn(msg):
    print(f"  [경고] {msg}", flush=True)


def fail(msg):
    print(f"  [실패] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Anima 3.8b 실측 스모크 테스트")
    ap.add_argument("pack_dir", help="3.8b 팩 디렉토리 (models/anima38/<팩이름>/)")
    ap.add_argument("--steps", type=int, default=4, help="스모크 생성 스텝 수 (기본 4)")
    ap.add_argument("--size", type=int, default=512, help="스모크 생성 해상도 (기본 512)")
    ap.add_argument("--out", default=None, help="출력 디렉토리 (기본 <pack_dir>/smoke_out)")
    args = ap.parse_args()

    pack_dir = os.path.abspath(args.pack_dir)
    out_dir = args.out or os.path.join(pack_dir, "smoke_out")
    overall_ok = True

    # -- 단계 1: GPU -----------------------------------------------------
    stage("1. GPU 확인")
    try:
        import torch

        if not torch.cuda.is_available():
            fail("CUDA 사용 불가 — 3.8b는 GPU 필수예요.")
            return 1
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            ok(f"GPU{i}: {props.name} / VRAM {total_gb:.1f}GB")
            if total_gb < 20:
                warn("VRAM 20GB 미만 — 3.8b 전체 로드는 21~25GB가 필요해요.")
    except Exception:
        fail("torch 로드 실패")
        print(traceback.format_exc())
        return 1

    # -- 단계 2: 팩 구성 ---------------------------------------------------
    stage("2. 팩 파일 확인")
    try:
        from atelier.engine.anima38 import pack_status

        st = pack_status(pack_dir)
        for key in ("dit", "qwen35_4b", "qwen3_06b", "adapter"):
            v = st[key]
            (ok if v else fail)(f"{key}: {v if v else '없음'}")
            if not v and key != "adapter":
                overall_ok = False
        toks = st["tokenizer_dirs"] or ["없음"]
        ok(f"tokenizer_dirs: {', '.join(toks)}")
        if len(st["tokenizer_dirs"]) < 2:
            warn("토크나이저 디렉토리가 2개 미만 — 0.6B용/4B용이 각각 필요해요.")
        ok(f"vae/: {'있음' if st['vae_dir'] else '없음 (vae_path 지정 필요)'}")
        if not overall_ok:
            fail("필수 팩 파일이 부족해요. 단계를 중단합니다.")
            return 1
    except Exception:
        fail("팩 상태 확인 실패")
        print(traceback.format_exc())
        return 1

    # -- 단계 3: 파이프라인 로드 -------------------------------------------
    stage("3. 파이프라인 로드 (DiT/TE/커넥터/VAE)")
    try:
        from atelier.engine.anima38 import Anima38Pipeline

        pipe = Anima38Pipeline(pack_dir, output_dir=out_dir)
        pipe.load()
        ok(f"로드 완료 (device={pipe.device}, dtype={pipe.torch_dtype})")

        missing = getattr(pipe, "_dit_missing", None) or []
        unexpected = getattr(pipe, "_dit_unexpected", None) or []
        if missing:
            warn(f"DiT missing 키 {len(missing)}개 (처음 20개):")
            for k in missing[:20]:
                print(f"    - {k}")
        else:
            ok("DiT missing 키 없음")
        if unexpected:
            warn(f"DiT unexpected 키 {len(unexpected)}개 (처음 20개):")
            for k in unexpected[:20]:
                print(f"    + {k}")
        else:
            ok("DiT unexpected 키 없음")

        conn = getattr(pipe, "_connector_report", None) or {}
        if conn.get("missing") or conn.get("unexpected"):
            warn(f"커넥터 매핑 이슈: {conn}")
        else:
            ok(f"커넥터: {conn.get('note', '어댑터 파일 적용')}")
    except Exception as e:
        fail(f"로드 실패: {e}")
        print(traceback.format_exc())
        return 1

    # -- 단계 4: 스모크 생성 -----------------------------------------------
    stage(f"4. 스모크 생성 ({args.size}x{args.size}, {args.steps}스텝)")
    try:
        paths = pipe.txt2img({
            "prompt": "1girl, white hair, smile, upper body",
            "negative_prompt": "lowres, blurry",
            "height": args.size,
            "width": args.size,
            "steps": args.steps,
            "cfg_scale": 4.0,
            "seed": 42,
            "batch_size": 1,
            "sampler": "euler_ancestral",
        })
        for p in paths:
            ok(f"생성됨: {p}")
    except Exception as e:
        fail(f"생성 실패: {e}")
        print(traceback.format_exc())
        return 1

    print("\n===== 실측 스모크 테스트 완료 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
