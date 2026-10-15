#!/usr/bin/env python3
"""Anima 2.9b / 3.8b 체크포인트 검사 스크립트.

safetensors 파일의 헤더만 읽어 텐서 이름·shape·dtype 목록을 덤프합니다.
가중치를 메모리에 올리지 않으므로 수 GB 파일도 가볍게 검사할 수 있습니다.

의존성: 표준 라이브러리만 사용 (torch / safetensors 패키지 불필요).
Python 3.11+ 에서 동작합니다.

사용법:
    python3 scripts/inspect_anima.py /path/to/anima38_dit.safetensors
    python3 scripts/inspect_anima.py *.safetensors --tokenizer-dir ./qwen35_tokenizer
    python3 scripts/inspect_anima.py model.safetensors --json-out dump.json --top 30

출력 결과를 Atelier AI 개발자에게 전달하면 가중치 매핑 설계를 확정할 수 있습니다.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys

# "xxx.12.yyy" 형태의 키에서 블록 인덱스를 찾는 패턴
_BLOCK_RE = re.compile(r"\.(\d+)\.")
# 흔한 dtype 문자 수 (safetensors 헤더의 dtype 문자열)
_DTYPE_SIZES = {
    "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
}


def read_header(path: str) -> dict:
    """safetensors 헤더(JSON)를 파싱해 반환. 텐서 메타만 포함."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("파일이 너무 작아 safetensors가 아닙니다.")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len > 256 * 1024 * 1024:
            raise ValueError(
                f"헤더 크기({header_len})가 비정상적입니다. "
                "safetensors 파일이 맞는지 확인하세요."
            )
        header_raw = f.read(header_len)
        if len(header_raw) < header_len:
            raise ValueError("헤더를 다 읽지 못했습니다. 파일이 잘렸을 수 있습니다.")
    header = json.loads(header_raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def analyze_file(path: str, top: int) -> dict:
    header = read_header(path)
    size = os.path.getsize(path)

    dtypes: collections.Counter = collections.Counter()
    total_params = 0
    # 블록 인덱스 탐지: prefix -> set(index)
    blocks: dict[str, set[int]] = collections.defaultdict(set)
    # 최상위 그룹 카운트: 첫 번째 키 조각 -> 개수
    groups: collections.Counter = collections.Counter()
    # shape 분포 (상위)
    shapes: collections.Counter = collections.Counter()

    for name, meta in header.items():
        dtype = meta.get("data_offsets") and meta.get("dtype", "?") or meta.get("dtype", "?")
        shape = meta.get("shape", [])
        dtypes[dtype] += 1
        total_params += numel(shape)
        shapes[tuple(shape)] += 1

        top_key = name.split(".")[0]
        groups[top_key] += 1

        m = _BLOCK_RE.search(name)
        if m:
            prefix = name[: m.start()]
            blocks[prefix].add(int(m.group(1)))

    block_summary = {
        prefix: {"count": len(idx), "max_index": max(idx), "min_index": min(idx)}
        for prefix, idx in sorted(blocks.items())
    }

    top_shapes = [
        {"shape": list(s), "count": c}
        for s, c in shapes.most_common(top)
    ]

    return {
        "path": path,
        "file_size_mb": round(size / (1024 * 1024), 1),
        "num_tensors": len(header),
        "dtypes": dict(dtypes),
        "total_params_m": round(total_params / 1e6, 1),
        "top_level_groups": dict(groups.most_common(20)),
        "indexed_blocks": block_summary,
        "most_common_shapes": top_shapes,
        "sample_keys": sorted(header.keys())[:10],
    }


def heuristics(summary: dict) -> list[str]:
    """덤프에서 아키텍처 힌트를 추론."""
    hints: list[str] = []
    for prefix, info in summary["indexed_blocks"].items():
        n = info["count"]
        low = prefix.lower()
        if n >= 10 and ("transformer" in low or "dit" in low):
            hints.append(
                f"'{prefix}' 아래에 {n}개 블록(인덱스 {info['min_index']}~{info['max_index']}) — "
                "확산 트랜스포머 본체일 가능성"
            )
        elif "layers" in low and n >= 10:
            hints.append(
                f"'{prefix}' 아래에 {n}개 레이어(인덱스 {info['min_index']}~{info['max_index']}) — "
                "LLM/텍스트 인코더일 가능성"
            )
    groups = summary["top_level_groups"]
    if "model" in groups and any(
        "layers" in p for p in summary["indexed_blocks"]
    ):
        hints.append("model.layers.* 패턴 — Qwen 계열 텍스트 인코더일 가능성")
    if any("connector" in k or "adapter" in k for k in groups):
        hints.append("connector/adapter 관련 키 존재 — 크로스 어댑터 가중치일 가능성")
    return hints


def inspect_tokenizer_dir(path: str) -> dict:
    if not os.path.isdir(path):
        return {"path": path, "exists": False, "files": []}
    files = []
    for root, _dirs, filenames in os.walk(path):
        for fn in sorted(filenames):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, path)
            try:
                sz = os.path.getsize(full)
            except OSError:
                sz = -1
            files.append({"name": rel, "size": sz})
    return {"path": path, "exists": True, "files": files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Anima 체크포인트 safetensors 헤더 검사 (가중치 로드 없음)"
    )
    ap.add_argument(
        "files", nargs="+", help="검사할 .safetensors 파일 (여러 개 가능)"
    )
    ap.add_argument(
        "--tokenizer-dir", default=None,
        help="토크나이저 디렉터리 (예: qwen35_tokenizer/) — 파일 목록만 확인",
    )
    ap.add_argument(
        "--json-out", default=None, help="전체 결과를 JSON 파일로 저장"
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="자주 등장하는 shape를 몇 개까지 보여줄지 (기본 20)",
    )
    args = ap.parse_args(argv)

    result: dict = {"files": [], "tokenizer": None, "hints": []}

    for path in args.files:
        if not os.path.isfile(path):
            print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
            return 2
        try:
            summary = analyze_file(path, args.top)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"[오류] {path}: {e}", file=sys.stderr)
            return 2
        summary["hints"] = heuristics(summary)
        result["files"].append(summary)

        # 사람이 읽기 좋은 요약 출력
        print(f"=== {path} ===")
        print(f"  파일 크기: {summary['file_size_mb']} MB")
        print(f"  텐서 수: {summary['num_tensors']}, "
              f"총 파라미터: 약 {summary['total_params_m']}M")
        print(f"  dtype 분포: {summary['dtypes']}")
        print(f"  최상위 그룹: {summary['top_level_groups']}")
        if summary["indexed_blocks"]:
            print("  인덱스 블록:")
            for prefix, info in summary["indexed_blocks"].items():
                print(f"    - {prefix}: {info['count']}개 "
                      f"(인덱스 {info['min_index']}~{info['max_index']})")
        if summary["hints"]:
            print("  힌트:")
            for h in summary["hints"]:
                print(f"    * {h}")
        print(f"  샘플 키(처음 10개):")
        for k in summary["sample_keys"]:
            print(f"    - {k}")
        print()

    if args.tokenizer_dir:
        tok = inspect_tokenizer_dir(args.tokenizer_dir)
        result["tokenizer"] = tok
        print(f"=== 토크나이저: {args.tokenizer_dir} ===")
        if not tok["exists"]:
            print("  디렉터리가 없습니다.")
        else:
            for f in tok["files"]:
                print(f"  - {f['name']} ({f['size']} bytes)")
        print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
