#!/usr/bin/env python3
"""릴리즈용 latest.json 생성.

Tauri 업데이터가 읽는 매니페스트. 각 플랫폼의 업데이터 아티팩트(.sig와
짝을 이루는 압축 파일)를 찾아 서명과 다운로드 URL을 적는다.

사용법: python3 scripts/make-latest-json.py <bundles_dir> <tag>
환경 변수: GH_REPO (owner/repo)
"""

import json
import os
import sys
from datetime import datetime, timezone


def detect_platform(archive_name: str) -> str | None:
    n = archive_name.lower()
    if "appimage.tar.gz" in n:
        return "linux-x86_64"
    if "nsis.zip" in n or ("setup" in n and n.endswith(".zip")):
        return "windows-x86_64"
    if "msi.zip" in n:
        return "windows-x86_64"
    if n.endswith(".app.tar.gz"):
        return "darwin-aarch64" if "aarch64" in n else "darwin-x86_64"
    return None


def main() -> None:
    bundles_dir, tag = sys.argv[1], sys.argv[2]
    repo = os.environ["GH_REPO"]
    version = tag.lstrip("v")

    platforms: dict[str, dict[str, str]] = {}
    for root, _, files in os.walk(bundles_dir):
        for f in files:
            if not f.endswith(".sig"):
                continue
            archive = os.path.join(root, f[: -len(".sig")])
            if not os.path.isfile(archive):
                print(f"경고: {f}에 짝이 없어요. 건너뜁니다.")
                continue
            plat = detect_platform(os.path.basename(archive))
            if plat is None:
                print(f"경고: {f}의 플랫폼을 몰라요. 건너뜁니다.")
                continue
            with open(os.path.join(root, f), encoding="utf-8") as fh:
                signature = fh.read().strip()
            url = (
                f"https://github.com/{repo}/releases/download/"
                f"{tag}/{os.path.basename(archive)}"
            )
            platforms[plat] = {"signature": signature, "url": url}
            print(f"{plat}: {os.path.basename(archive)}")

    if not platforms:
        raise SystemExit("업데이터 아티팩트(.sig)를 찾지 못했어요.")

    latest = {
        "version": version,
        "notes": f"Atelier AI {tag}",
        "pub_date": datetime.now(timezone.utc).isoformat(),
        "platforms": platforms,
    }
    with open("latest.json", "w", encoding="utf-8") as fh:
        json.dump(latest, fh, indent=2)
    print(f"latest.json 작성됨: version={version}")


if __name__ == "__main__":
    main()
